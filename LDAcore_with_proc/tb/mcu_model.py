"""
mcu_model.py - the microcontroller side of the chip's SPI link, for the cocotb testbenches
(step 3, 1 Nov 2026; step 4 adds REST_SUM_ADDR). The microcontroller is the SPI master: mode 0, MSB first, SCLK 250 kHz by
default (the fastest the chip allows), and the timing rules of emg_spi_slave.v:
CS low -> first SCLK edge 4 us, last SCLK edge -> CS high 2 us, CS high between commands 4 us.

  mcu = Mcu(dut)                    dut has mcu_cs_n, mcu_sclk, mcu_mosi, mcu_miso, mcu_drdy
  await mcu.write(addr, value)      command 0x02
  value = await mcu.read(addr)      command 0x03 (2 address bytes, 1 spare byte, 4 value bytes)
  frame = await mcu.frame(C)        command 0x01, returns the 5C + 6 frame bytes
  expected_frame(count, flags, dead, q)   the bytes the chip must send (golden model format)
"""
import cocotb
from cocotb.triggers import Lock, Timer

FLAG_REST, FLAG_FAULT, FLAG_LINK, FLAG_CFG, FLAG_SKIP, FLAG_VALID, FLAG_OVERRUN = (1 << b for b in range(7))
STATUS_ADDR = 0x7FA
REST_SUM_ADDR = 0x7FB          # rest sum of the last published frame (step 4)


def flags_byte(fault=0, link=0, cfg=0, skipped=0, valid=0, overrun=0, rest=0):
    return (rest * FLAG_REST | fault * FLAG_FAULT | link * FLAG_LINK | cfg * FLAG_CFG
            | skipped * FLAG_SKIP | valid * FLAG_VALID | overrun * FLAG_OVERRUN)


def checksum(body):
    return sum(body) & 0xFF


def expected_frame(count, flags, dead, q):
    """Frame bytes: counter, flags, 3-byte dead mask (channel 0 = byte 2 bit 0), the 5C features
    (signed, as bytes), checksum of all bytes before it."""
    mask = sum(1 << c for c, d in enumerate(dead) if d)
    body = [count & 0xFF, flags, mask & 0xFF, (mask >> 8) & 0xFF, (mask >> 16) & 0xFF] + [int(v) & 0xFF for v in q]
    return body + [checksum(body)]


class Mcu:
    def __init__(self, dut, sclk_khz=250.0):
        self.dut = dut
        self.half_ns = 1e6 / sclk_khz / 2
        self.lock = Lock()
        self.x_bits = 0                     # MISO bits that were X or Z (should stay 0)
        dut.mcu_cs_n.value, dut.mcu_sclk.value, dut.mcu_mosi.value = 1, 0, 0

    async def xfer(self, tx, half_ns=None):
        """One command: CS low, the bytes of tx out on MOSI, the same number of bytes in on MISO."""
        half = round(self.half_ns if half_ns is None else half_ns)       # whole ns
        d = self.dut
        async with self.lock:
            d.mcu_cs_n.value = 0
            await Timer(4, unit="us")
            rx = []
            for byte in tx:
                r = 0
                for i in range(8):
                    d.mcu_mosi.value = (byte >> (7 - i)) & 1
                    await Timer(half, unit="ns")
                    v = str(d.mcu_miso.value)                     # sampled at the rising edge
                    if v not in ("0", "1"):                       # X/Z: count it, read as 0, so the
                        self.x_bits += 1                          # test can report the real cause
                        v = "0"
                    r = (r << 1) | int(v)
                    d.mcu_sclk.value = 1
                    await Timer(half, unit="ns")
                    d.mcu_sclk.value = 0
                rx.append(r)
            await Timer(2, unit="us")
            d.mcu_cs_n.value = 1
            d.mcu_mosi.value = 0
            await Timer(4, unit="us")
        return rx

    async def write(self, addr, value):
        v = int(value) & 0xFFFFFFFF
        await self.xfer([0x02, (addr >> 8) & 0x0F, addr & 0xFF] + [(v >> s) & 0xFF for s in (24, 16, 8, 0)])

    async def read(self, addr):
        rx = await self.xfer([0x03, (addr >> 8) & 0x0F, addr & 0xFF, 0, 0, 0, 0, 0])
        return int.from_bytes(bytes(rx[4:8]), "big")

    async def frame(self, C, half_ns=None):
        rx = await self.xfer([0x01] + [0] * (5 * C + 6), half_ns)
        return rx[1:]

    async def status(self):
        return await self.read(STATUS_ADDR)
