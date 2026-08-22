FORS-EMG: A Novel sEMG Dataset for Hand Gesture Recognition Across Different Forearm Orientations

This dataset serves as a comprehensive resource for developing robust machine-learning classification algorithms and hand gesture recognition applications. The dataset's key features are summarized as follows:

1. The dataset was collected from 19 able-bodied subjects, who performed 12 distinct finger and wrist gestures across three forearm orientations: supination, neutral (rest), and pronation. Each gesture was repeated five times. Additionally, two electrode placement positions were used during sEMG signal recording: near the elbow and on the forearm.

2. The sEMG data for all hand gestures with different forearm orientations were recorded over eight-second intervals. The recordings were taken from eight channels—four positioned near the elbow and four along the mid-forearm—and sampled at 985 Hz.

3. The sEMG data were recorded using MATLAB 2020a (MathWorks, USA) in .mat format, providing users with direct access to the sEMG recordings in physical units, eliminating the need for additional conversion before signal preprocessing.

4. The dataset is organized into 19 folders, each corresponding to 19 subject. Each subject folder contains three subfolder which corresponds to three forearm orientations (pronation, rest, supination). The raw data files of all hand gestures of all trials are in the subfolder named with considered forearm orientation. The naming convention of raw sEMG data of specific gestures is gesture_name-trial.mat. For example, Thumb_UP-4.mat contains the data performing Thumb_UP (TU) gesture of the fourth trial.

5. Additional folders are provided including supplementary files, such as participant physical measurements, electrode attachment instructions, machine and electrode configuration details, and gesture sequences with illustrations."