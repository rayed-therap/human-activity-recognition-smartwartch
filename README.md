# Human activity recognition (HAR) using smartwatch data

## Dataset
The dataset is the [1] [WISDM Smartphone and Smartwatch Activity and Biometrics Dataset Data Set](https://archive.ics.uci.edu/ml/datasets/WISDM+Smartphone+and+Smartwatch+Activity+and+Biometrics+Dataset+) which contains accelerometer and gyroscope time-series sensor data collected from a smartphone and smartwatch as 51 test subjects perform 18 activities for 3 minutes each. **Only smartwatch readings are used in this project.**

<img src='images/accel_xyz_1601.svg'>

## Process
1. First the sensor readings are read from their corresponding text files and merged based on subject, activity, and timestamp.
2. Then a sliding window of 60 readings (corresponding to 3 secs since the data was sampled at 20 Hz) was taken along with the 6 sensor readings (3 axes for each sensor) to create sequences of shape (60, 6).
3. Finally, a LSTM model with 1D convolution was trained that yields 90% test accuracy.

<img src='images/epoch_accuracy.svg'>

## Environment
Use the ```requirements.txt``` file to create a conda environment that will install all necessary packages. Then step through the ```har.ipynb``` to replicate this result.

## References
1. Smartphone and Smartwatch-Based Biometrics Using Activities of Daily Living. IEEE Access, 7:133190-133202, Sept. 2019.
