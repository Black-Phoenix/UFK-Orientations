# Orientation estimation
![Roll 1](./imgs/4_dataset3/roll.png)
![Pitch 1](./imgs/4_dataset3/pitch.png)
![Roll 1](./imgs/4_dataset3/yaw.png)
## Overview
This repostory estimates the orientation (roll, pitch and yaw) given IMU data using a unscented kalman filter. Ground truth was found out using a vicon mocap setup.
## Dependencies
* Numpy 
* Scipy
* Matplotlib
## Approach
The main approach taken can be found in [this paper](https://kodlab.seas.upenn.edu/uploads/Arun/UKFpaper.pdf). 2 approaches were taken to the state size, a state with 7 elements  (4 for orientation and 3 for angular velocity) and a state with 4 elements (for orientation). The 7 state version still has to be tuned more but the 4 state has been tuned for the given datasets. 

## Results
Below are some results using the first dataset. The roll and pitch are on point, while the yaw drifts. This is a limitation with estimating yaw using only an IMU. 

|       | 7 state                          | 4 state                          |
| ----- | -------------------------------- | -------------------------------- |
| Roll  | ![](./imgs/7_dataset1/roll.png)  | ![](./imgs/4_dataset1/roll.png)  |
| Pitch | ![](./imgs/7_dataset1/pitch.png) | ![](./imgs/4_dataset1/pitch.png) |
| Yaw   | ![](./imgs/7_dataset1/yaw.png)   | ![](./imgs/4_dataset1/yaw.png)   |


