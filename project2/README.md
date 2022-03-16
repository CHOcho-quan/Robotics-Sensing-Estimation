# Project 2 SLAM

- To run the code please download the data in the directory codes/data/
- Directly open the notebook ECE276A_project2.ipynb and run the codes in each cell respectively



The codes are formulated as follow

- sensors
  - Camera.py : Sensor class dealing with camera inputs
  - Lidar.py : Sensor class dealing with lidar inputs
  - Encoder.py : Sensor class dealing with Encoder inputs
  - FOG.py : Sensor class dealing with gyroscope inputs
- Environment.py : Env class that stores all the data in a pickle and replay
- OccupancyMap.py : Occupancy map class dealing with map log-odds update and map value storation
- Robot.py : Robot class aggregates occupancy map and all the sensor classes together. Particle Filter is implemented here. All the inputs get into robot class, then state estimation and map construction are built inside the class