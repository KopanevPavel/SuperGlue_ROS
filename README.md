# SuperGlue ROS
ROS wrapper for SuperGlue and SuperPoint models

 - SuperGlue: [paper](https://arxiv.org/abs/1911.11763)
 - SuperPoint: [paper](https://arxiv.org/abs/1712.07629)

You can find **the utilization of this module** in [this](https://github.com/KopanevPavel/runbot_custom_localization) repository in the frontend part of the visual-inertial SLAM system

## System
 - ROS Melodic (with python3 support)
 - Pytorch
 - Numpy
 - OpenCV
 - CUDA (highly recommended)

## Usage

Download and build:
```sh
mkdir catkin_ws
cd catkin_ws
mkdir src
cd src
git clone --recursive https://github.com/KopanevPavel/SuperGlue_ROS
cd ..
catkin build
source devel/setup.bash
```

Add to python path:
```sh
export PYTHONPATH=$PYTHONPATH:/home/jetson/SLAM/runbot_custom_localization/frontend/SuperGlue_ROS/src/SuperGlue_ROS
export PYTHONPATH=$PYTHONPATH:/home/jetson/SLAM/runbot_custom_localization/frontend/SuperGlue_ROS/src/SuperGlue_ROS/models/SuperGluePretrainedNetwork
export PYTHONPATH=$PYTHONPATH:/home/jetson/SLAM/runbot_custom_localization/frontend/SuperGlue_ROS/src/SuperGlue_ROS/models/SuperPointPretrainedNetwork
```

Run SuperPoint detector:
```sh
rosrun SuperGlue_ROS detector_node
```

Run SuperGlue matcher:
```sh
rosrun SuperGlue_ROS matcher_node
```

*The matcher_node node publishes matching result as string message transformed via json*

*Subsciber example (string -> numpy array) could be found in vo_node node*


Credits:
https://github.com/Shiaoming/Python-VO

