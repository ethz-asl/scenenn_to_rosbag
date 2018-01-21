# Interfacing SceneNN and ROS
Tools for publishing the SceneNN datasets via ROS messages.

## How to use these tools
1. Follow the instructions in [scenenn](https://github.com/scenenn/scenenn) and download the SceneNN data.
Your folder structure should be at the and as follows:

```
scenenn_data
├── 123
│   ├── depth
│   │   ├── depth00001.png
│   │   ├── ...
│   ├── image
│   │   ├── image00001.png
│   │   ├── ...
│   ├── mask
│   │   ├── mask00001.png
│   │   ├── ...
│   ├── timestamp.txt
│   ├── trajectory.log
│   ├── ...
└── intrinsic
    ├──  asus.ini
    ├──  kinect2.ini
```

2. Clone this repository to the `src` folder of your catkin workspace, build your workspace and source it.
3. Run the script to write the SceneNN data to a rosbag as a ROS node with the following command:

```bash
$ rosrun scenenn_ros_tools scenenn_to_rosbag.py -scenenn_data_folder /PATH/TO/SCENENN_DATA -scene_id SCENE_ID_TO_READ -output_bag SCENENN.BAG
```

For example:
```bash
$ rosrun scenenn_ros_tools scenenn_to_rosbag.py -scenenn_data_folder scenenn_data -scene_id 066 -output_bag scenenn_066.bag
```
