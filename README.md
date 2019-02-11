# Interfacing SceneNN and ROS
Tools for working with the [SceneNN](http://people.sutd.edu.sg/~saikit/projects/sceneNN/) dataset and converting it to a ROS bag.

## How to use these tools
1. Follow the instructions in the [scenenn repository](https://github.com/scenenn/scenenn) and download the SceneNN data.
Your `scenenn_data` folder structure should be at the end as follows:

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

    ```bash
    cd <catkin_ws>/src
    git clone git@github.com:ethz-asl/scenenn_ros_tools.git
    catkin build
    source <catkin_ws>/devel/setup.bash
    ```

3. Make the Python script executable and run it as a ROS node to write the SceneNN data to a rosbag.

    ```bash
    cd scenenn_ros_tools && chmod +x nodes/scenenn_to_rosbag.py
    rosrun scenenn_ros_tools scenenn_to_rosbag.py -scenenn_data_folder PATH/TO/scenenn_data -scene_id SCENE_ID -to_frame TO_FRAME -frame_step FRAME_STEP -output_bag OUTPUT_BAG
    ```

    For example:
    ```bash
    rosrun scenenn_ros_tools scenenn_to_rosbag.py -scenenn_data_folder ../../../scenenn/download/scenenn_data/ -scene_id 066 -frame_step 1 -output_bag scenenn_066.bag
    ```
