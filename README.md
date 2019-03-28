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

3. Make the Python script executable and run it as a ROS node to convert data from a SceneNN scene to a rosbag. The rosbag will contain a sequence of RGB and depth images, ground truth 2D instance label images, and relative transforms. Optionally, it can contain colorized ground truth 2D instance label images, colored pointclouds of the scene, and colored pointclouds of ground truth instance segments.

    ```bash
    cd scenenn_ros_tools && chmod +x nodes/scenenn_to_rosbag.py
    rosrun scenenn_ros_tools scenenn_to_rosbag.py --scenenn-path PATH/TO/scenenn_data --scene-id ID [--limit NUM] [--output-bag NAME]
    ```

    For example:
    ```bash
    rosrun scenenn_ros_tools scenenn_to_rosbag.py --scenenn-path ../../../scenenn/download/scenenn_data/ --scene-id 066 --output-bag scenenn_066.bag
    ```
    The output bag contains the following topics:
      ```bash
      # RGB and depth images
      /camera/rgb/camera_info         : sensor_msgs/CameraInfo
      /camera/rgb/image_raw           : sensor_msgs/Image
      /camera/depth/camera_info       : sensor_msgs/CameraInfo
      /camera/depth/image_raw         : sensor_msgs/Image        

      # Ground truth 2D instance segmentation image
      /camera/instances/image_raw     : sensor_msgs/Image

      # Ground truth colorized 2D instance segmentation image [Disabled by default]
      /camera/instances/image_rgb     : sensor_msgs/Image

      # Colored pointclouds of ground truth instance segments [Disabled by default]
      /scenenet_node/object_segment   : sensor_msgs/PointCloud2

      # Colored pointcloud of the scene                       [Disabled by default]
      /scenenn_node/scene            : sensor_msgs/PointCloud2

      # Transform from /scenenn_camera_frame to /world
      /tf                             : tf/tfMessage
      ```
