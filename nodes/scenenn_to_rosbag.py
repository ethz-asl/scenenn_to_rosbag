#!/usr/bin/env python
import os
import sys
import argparse
import numpy as np
import rosbag
import rospy
import cv2

from cv_bridge import CvBridge
from geometry_msgs.msg import Point32, TransformStamped
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header
from tf.msg import tfMessage
import sensor_msgs.point_cloud2 as pc2
import tf


def rgb_path_from_frame(scenenn_path, scene, frame):
    scene_path = os.path.join(scenenn_path, scene)
    image_path = os.path.join(scene_path, 'image')
    return os.path.join(image_path, 'image{0:05d}.png'.format(frame))


def depth_path_from_frame(scenenn_path, scene, frame):
    scene_path = os.path.join(scenenn_path, scene)
    image_path = os.path.join(scene_path, 'depth')
    return os.path.join(image_path, 'depth{0:05d}.png'.format(frame))


def instance_path_from_frame(scenenn_path, scene, frame):
    scene_path = os.path.join(scenenn_path, scene)
    image_path = os.path.join(scene_path, 'mask')
    return os.path.join(image_path, 'mask_image{0:05d}.png'.format(frame))


def timestamp_path(scenenn_path, scene):
    scene_path = os.path.join(scenenn_path, scene)
    return os.path.join(scene_path, 'timestamp.txt')


def trajectory_path(scenenn_path, scene):
    scene_path = os.path.join(scenenn_path, scene)
    return os.path.join(scene_path, 'trajectory.log')


def intrinsics_path(scenenn_path):
    intrinsics_path = os.path.join(scenenn_path, 'intrinsic')
    return os.path.join(intrinsics_path, 'asus.ini')


def convert_bgrd_to_pcl(bgr_image, depth_image, camera_model):
    center_x = camera_model.cx()
    center_y = camera_model.cy()

    constant_x = 1 / camera_model.fx()
    constant_y = 1 / camera_model.fy()

    pointcloud_xyzrgb_fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)
    ]

    vs = np.array(
        [(v - center_x) * constant_x for v in range(0, depth_image.shape[1])])
    us = np.array(
        [(u - center_y) * constant_y for u in range(0, depth_image.shape[0])])

    # Convert depth from mm to m.
    depth_image = depth_image / 1000.0

    x = np.multiply(depth_image, vs)
    y = depth_image * us[:, np.newaxis]

    stacked = np.ma.dstack((x, y, depth_image, bgr_image))
    compressed = stacked.compressed()
    pointcloud = compressed.reshape((int(compressed.shape[0] / 6), 6))

    pointcloud = np.hstack((pointcloud[:, 0:3],
                            pack_bgr(*pointcloud.T[3:6])[:, None]))
    pointcloud = [[point[0], point[1], point[2], point[3]]
                  for point in pointcloud]

    pointcloud = pc2.create_cloud(Header(), pointcloud_xyzrgb_fields,
                                  pointcloud)
    return pointcloud


def pack_bgr(blue, green, red):
    # Pack the 3 BGR channels into a single UINT32 field as RGB.
    return np.bitwise_or(
        np.bitwise_or(
            np.left_shift(red.astype(np.uint32), 16),
            np.left_shift(green.astype(np.uint32), 8)), blue.astype(np.uint32))


def pack_bgra(blue, green, red, alpha):
    # Pack the 4 BGRA channels into a single UINT32 field.
    return np.bitwise_or(
        np.left_shift(blue.astype(np.uint32), 24),
        np.bitwise_or(
            np.left_shift(green.astype(np.uint32), 16),
            np.bitwise_or(
                np.left_shift(red.astype(np.uint32), 8), alpha.astype(
                    np.uint32))))


def parse_timestamps(scenenn_path, scene):
    timestamps = {}
    try:
        with open(timestamp_path(scenenn_path, scene)) as fileobject:
            for line in fileobject:
                ws = line.split()
                timestamps[int(ws[0])] = int(ws[1])
    except IOError:
        print('SceneNN timestamp data not found at location:{0}'.format(
            timestamp_path(scenenn_path, scene)))
        sys.exit('Please ensure you have downloaded the timestamp data.')

    return timestamps


def parse_trajectory(scenenn_path, scene):
    trajectory = {}
    try:
        with open(trajectory_path(scenenn_path, scene)) as fileobject:
            while 1:
                R = np.diag(np.ones(4))
                frame_line = fileobject.readline()
                if not frame_line:
                    break
                frame = int(frame_line.split()[0])
                R[0, :] = fileobject.readline().split()
                R[1, :] = fileobject.readline().split()
                R[2, :] = fileobject.readline().split()
                R[3, :] = fileobject.readline().split()
                trajectory[frame] = R
    except IOError:
        print('SceneNN trajectory data not found at location:{0}'.format(
            trajectory_path(scenenn_path, scene)))
        sys.exit('Please ensure you have downloaded the trajectory data.')
    return trajectory


def parse_intrinsics(scenenn_path):
    intrinsics = {}
    try:
        with open(intrinsics_path(scenenn_path)) as fileobject:
            for line in fileobject:
                ws = line.split()
                intrinsics[ws[0]] = float(ws[1])
    except IOError:
        print('SceneNN intrinsics data not found at location: {0}'.format(
            intrinsics_path(scenenn_path)))
        sys.exit('Please ensure you have downloaded the intrinsics data.')
    return intrinsics


def camera_intrinsic_transform(intrinsics):
    pixel_width = intrinsics['depth_width']
    pixel_height = intrinsics['depth_height']

    fx = intrinsics['fx']
    fy = intrinsics['fy']

    pixel_width = intrinsics['depth_width']
    pixel_height = intrinsics['depth_height']

    camera_intrinsics = np.zeros((3, 4))
    camera_intrinsics[2, 2] = 1.0
    camera_intrinsics[0, 0] = fx
    camera_intrinsics[0, 2] = pixel_width / 2.0
    camera_intrinsics[1, 1] = fy
    camera_intrinsics[1, 2] = pixel_height / 2.0
    return camera_intrinsics


def parse_camera_info(scenenn_path):
    intrinsics = parse_intrinsics(scenenn_path)
    camera_intrinsic_matrix = camera_intrinsic_transform(intrinsics)

    camera_info = CameraInfo()
    camera_info.height = intrinsics['depth_height']
    camera_info.width = intrinsics['depth_width']

    camera_info.distortion_model = "plumb_bob"
    camera_info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
    camera_info.R = np.ndarray.flatten(np.identity(3))
    camera_info.K = np.ndarray.flatten(camera_intrinsic_matrix[:, :3])
    camera_info.P = np.ndarray.flatten(camera_intrinsic_matrix)

    return camera_info


def writeTransform(transform, timestamp, frame_id, output_bag):
    scale, shear, angles, transl, persp = tf.transformations.decompose_matrix(
        transform)
    rotation = tf.transformations.quaternion_from_euler(*angles)

    trans = TransformStamped()
    trans.header.stamp = timestamp
    trans.header.frame_id = 'world'
    trans.child_frame_id = frame_id
    trans.transform.translation.x = transl[0]
    trans.transform.translation.y = transl[1]
    trans.transform.translation.z = transl[2]
    trans.transform.rotation.x = rotation[0]
    trans.transform.rotation.y = rotation[1]
    trans.transform.rotation.z = rotation[2]
    trans.transform.rotation.w = rotation[3]

    msg = tfMessage()
    msg.transforms.append(trans)
    output_bag.write('/tf', msg, timestamp)


def convert(scenenn_path, scene, to_frame, output_bag):
    rospy.init_node('scenenn_node', anonymous=True)
    frame_id = "/scenenn_camera_frame"

    # Write RGB and depth images.
    write_rgbd = True
    # Write instance image.
    write_instances = True
    # Write colorized instance image.
    write_instances_color = False
    # Write colored pointclouds of the instance segments.
    write_object_segments = False
    # Write colored pointcloud of the whole scene.
    write_scene_pcl = False

    # Set camera information and model.
    camera_info = parse_camera_info(scenenn_path)
    camera_model = PinholeCameraModel()
    camera_model.fromCameraInfo(camera_info)

    timestamps = parse_timestamps(scenenn_path, scene)
    trajectory = parse_trajectory(scenenn_path, scene)

    # Initialize some vars.
    header = Header(frame_id=frame_id)
    cvbridge = CvBridge()

    # Skip frame 1 as its timestamp (0.00) does not work in ROS.
    # TODO(margaritaG): avoid skipping the first frame, simply shift all timestamps.
    start_frame = 2
    frame = start_frame
    while (not rospy.is_shutdown() and frame < (to_frame + 1)
           and frame < len(timestamps)):
        timestamp = rospy.Time.from_sec(
            timestamps[frame] / np.power(10.0, 6.0))
        try:
            transform = trajectory[frame - 1]
        except KeyError:
            sys.exit(
                'It is common for transform data to be missing in the last' \
                'few frames of a SceneNN sequence. Stopping reading frames.'
            )
        writeTransform(transform, timestamp, frame_id, output_bag)
        header.stamp = timestamp

        # Read RGB, Depth and Instance images for the current view.
        bgr_image = cv2.imread(
            rgb_path_from_frame(scenenn_path, scene, frame),
            cv2.IMREAD_UNCHANGED)
        depth_image = cv2.imread(
            depth_path_from_frame(scenenn_path, scene, frame),
            cv2.IMREAD_UNCHANGED)
        if (not os.path.exists(
                instance_path_from_frame(scenenn_path, scene, frame))):
            print('SceneNN instance data not found at {0}'.format(
                instance_path_from_frame(scenenn_path, scene, frame)))
            if frame == start_frame:
                sys.exit(
                    'Please either ensure you have downloaded the instance '\
                    'data for this scene or disable instance writing.')
            else:
                print(
                    'It is common for instance data to be missing ' \
                    'for some frames. Ignoring.'
                )
                frame += 1
                continue
        instance_image_bgra = cv2.imread(
            instance_path_from_frame(scenenn_path, scene, frame),
            cv2.IMREAD_UNCHANGED)

        instance_image_mono_32 = pack_bgra(
            instance_image_bgra[:, :, 0], instance_image_bgra[:, :, 1],
            instance_image_bgra[:, :, 2], instance_image_bgra[:, :, 3])

        # From uint32 BGRA to uint16 (RA-only).
        # Warning: this is an unsafe casting, as values cannot be preserved.
        instance_image = np.uint16(instance_image_mono_32)

        if (write_object_segments):
            # Write all the instances in the current view as pointclouds.
            instances_in_current_frame = np.unique(instance_image)

            for instance in instances_in_current_frame:
                instance_mask = np.ma.masked_not_equal(instance_image,
                                                       instance).mask
                masked_depth_image = np.ma.masked_where(
                    instance_mask, depth_image)

                # Workaround for when 2D mask is only False values
                # and collapses to a single boolean False.
                if (not instance_mask.any()):
                    instance_mask_3D = np.broadcast_arrays(
                        instance_mask[np.newaxis, np.newaxis, np.newaxis],
                        bgr_image)
                else:
                    instance_mask_3D = np.broadcast_arrays(
                        instance_mask[:, :, np.newaxis], bgr_image)

                masked_bgr_image = np.ma.masked_where(instance_mask_3D[0],
                                                      bgr_image)

                object_segment_pcl = convert_bgrd_to_pcl(
                    masked_bgr_image, masked_depth_image, camera_model)
                object_segment_pcl.header = header
                output_bag.write('/scenenn_node/object_segment',
                                 object_segment_pcl, timestamp)

        if (write_scene_pcl):
            # Write the scene from the current view as pointcloud.
            scene_pcl = convert_bgrd_to_pcl(bgr_image, depth_image,
                                            camera_model)
            scene_pcl.header = header
            output_bag.write('/scenenn_node/scene', scene_pcl, timestamp)

        if (write_rgbd):
            # Write the RGBD data.
            bgr_msg = cvbridge.cv2_to_imgmsg(bgr_image, "8UC3")
            bgr_msg.encoding = "bgr8"
            bgr_msg.header = header
            output_bag.write('/camera/rgb/image_raw', bgr_msg, timestamp)

            depth_msg = cvbridge.cv2_to_imgmsg(depth_image, "16UC1")
            depth_msg.header = header
            output_bag.write('/camera/depth/image_raw', depth_msg, timestamp)

            camera_info.header = header

            output_bag.write('/camera/rgb/camera_info', camera_info, timestamp)
            output_bag.write('/camera/depth/camera_info', camera_info,
                             timestamp)

        if (write_instances):
            # Write the instance data.
            instance_msg = cvbridge.cv2_to_imgmsg(instance_image, "16UC1")
            instance_msg.header = header
            output_bag.write('/camera/instances/image_raw', instance_msg,
                             timestamp)
        if (write_instances_color):
            instance_bgra_msg = cvbridge.cv2_to_imgmsg(instance_image_bgra,
                                                       "8UC4")
            instance_bgra_msg.header = header
            output_bag.write('/camera/instances/image_rgb', instance_bgra_msg,
                             timestamp)

        print("Dataset timestamp: " + '{:4}'.format(timestamp.secs) + "." +
              '{:09}'.format(timestamp.nsecs) + "     Frame: " +
              '{:3}'.format(frame) + " / " + str(len(timestamps)))

        frame += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        usage='''%(prog)s [-h] --scenenn-path PATH --scene-id ID
                            [--limit NUM] [--output-bag NAME]''',
        description='Write data from a SceneNN scene to a rosbag.')
    parser.add_argument(
        "--scenenn-path",
        required=True,
        help="path to the scenenn_data folder",
        metavar="PATH")
    parser.add_argument(
        "--scene-id",
        required=True,
        help="select the scene with id ID",
        metavar="ID")
    parser.add_argument(
        "--limit",
        default=np.inf,
        type=int,
        help="only write NUM frames to the bag (Default: infinite)",
        metavar="NUM")
    parser.add_argument(
        "--output-bag",
        default="scenenn.bag",
        help="write to bag with name NAME.bag.",
        metavar="NAME")

    args = parser.parse_args()
    scenenn_path = args.scenenn_path
    scene = args.scene_id.zfill(3)
    output_bag_path = args.output_bag
    to_frame = args.limit

    if not os.path.isdir(os.path.join(scenenn_path, scene)):
        print("SceneNN scene data not found at {0}".format(
            os.path.join(scenenn_path, scene)))
        sys.exit('Please ensure you have downloaded the scene data.')

    if not output_bag_path.endswith(".bag"):
        output_bag_path = output_bag_path + ".bag"
    output_bag = rosbag.Bag(output_bag_path, 'w')
    try:
        convert(scenenn_path, scene, to_frame, output_bag)
    except rospy.ROSInterruptException:
        pass
    finally:
        output_bag.close()
