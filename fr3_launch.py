#!/usr/bin/env python3
"""
ROS2 launch file converted from the original ROS1 version.
Compatible with ROS2 Humble or newer.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare   # <--- fix
from launch.actions import ExecuteProcess


def generate_launch_description():
    # === TF Broadcasters ===
    # TF: fr3_link0 → ref_frame
    tf1 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_fr3_to_ref',
        arguments=[
            '0.735097', '0.646207', '0.664998',    # translation (x y z)
            '-0.370981', '-0.863263', '0.320188', '0.120952',  # quaternion (x y z w)
            'fr3_link0', 'ref_frame'
        ]
    )
    # TF: fr3_link0 → ref_frame
    # tf1 = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     name='static_tf_fr3_to_ref',
    #     arguments=[
    #         '0.749022', '0.656649', '0.659689',
    #         '-0.382599', '-0.868189', '0.293314', '0.117616',
    #         'fr3_link0', 'ref_frame'
    #     ]
    # )
    # TF: ref_frame → camera_link
    tf2 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_ref_to_cam',
        arguments=[
            '-0.0010067', '0.014068', '-0.002151',
            '0.49272', '-0.49219', '0.50886', '0.50601',
            'ref_frame', 'camera_link'
        ]
    )

    # RViz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', '/home/u0161364/fr3_grasp_ros2/configs/fr3_grasp.rviz'],
        output='screen'
    )

    # Joy
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node'
    )

    # === Realsense camera ===
    left_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                FindPackageShare('realsense2_camera').find('realsense2_camera'),
                'launch',
                'rs_launch.py'
            )
        ]),
        launch_arguments={
            'camera_name': 'left_camera',
            'serial_no': "'047322071010'"
            # 'depth_module.depth_profile': '1280x720x30',
            # 'rgb_camera.color_profile': '1280x720x30',
            # 'enable_color': 'true',
            # 'enable_depth': 'true',
            # 'pointcloud.enable': 'true',
            # 'align_depth.enable': 'true',
            # 'initial_reset': 'true',
        }.items()
    )
    wrist_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([ 
            os.path.join(
                FindPackageShare('realsense2_camera').find('realsense2_camera'),
                'launch',
                'rs_launch.py'
            )
        ]),
        launch_arguments={
            'camera_name': 'wrist_camera',
            'serial_no': "'309622300781'",
            'rgb_camera.color_profile': '424,240,30',
        }.items()
    )
    front_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                FindPackageShare('realsense2_camera').find('realsense2_camera'),
                'launch',
                'rs_launch.py'
            )
        ]),
        launch_arguments={
            'camera_name': 'front_camera',
            'serial_no': "'f1421698'",
            # 'color_width': '424',
            # 'color_height': '240',
            # 'color_fps': '30',
        }.items()
    )
    # === Declare launch arguments ===
    declared_arguments = [
        DeclareLaunchArgument('robot_ip', default_value='172.16.0.2', description='Franka robot IP'),
        DeclareLaunchArgument('load_gripper', default_value='true', description='Load Franka gripper'),
        DeclareLaunchArgument('markerId', default_value='2'),
        DeclareLaunchArgument('markerSize', default_value='0.07'),
        DeclareLaunchArgument('eye', default_value='left'),
        DeclareLaunchArgument('marker_frame', default_value='aruco_marker_frame'),
        DeclareLaunchArgument('ref_frame', default_value=''),
        DeclareLaunchArgument('corner_refinement', default_value='LINES'),
        DeclareLaunchArgument('camera_frame', default_value='camera_color_optical_frame'),
        DeclareLaunchArgument('camera_image_topic', default_value='/camera/color/image_raw'),
        DeclareLaunchArgument('camera_info_topic', default_value='/camera/color/camera_info'),
    ]

    # === Assemble LaunchDescription ===
    return LaunchDescription(
        declared_arguments + [
            left_camera,
            wrist_camera,
            # front_camera,
            # tf1,
            # tf2,
            rviz_node,
            joy_node,
        ]
    )
