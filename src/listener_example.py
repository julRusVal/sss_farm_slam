#!/usr/bin/env python3
# By Julian Valdez
# Inspiration from https://github.com/smarc-project/smarc_perception/blob/noetic-devel/sss_object_detection/scripts/sss_detection_listener.py

import rospy

import tf2_ros
import tf2_geometry_msgs
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection2DArray

import gtsam
import csv


class sam_slam_listener:
    def __init__(self, gt_top_name, dr_top_name, det_top_name, frame_name):
        print('Start: sam_slam_listener class')
        # Topics and stuff
        self.gt_topic = gt_top_name
        self.dr_topic = dr_top_name
        self.det_topic = det_top_name
        self.frame = frame_name

        # tf stuff
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Logging
        self.gt_poses = []
        self.dr_poses = []
        self.detections = []

        # Timing and state
        self.last_time = rospy.Time.now()
        self.pose_update = False
        self.data_written = False

        # Subscribers
        # Ground truth
        self.gt_pose = None
        self.gt_subscriber = rospy.Subscriber(self.gt_topic,
                                              Odometry,
                                              self.gt_callback)

        # Dead reckoning
        self.dr_pose = None
        self.dr_subscriber = rospy.Subscriber(self.dr_topic,
                                              Odometry,
                                              self.dr_callback)

        # Detections
        self.det_info = None
        self.det_subscriber = rospy.Subscriber(self.det_topic,
                                               Detection2DArray,
                                               self.det_callback)

        self.time_check = rospy.Timer(rospy.Duration(2.0),
                                      self.time_check_callback)

    # Subscriber callbacks
    def gt_callback(self, msg):
        transformed_pose = self.transform_pose(msg.pose, from_frame=msg.header.frame_id, to_frame=self.frame)
        self.gt_pose = transformed_pose
        gt_position = transformed_pose.pose.position
        self.gt_poses.append([gt_position.x, gt_position.y, gt_position.z])
        # print(msg.pose.pose.position)

    def dr_callback(self, msg):
        self.pose_update = True
        self.last_time = rospy.Time.now()

        transformed_pose = self.transform_pose(msg.pose, from_frame=msg.header.frame_id, to_frame=self.frame)
        self.dr_pose = transformed_pose

        dr_position = transformed_pose.pose.position
        self.dr_poses.append([dr_position.x, dr_position.y, dr_position.z])
        # print(msg.pose.pose.position)

    def det_callback(self, msg):
        self.det_info = msg
        for det_ind, detection in enumerate(msg.detections):
            for res_ind, result in enumerate(detection.results):
                # Pose in base_link
                detection_position = result.pose
                # Convert to map
                transformed_pose = self.transform_pose(detection_position,
                                                       from_frame=msg.header.frame_id,
                                                       to_frame=self.frame)
                # Extract the position
                det_position = transformed_pose.pose.position
                # Append [x,y,z,id,score]
                self.detections.append([det_position.x,
                                        det_position.y,
                                        det_position.z,
                                        result.id,
                                        result.score])
        # print(msg)

    # Timer callback
    def time_check_callback(self, event):
        if not self.pose_update:
            return
        delta_t = rospy.Time.now() - self.last_time
        if delta_t.to_sec() >= 1 and not self.data_written:
            print('Data written')
            self.write_data()
            self.data_written = True
        return

    # Transforms
    def transform_pose(self, pose, from_frame, to_frame):
        trans = self.wait_for_transform(from_frame=from_frame,
                                        to_frame=to_frame)
        pose_transformed = tf2_geometry_msgs.do_transform_pose(pose, trans)
        return pose_transformed

    def wait_for_transform(self, from_frame, to_frame):
        """Wait for transform from from_frame to to_frame"""
        trans = None
        while trans is None:
            try:
                trans = self.tf_buffer.lookup_transform(
                    to_frame, from_frame, rospy.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as error:
                print('Failed to transform. Error: {}'.format(error))
        return trans

    # random utility methods
    def write_data(self):
        gt_poses_file_path = 'gt_poses.csv'
        dr_poses_file_path = 'dr_poses.csv'
        detections_file_path = 'detections.csv'

        # Save dead reckoning
        with open(dr_poses_file_path, "w", newline="") as f:
            writer = csv.writer(f)
            for row in self.dr_poses:
                writer.writerow(row)

        # Save ground truth
        with open(gt_poses_file_path, "w", newline="") as f:
            writer = csv.writer(f)
            for row in self.gt_poses:
                writer.writerow(row)

        # Save detections
        with open(detections_file_path, "w", newline="") as f:
            writer = csv.writer(f)
            for row in self.detections:
                writer.writerow(row)

def main():
    rospy.init_node('slam_listener', anonymous=True)
    rospy.Rate(5)

    ground_truth_topic = '/sam/sim/odom'
    dead_reckon_topic = '/sam/dr/global/odom/filtered'
    detection_topic = '/sam/payload/sidescan/detection_hypothesis'
    frame = 'map'

    print('initializing listener')
    listener = sam_slam_listener(ground_truth_topic, dead_reckon_topic, detection_topic, frame)

    while not rospy.is_shutdown():
        rospy.spin()


if __name__ == '__main__':
    main()
