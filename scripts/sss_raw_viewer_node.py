#!/usr/bin/env python3
import rospy
import cv2
from smarc_msgs.msg import Sidescan
from functools import partial
import numpy as np

"""
Utility for viewing and saving the raw sidescan imagery produced by the sss_detector.
This is based on the view_sidescan.py found in smarc_utils/sss_viewer

Will save the raw measurements when the buffer is full and continue to show thereafter. 
"""


def callback(img, msg):
    """
    Read new sss data and add to existing imagery
    """
    port = np.array(bytearray(msg.port_channel), dtype=np.ubyte)
    stbd = np.array(bytearray(msg.starboard_channel), dtype=np.ubyte)
    meas = np.concatenate([np.flip(port), stbd])
    img[1:, :] = img[:-1, :]
    img[0, :] = meas


# Parameters
buffer_len = 1000
sss_data_len = 1000  # This is determined by the message

rospy.init_node('sidescan_viewer', anonymous=True)

img = np.zeros((buffer_len, 2 * sss_data_len), dtype=np.ubyte)
cv2.namedWindow('Sidescan image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Sidescan image', 2 * 256, buffer_len)

rospy.Subscriber("/sam/payload/sidescan", Sidescan, partial(callback, img))

# spin() simply keeps python from exiting until this node is stopped
r = rospy.Rate(5)  # 10hz
count = 0
while not rospy.is_shutdown():
    # Save sss data
    count += 1
    if count == buffer_len:
        cv2.imwrite("sss_data.png", img)
    resized = cv2.resize(img, (2 * 256, buffer_len), interpolation=cv2.INTER_AREA)

    # Display sss data
    cv2.imshow("Sidescan image", resized)
    cv2.waitKey(1)
    r.sleep()
