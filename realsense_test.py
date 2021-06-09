#!/usr/bin/python
# -*- coding: utf-8 -*-
## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2019 Intel Corporation. All Rights Reserved.

#####################################################
##           librealsense T265 example             ##
#####################################################

# First import the library
import time

import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs

# Declare RealSense pipeline, encapsulating the actual device and sensors
pipe = rs.pipeline()

# Build config object and request pose data
cfg = rs.config()
# cfg.enable_all_streams()

# Start streaming with requested config
pipe.start(cfg)

try:
    plt.ion()
    ax = plt.subplot()
    init_img = np.zeros((800, 848), dtype=np.uint8)
    im = ax.imshow(init_img, norm=matplotlib.colors.Normalize(0, 255))
    # plt.show()
    for _ in range(50):
        # Wait for the next set of frames from the camera
        st = time.time()
        frames = pipe.wait_for_frames()
        img = np.array(frames.get_fisheye_frame().get_data())
        # ax.imshow(img)
        im.set_data(img)
        plt.show()
        plt.pause(0.01)
        # plt.imshow(img)

        # Fetch pose frame
        pose = frames.get_pose_frame()
        if pose:
            # Print some of the pose data to the terminal
            data = pose.get_pose_data()
            print("Frame #{}".format(pose.frame_number))
            print("Position: {}".format(data.translation))
            print("Velocity: {}".format(data.velocity))
            print("Acceleration: {}\n".format(data.acceleration))

finally:
    pipe.stop()
