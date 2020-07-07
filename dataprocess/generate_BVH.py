#!/usr/bin/env python
#encoding=utf-8
import sys
sys.path.append("../utils")
import numpy as np
from Quaternions import Quaternions
import rosbag
import pdb
import h5py
import tf

import sys
import getopt

import cv2
import rospy
import os
from cv_bridge import CvBridge, CvBridgeError

def linear_interpolate(quaternions,tgt_len):
    src_len = quaternions.shape[0]
    src_points = np.linspace(0,1,src_len)
    tgt_points = np.linspace(0,1,tgt_len)
    result = np.zeros([tgt_len]+list(quaternions.shape[1:])+[4])
    for i,tgt_point in enumerate(tgt_points):
        down_val = np.max(src_points[src_points<=tgt_point])
        up_val = np.min(src_points[src_points>=tgt_point]) 
        p = (tgt_point - down_val)/float(up_val-down_val) if up_val>down_val else 0.0
        quat1 = quaternions[src_points==down_val,...]
        quat2 = quaternions[src_points==up_val,...]
        result[i] = np.array(Quaternions.slerp(quat1,quat2,p))
    return result

def pos_to_ndarray(pos):
    # combined position and orientation information of a geometry_msgs/Pose message into a list

    # from quaternion([x,y,z,w]) to rotation matrix
    #wri_rotm = tf.transformations.quaternion_matrix([wri_pose.orientation.x, wri_pose.orientation.y, wri_pose.orientation.z, wri_pose.orientation.w])

    # pack up
    #packed_state = [wri_pose.position.x, wri_pose.position.y, wri_pose.position.z] + wri_rotm.reshape(9).tolist() + [elb_pose.position.x, elb_pose.position.y, elb_pose.position.z]

    return np.array([pos.x, pos.y, pos.z])

def quat_to_ndarray(quat):
    # returned quaternion is (x,y,z,w)
    return np.array([quat.x, quat.y, quat.z, quat.w])

def quat_to_euler(rotations):
    norm = rotations[:, :, 0] ** 2 + rotations[:, :, 1] ** 2 + rotations[:, :, 2] ** 2 + rotations[:, :, 3] ** 2
    norm = np.repeat(norm[:, :, np.newaxis], 4, axis=2)
    rotations /= norm
    rotations = Quaternions(rotations)
    rotations = np.degrees(rotations.euler())
    return rotations

def bag_to_bvh(bag_name, bvh_name):
    ## This function converts the rosbag file to h5 containing all the data(hand, forearm, upperarm) and the corresponding video.


    # Iterate to get the message contents
    bag_file = rosbag.Bag(bag_name)
    count = bag_file.get_message_count()
    bridge = CvBridge()

    idx = 0
    start_time = None # used to transform Unix timestamp to timestamps starting from 0

    time = np.zeros([count, 1])

    l_up_pos = np.zeros([count, 3])
    l_up_quat = np.zeros([count, 4])
    l_fr_pos = np.zeros([count, 3])
    l_fr_quat = np.zeros([count, 4])
    l_hd_pos = np.zeros([count, 3])
    l_hd_quat = np.zeros([count, 4])

    r_up_pos = np.zeros([count, 3])
    r_up_quat = np.zeros([count, 4])
    r_fr_pos = np.zeros([count, 3])
    r_fr_quat = np.zeros([count, 4])
    r_hd_pos = np.zeros([count, 3])
    r_hd_quat = np.zeros([count, 4])

    l_glove_angle = np.zeros([count, 14])
    r_glove_angle = np.zeros([count, 14])


    for topic, msg, t in bag_file.read_messages():
        ## topic name
        #print("Topic name is: " + topic)

        ## h5 content
        # do not use `t`!!! because it is the time at which a message is recorded through rosbag, instead of the original timestamp
        if idx == 0:
            start_time = msg.right_forearm_pose.header.stamp.to_sec()

        time[idx] = msg.right_forearm_pose.header.stamp.to_sec() - start_time

        l_up_pos[idx, :] = pos_to_ndarray(msg.left_upperarm_pose.pose.position)
        l_up_quat[idx, :] = quat_to_ndarray(msg.left_upperarm_pose.pose.orientation)
        l_fr_pos[idx, :] = pos_to_ndarray(msg.left_forearm_pose.pose.position)
        l_fr_quat[idx, :] = quat_to_ndarray(msg.left_forearm_pose.pose.orientation)
        l_hd_pos[idx, :] = pos_to_ndarray(msg.left_hand_pose.pose.position)
        l_hd_quat[idx, :] = quat_to_ndarray(msg.left_hand_pose.pose.orientation)   

        r_up_pos[idx, :] = pos_to_ndarray(msg.right_upperarm_pose.pose.position)
        r_up_quat[idx, :] = quat_to_ndarray(msg.right_upperarm_pose.pose.orientation)
        r_fr_pos[idx, :] = pos_to_ndarray(msg.right_forearm_pose.pose.position)
        r_fr_quat[idx, :] = quat_to_ndarray(msg.right_forearm_pose.pose.orientation)
        r_hd_pos[idx, :] = pos_to_ndarray(msg.right_hand_pose.pose.position)
        r_hd_quat[idx, :] = quat_to_ndarray(msg.right_hand_pose.pose.orientation)  

        l_glove_angle[idx, :] = np.array(msg.glove_state.left_glove_state)
        r_glove_angle[idx, :] = np.array(msg.glove_state.right_glove_state)

        #import pdb
        #pdb.set_trace()


        ## video


        ## Set counter
        idx = idx + 1

    ### Store the results
    # to construct a dataset for release, the data content must be complete!!!
    # l_hand_pos/l_hand_quat, l_forearm_pos/l_forearm_quat, l_upperarm_pos/l_upperarm_quat; after this, do an extraction to get wrist, elbow information for use in sign language robot!!!
    # store each part separately!!!  

   # Interpolate & transform to euler
    root = np.array([0.919788, 78.902473, 8.082646])
    rotations = np.stack([l_up_quat,l_fr_quat,l_hd_quat,r_up_quat,r_fr_quat,r_hd_quat],1)
    rotations = Quaternions(rotations)
    rotations = linear_interpolate(rotations,128)
    eulers = quat_to_euler(rotations)
    # Write to BVH
    f = open(bvh_name, "a") # open in append mode
    frame = eulers.shape[0]
    frametime = 1/30.0
    joint_num = 29
    file_string = '\nMOTION\n' + 'Frames: {}\n'.format(frame) + 'Frame Time: %.8f\n' % frametime
    for euler  in eulers:
        file_string += '%.6f %.6f %.6f ' % (root[0], root[1], root[2])
        for i in range(joint_num):
            if i==21:
                file_string += '%.6f %.6f %.6f ' % (euler[0][2], euler[0][1], euler[0][0])
            elif i==22:
                file_string += '%.6f %.6f %.6f ' % (euler[1][2], euler[1][1], euler[1][0])
            elif i==23:
                file_string += '%.6f %.6f %.6f ' % (euler[2][2], euler[2][1], euler[2][0])
            elif i==26:
                file_string += '%.6f %.6f %.6f ' % (euler[3][2], euler[3][1], euler[3][0])
            elif i==27:
                file_string += '%.6f %.6f %.6f ' % (euler[4][2], euler[4][1], euler[4][0])
            elif i==28:
                file_string += '%.6f %.6f %.6f ' % (euler[5][2], euler[5][1], euler[5][0])
            else:
                file_string += '%.6f %.6f %.6f ' % (0, 0, 0)
        file_string += '\n'
    f.write(file_string)

    f.close()



if __name__=="__main__":
    bag_name = "/media/liweijie/代码和数据/datasets/motionRetargeting/data/baozhu_1.bag"
    bvh_name = './results/result.bvh'
    bag_to_bvh(bag_name,bvh_name)