import numpy as np
from math import pi
YUMI_KINEMATIC_TREE = {
    "yumi_body":["yumi_link_1_r","yumi_link_1_l"],
    "yumi_link_1_r":["yumi_link_2_r"],
    "yumi_link_2_r":["yumi_link_3_r"],
    "yumi_link_3_r":["yumi_link_4_r"],
    "yumi_link_4_r":["yumi_link_5_r"],
    "yumi_link_5_r":["yumi_link_6_r"],
    "yumi_link_6_r":["yumi_link_7_r"],
    "yumi_link_7_r":[],
    "yumi_link_1_l":["yumi_link_2_l"],
    "yumi_link_2_l":["yumi_link_3_l"],
    "yumi_link_3_l":["yumi_link_4_l"],
    "yumi_link_4_l":["yumi_link_5_l"],
    "yumi_link_5_l":["yumi_link_6_l"],
    "yumi_link_6_l":["yumi_link_7_l"],
    "yumi_link_7_l":[],
}
YUMI_NAME2IND = {
    "yumi_body":0,
    "yumi_link_1_r":1,
    "yumi_link_2_r":2,
    "yumi_link_3_r":3,
    "yumi_link_4_r":4,
    "yumi_link_5_r":5,
    "yumi_link_6_r":6,
    "yumi_link_7_r":7,
    "yumi_link_1_l":8,
    "yumi_link_2_l":9,
    "yumi_link_3_l":10,
    "yumi_link_4_l":11,
    "yumi_link_5_l":12,
    "yumi_link_6_l":13,
    "yumi_link_7_l":14,
}
YUMI_TRANSLATIONS = \
        np.array([
            [0.05355, -0.0725, 0.51492],
            [0.03, 0.0, 0.1],
            [-0.03, 0.17283, 0.0],
            [-0.04188, 0.0, 0.07873],
            [0.0405, 0.16461, 0.0],
            [-0.027, 0, 0.10039],
            [0.027, 0.029, 0.0],
            [0.05355, 0.0725, 0.51492],
            [0.03, 0.0, 0.1],
            [-0.03, 0.17283, 0.0],
            [-0.04188, 0.0, 0.07873],
            [0.0405, 0.16461, 0.0],
            [-0.027, 0, 0.10039],
            [0.027, 0.029, 0.0]
        ])
YUMI_ROTATIONS = \
        np.array([
            [-0.9795,  -0.5682,   -2.3155],
            [1.57079632679, 0, 0],
            [-1.57079632679, 0, 0],
            [1.57079632679, -1.57079632679, 0],
            [-1.57079632679, 0, 0],
            [1.57079632679, 0, 0],
            [-1.57079632679, 0, 0],
            [0.9781, -0.5716, 2.3180],
            [1.57079632679, 0.0, 0.0],
            [-1.57079632679, 0.0, 0.0],
            [1.57079632679, -1.57079632679, 0],
            [-1.57079632679, 0, 0],
            [1.57079632679, 0, 0],
            [-1.57079632679, 0, 0]
        ])
YUMI_NAMES = [
    "yumi_link_1_r",
    "yumi_link_2_r",
    "yumi_link_3_r",
    "yumi_link_4_r",
    "yumi_link_5_r",
    "yumi_link_6_r",
    "yumi_link_7_r"
]
YUMI_ROOT = "yumi_body"
YUMI_SHOULDER_R_NAME = "yumi_link_1_r"
YUMI_ELBOW_R_NAME = "yumi_link_4_r"
YUMI_WRIST_R_NAME = "yumi_link_7_r"
YUMI_SHOULDER_L_NAME = "yumi_link_1_l"
YUMI_ELBOW_L_NAME = "yumi_link_4_l"
YUMI_WRIST_L_NAME = "yumi_link_7_l"
ROBOTHAND_LB = np.array([-1.6, -1.7, -1.6, -1.7, -1.6, -1.7, -1.6, -1.7, -1.0, 0.0, -0.4, -1.0])
ROBOTHAND_UB = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.4, 0.0, 0.0])
WISEGLOVE_LB = np.array([0, 0, 53, 0, 0, 22, 0, 0, 22, 0, 0, 35, 0, 0])*pi/180
WISEGLOVE_UB = np.array([45, 100, 0, 90, 120, 0, 90, 120, 0, 90, 120, 0, 90, 120])*pi/180
