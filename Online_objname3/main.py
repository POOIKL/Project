#!/usr/bin/env python2.7
# -*- coding: utf-8 -*
import cv2
import numpy as np
import rospy
from NodeInfo import getNodeInfo
from BuildMap import Build
import Mylibrary as ml
import networkx as nx
import warnings
import os


if __name__=='__main__':
    Path = './'
    warnings.filterwarnings('ignore')
    rospy.init_node('Topology_Map')
    Info = getNodeInfo()
    TopologyMap = Build(Path)

    while(True):
        # 导航
        # NavigationType = input('Choose Navigation Type: 1 or 2')
        # if NavigationType == 1:
        #     Tp = input('Input The Target Position:')
        #     print(ml.Navigation(TopologyMap.g, 1, Tp, TopologyMap.CoordNodeDict, TopologyMap.DirVector, TopologyMap.OBJDict))

        # elif NavigationType == 2:
        #     Np = input('Input The Start Position: ')
        #     Tp = input('Input The Target Position: ')
        #     DirVec = input('Input Start Position Direction Vector (eg: 1, 1): ')
        #     print(ml.Navigation(TopologyMap.g, Np, Tp, TopologyMap.CoordNodeDict, DirVec, TopologyMap.OBJDict))
        # else:
        #     print('Error')

        Tp = input('Input The Destination Node:')
        print(ml.Navigation(TopologyMap.g, 1, Tp, TopologyMap.CoordNodeDict, TopologyMap.DirVector, TopologyMap.OBJDict))

    rospy.spin()
