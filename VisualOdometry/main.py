#!/usr/bin/env python2.7
# -*- coding: utf-8 -*
import cv2
import numpy as np
import rospy
from NodeInfo import getNodeInfo
import Mylibrary as ml
import networkx as nx
import warnings
import os


if __name__=='__main__':
    Path = './'
    warnings.filterwarnings('ignore')
    rospy.init_node('Topology_Map')
    Info = getNodeInfo()
    rospy.spin()
