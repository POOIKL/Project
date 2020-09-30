#!/usr/bin/env python2.7
# -*- coding: utf-8 -*
import cv2
import numpy as np
import rospy
from NodeInfo import getNodeInfo
from BuildMap import Build
import MyToolBox
import networkx as nx
import warnings



if __name__=='__main__':
    Path = '/home/kou-ikeda/catkin_ws/src/result_time_online_time_ver2/'
    warnings.filterwarnings('ignore')
    rospy.init_node('Topology_Map')
    Info = getNodeInfo()
    TopologyMap = Build(Path)
    #r = nx.Graph()

    while(True):
        r = nx.Graph()
        # 导航
        NavigationType = input('Choose Navigation Type: 1 or 2 ')
        if NavigationType == 1:
            Tp = input('Input The Target Position:')
            route = nx.dijkstra_path(TopologyMap.g, source=1, target=Tp)
            print(MyToolBox.Navigation(TopologyMap.g, 1, Tp, TopologyMap.CoordNodeDict, TopologyMap.DirVector))
            r = MyToolBox.SaveFig(TopologyMap.g, TopologyMap.dg, r, 0, TopologyMap.CurrentPosition, TopologyMap.CoordNodeDict, Path, True, route)
        
        elif NavigationType == 2:
            Np = input('Input The Start Position: ')
            Tp = input('Input The Target Position: ')
            DirVec = input('Input Start Position Direction Vector (eg: 1, 1): ')
            route = nx.dijkstra_path(TopologyMap.g, source=Np, target=Tp)            
            print(MyToolBox.Navigation(TopologyMap.g, Np, Tp, TopologyMap.CoordNodeDict, DirVec))
            r = MyToolBox.SaveFig(TopologyMap.g, TopologyMap.dg, r, 0, TopologyMap.CurrentPosition, TopologyMap.CoordNodeDict, Path, True, route)
        
        else:
            print('Error')

        # 检查物体
        Obj = raw_input('Do you want to check object ? (N or Y) ')
        if Obj == 'Y':
            NodeNum = input('Input the Node number: ')
            print(MyToolBox.OBJTranslation(TopologyMap.OBJDict[NodeNum]))

        else:
            pass

    rospy.spin()
