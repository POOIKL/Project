#!/usr/bin/env python2.7
# -*- coding: utf-8 -*
import cv2
import rospy 
import numpy as np
import cv_bridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_msgs.msg import Int32
import networkx as nx
import MyToolBox
import warnings


class Build:
    def __init__(self, ThePath):
        self.SavePath = ThePath
        self.surf = cv2.xfeatures2d.SURF_create(300)
        self.bf = cv2.BFMatcher_create()
        self.bridge = cv_bridge.CvBridge()

        self.Online = rospy.Subscriber('/image_changed', Image, self.OnlineFunction)
        self.NodeImg = rospy.Subscriber('/image_node', Image, self.NodeImageCallback)
        self.OBJ_Info = rospy.Subscriber('/node_info_obj', String, self.ObjInfoCallback)
        self.Mp_Info = rospy.Subscriber('/node_info_mp', String, self.MpInfoCallback)
        self.Inter_Info = rospy.Subscriber('/node_info_inter', Int32, self.InterInfoCallback)
        self.ImgNum_Info = rospy.Subscriber('/img_num', Int32, self.ImgNumCallback)
        self.UnionInfo = []
        self.Info = []

        self.SURFImgset = []
        self.ReadNode = 1
        self.NodeNow = 1
        self.NodePre = 1
        self.SaveImg = 1
        self.g = nx.Graph()
        self.dg = nx.DiGraph()
        self.arrowlen = 5

        """
        self.CoordNodeDict:    储存各个节点坐标的字典
        self.SURFImgNodeDict:  储存各个节点SURF特征量的字典
        self.OBJDict:          储存各个节点物体的字典
        """
        self.CoordNodeDict = {}
        self.SURFImgNodeDict = {}
        self.OBJDict = {}

    def ImgNumCallback(self, msg):
        self.NodeImgNum = msg.data

    def NodeImageCallback(self, msg):
        self.NodeImage = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
        self.SURFImgset.append(self.surf.detectAndCompute(self.NodeImage,None)[1])

    def ObjInfoCallback(self, msg):
        self.UnionInfo.append(msg.data)

    def MpInfoCallback(self, msg):        
        self.UnionInfo.append(msg.data)
    
    def InterInfoCallback(self, msg):        
        self.UnionInfo.append(msg.data)
    
    def OnlineFunction(self, msg):
        if len(self.UnionInfo) == 3:
            # self.Info[距离, 行进模式, 存在的物体]
            self.Info = MyToolBox.InfoExtract(self.UnionInfo)

            if len(self.SURFImgset) == self.NodeImgNum:
                if self.ReadNode == 1:
                    # 初始化坐标 方向
                    self.CurrentPosition = [0, 0]
                    self.DirVector = np.array([0, -1])
                    
                    # 字典记录图像集合 坐标 物体
                    self.SURFImgNodeDict[self.NodeNow] = self.SURFImgset
                    self.CoordNodeDict[self.NodeNow] = self.CurrentPosition
                    self.OBJDict[self.NodeNow] = self.Info[2] 
                    self.MpInfoPre = self.Info[1]           

                    # networkx 图更新
                    self.g.add_node(self.NodeNow, pos=(self.CurrentPosition[0], self.CurrentPosition[1]))

                    # networkx 箭头更新
                    self.ArrowVec = MyToolBox.rotaMatArrow(self.DirVector, self.Info[1], self.arrowlen)
                    self.dg.add_node(0, pos=(self.CurrentPosition[0], self.CurrentPosition[1]))
                    self.dg.add_node(-1, pos=(self.CurrentPosition[0] + self.ArrowVec[0], self.CurrentPosition[1] + self.ArrowVec[1]))
                    self.dg.add_edges_from([(0, -1)], edge_color='r')

                    # 画图
                    self.g, self.dg = MyToolBox.SaveFig(self.g, self.dg, self.SaveImg, self.NodeNow, self.CoordNodeDict, self.SavePath)
                    
                    self.NodePre = self.NodeNow
                    self.ReadNode += 1
                    self.SaveImg += 1

                else:
                    # 所在前一节点的方向向量
                    self.DirPrevious = self.DirVector
                    
                    # 计算顺（逆）时针旋转角度
                    RotAngle = MyToolBox.AngleTransform(self.DirVector, self.MpInfoPre)
                    self.MpInfoPre = self.Info[1]

                    # 计算当前坐标，方向向量进行变换
                    self.CurrentPosition, self.DirVector = MyToolBox.rotaMat(self.CurrentPosition, self.DirVector, RotAngle, self.Info[0])

                    # 查找是否有图像比对的点
                    similarset = []
                    for i in range(1, len(self.CoordNodeDict)+1):
                        if MyToolBox.calDistance(self.CurrentPosition, self.CoordNodeDict[i]) < 5:
                            simiValue = MyToolBox.GetSimilarity(self.SURFImgset, self.SURFImgNodeDict[i])
                            if simiValue > 0.3:
                                similarset.append((simiValue, i))
                    similarset = sorted(similarset, reverse=True)


                    # 地图修正
                    if similarset != []:
                        ModifyPointSet, RotFlag = MyToolBox.SearchPoint(similarset[0][1], self.CurrentPosition, self.DirPrevious, self.g, self.CoordNodeDict)
                    
                        if ModifyPointSet != []:
                            if RotFlag == 'x':
                                detx = self.CoordNodeDict[similarset[0][1]][0] - self.CurrentPosition[0]
                                self.g.add_edges_from([(ModifyPointSet[len(ModifyPointSet) - 1], similarset[0][1])])

                                # 整体修正
                                for i in ModifyPointSet:
                                    self.CoordNodeDict[i][0] += detx
                                    self.g.add_node(i, pos=(self.CoordNodeDict[i][0], self.CoordNodeDict[i][1]))
                                
                                self.CurrentPosition = self.CoordNodeDict[similarset[0][1]]
                                
                                # networkx 箭头更新
                                self.ArrowVec = MyToolBox.rotaMatArrow(self.DirVector, self.Info[1], self.arrowlen)
                                self.dg.add_node(0, pos=(self.CurrentPosition[0], self.CurrentPosition[1]))
                                self.dg.add_node(-1, pos=(self.CurrentPosition[0] + self.ArrowVec[0], self.CurrentPosition[1] + self.ArrowVec[1]))
                                self.dg.add_edges_from([(0, -1)], edge_color='r')
                                
                                # 画图
                                self.g, self.dg = MyToolBox.SaveFig(self.g, self.dg, self.SaveImg, similarset[0][1], self.CoordNodeDict, self.SavePath)
                                self.NodePre = similarset[0][1]
                                self.SaveImg += 1
                        
                            elif RotFlag == 'y':
                                dety = self.CoordNodeDict[similarset[0][1]][1] - self.CurrentPosition[1]
                                self.g.add_edges_from([(ModifyPointSet[len(ModifyPointSet) - 1], similarset[0][1])])

                                # 整体修正
                                for i in ModifyPointSet:
                                    self.CoordNodeDict[i][1] += dety
                                    self.g.add_node(i, pos=(self.CoordNodeDict[i][0], self.CoordNodeDict[i][1]))
                                
                                self.CurrentPosition = self.CoordNodeDict[similarset[0][1]]

                                # networkx 箭头更新
                                self.ArrowVec = MyToolBox.rotaMatArrow(self.DirVector, self.Info[1], self.arrowlen)
                                self.dg.add_node(0, pos=(self.CurrentPosition[0], self.CurrentPosition[1]))
                                self.dg.add_node(-1, pos=(self.CurrentPosition[0] + self.ArrowVec[0], self.CurrentPosition[1] + self.ArrowVec[1]))
                                self.dg.add_edges_from([(0, -1)], edge_color='r')

                                # 画图
                                self.g, self.dg = MyToolBox.SaveFig(self.g, self.dg, self.SaveImg, similarset[0][1], self.CoordNodeDict, self.SavePath)
                                self.NodePre = similarset[0][1]
                                self.SaveImg += 1

                        elif ModifyPointSet == []:
                            self.g.add_edges_from([(self.NodePre, similarset[0][1])])                    
                            self.CurrentPosition = self.CoordNodeDict[similarset[0][1]]

                            # networkx 箭头更新
                            self.ArrowVec = MyToolBox.rotaMatArrow(self.DirVector, self.Info[1], self.arrowlen)
                            self.dg.add_node(0, pos=(self.CurrentPosition[0], self.CurrentPosition[1]))
                            self.dg.add_node(-1, pos=(self.CurrentPosition[0] + self.ArrowVec[0], self.CurrentPosition[1] + self.ArrowVec[1]))
                            self.dg.add_edges_from([(0, -1)], edge_color='r')

                            # 画图
                            self.g, self.dg = MyToolBox.SaveFig(self.g, self.dg, self.SaveImg, similarset[0][1], self.CoordNodeDict, self.SavePath)
                            self.NodePre = similarset[0][1]
                            self.SaveImg += 1

                    else:
                        neighbor = []
                        self.NodeNow += 1
                        self.CoordNodeDict[self.NodeNow] = self.CurrentPosition
                        self.SURFImgNodeDict[self.NodeNow] = self.SURFImgset
                        self.OBJDict[self.NodeNow] = self.Info[2]

                        neighbor = MyToolBox.SearchNeighbors(self.CurrentPosition, self.NodePre, self.CoordNodeDict, self.g)

                        if neighbor != []:                    
                            self.g.add_node(self.NodeNow, pos=(self.CurrentPosition[0], self.CurrentPosition[1]))
                            self.g.remove_edge(neighbor[0], self.NodePre)
                            self.g.add_edges_from([(self.NodeNow, self.NodePre)])
                            self.g.add_edges_from([(self.NodeNow, neighbor[0])])

                        else:
                            self.g.add_node(self.NodeNow, pos=(self.CurrentPosition[0], self.CurrentPosition[1]))
                            self.g.add_edges_from([(self.NodeNow, self.NodePre)])
                        
                        self.NodePre = self.NodeNow

                        # networkx 箭头更新
                        self.ArrowVec = MyToolBox.rotaMatArrow(self.DirVector, self.Info[1], self.arrowlen)
                        self.dg.add_node(0, pos=(self.CurrentPosition[0], self.CurrentPosition[1]))
                        self.dg.add_node(-1, pos=(self.CurrentPosition[0] + self.ArrowVec[0], self.CurrentPosition[1] + self.ArrowVec[1]))
                        self.dg.add_edges_from([(0, -1)], edge_color='r')

                        # 画图
                        self.g, self.dg = MyToolBox.SaveFig(self.g, self.dg, self.SaveImg, self.NodeNow, self.CoordNodeDict, self.SavePath)
                        self.SaveImg += 1
                
                self.NodeImgNum = 0
                self.SURFImgset = []
                self.UnionInfo = []
                self.Info = []

        else:
            pass