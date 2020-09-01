#!/usr/bin/env python2.7
# -*- coding: utf-8 -*
import cv2
import rospy 
import numpy as np
from darknet_ros_msgs.msg import BoundingBoxes
import cv_bridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_msgs.msg import Int32
from MyToolBox import KalmanFilter
import OpticalFlowToolBox
import LandmarkToolBox
import MyToolBox
import warnings



class getNodeInfo:
    def __init__(self):
        # 共用时间
        self.UnionTimeEnd = rospy.get_time()
        self.UnionTimeStart = 0
        self.UnionTimeInterval = 0

        # Landmark 初始变量
        self.bridge = cv_bridge.CvBridge()
        self.BBinfo = rospy.Subscriber("darknet_ros/bounding_boxes", BoundingBoxes, self.callback_boundingbox)
        self.img = rospy.Subscriber('/image_changed', Image, self.image_callback)

        self.image_pub = rospy.Publisher('/image_node', Image)
        self.OBJ_Info_pub = rospy.Publisher('/node_info_obj', String)
        self.MovePartten_info_pub = rospy.Publisher('/node_info_mp', String)
        self.Interval_pub = rospy.Publisher('/node_info_inter', Int32)
        self.ImgNum_pub = rospy.Publisher('/img_num', Int32)
        
        self.SetYoloTime = 1
        self.YoloTimeEnd = 0
        self.YoloTimeInterval = 0
        self.StartTime = rospy.get_time()
        self.Start_flag = 0

        self.TrashDict = {}
        self.IndicDict = {}
        self.TempImgDict = {}

        self.ImgCount = 0
        self.ImgNum = 0
        self.FileNum = 0
        self.OBJNameListFlag = 1

        self.TempObjNum = []
        self.DictNum = []


        # Optical Flow 初始变量
        self.feature_params = dict(maxCorners = 300, qualityLevel = 0.2, minDistance = 5, blockSize = 7)
        self.lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.initialize_flag = False
        self.color = (0, 255, 0)        
        self.LX, self.LY, self.RX, self.RY = 0, 0, 0, 0

        self.dt = 1.0/600
        self.F = np.array([[1, self.dt], [0, 1]])
        self.H = np.array([1, 0]).reshape(1, 2)
        self.Q = np.array([[0.05, 0.0], [0.0, 0.0]])
        self.R = np.array([0.5]).reshape(1, 1)

        self.Left_kf_x = KalmanFilter(F = self.F, H = self.H, Q = self.Q, R = self.R)
        self.Left_kf_y = KalmanFilter(F = self.F, H = self.H, Q = self.Q, R = self.R)
        self.Right_kf_x = KalmanFilter(F = self.F, H = self.H, Q = self.Q, R = self.R)
        self.Right_kf_y = KalmanFilter(F = self.F, H = self.H, Q = self.Q, R = self.R)

        self.__angle_flag = 0
        self.dir_flag = 1
        self.count = 0
        self.pre_text = 'forward'
        self.Direction = 'forward'
        self.DirectionNow = 'forward'
        self.cornor_direction  = 'forward'

        self.TempOfImg = {}
        self.OfImgNum = 0
        self.SaveFlagOf = 0
        self.OfCount = 0
        self.angle = 0
        self.OF_sum = 0
        self.OFImgSet = {}
        self.DictImgSURF = {}

    
    def callback_boundingbox(self, message):
        if self.Start_flag == 0:
            self.StartInterval = rospy.get_time() - self.StartTime
            if self.StartInterval > 5:
                self.Start_flag = 1
        else:
            num = len(message.bounding_boxes)
            OBJNameList = []

            self.YoloTimeStart = rospy.get_time()
            self.YoloTimeInterval = self.YoloTimeStart - self.YoloTimeEnd

            # 满足时间条件时，判断是否为真节点，根据结果保存图像
            if self.YoloTimeInterval >= self.SetYoloTime:
                self.TrashDict, self.IndicDict, self.TempImgDict, self.ImgNum, self.TempObjNum, self.ImgCount, self.FileNum, \
                    LMset, Obj, nowd, SendFlag = LandmarkToolBox.CheckSaveOrPass(self.ImgCount, self.TrashDict, self.IndicDict, self.ImgNum, self.TempObjNum, self.TempImgDict, \
                        self.FileNum, self.now_d)
                
                if SendFlag == 1:
                    # 计算和前一节点的时间间隔
                    self.UnionTimeStart = rospy.get_time()
                    self.UnionTimeInterval = self.UnionTimeStart - self.UnionTimeEnd
                    self.UnionTimeEnd = rospy.get_time()

                    # 初始节点时间距离归0
                    if self.FileNum == 1:
                        self.UnionTimeInterval = 0

                    '''
                    LMset: 图像集合
                    Obj: 此处物体类型
                    nowd: 此处运动方式
                    self.UnionTimeTnterval: 距离上一个节点的时间间隔
                    '''
                    # 发布消息
                    self.ImgNum_pub.publish(len(LMset))
                    for i in range(1, len(LMset)+1):
                        self.image_pub.publish(self.bridge.cv2_to_imgmsg(LMset[i], "bgr8"))
                    self.OBJ_Info_pub.publish(Obj)
                    self.MovePartten_info_pub.publish(MyToolBox.DirTransform(nowd))
                    self.Interval_pub.publish(int(self.UnionTimeInterval))                    


            # 还是同一节点
            for i in range(num):
                # 获取物体名
                OBJName = message.bounding_boxes[i].Class

                # 获取物体坐标
                x = (message.bounding_boxes[i].xmax + message.bounding_boxes[i].xmin) 
                y = (message.bounding_boxes[i].ymax + message.bounding_boxes[i].ymin) 
                coordinate = [x/2, y/2]

                # 位置筛选
                if coordinate[1] > 110 and coordinate[1] < 360:
                    if coordinate[0] > 100 and coordinate[0] < 328:
                        OBJNameList.append(OBJName)
                        self.DictNum = num
                        self.TempObjNum.append(self.DictNum)

                    elif coordinate[0] > 548 and coordinate[0] < 776:
                        OBJNameList.append(OBJName)
                        self.DictNum = num
                        self.TempObjNum.append(self.DictNum)

                # 如果次帧为空列表，跳出循环
                if OBJNameList == []:
                    self.OBJNameListFlag = 0
                    break
                TrashNum, IndicNum = LandmarkToolBox.NumCount(OBJNameList)
                self.TrashDict, self.IndicDict = LandmarkToolBox.UpdateDict([TrashNum, IndicNum], self.TrashDict, self.IndicDict)
                self.OBJNameListFlag = 1
            
            if self.OBJNameListFlag == 1:
                self.TempImgDict[self.ImgNum] = (self.DictNum, self.image)
                self.YoloTimeEnd = rospy.get_time()

                self.ImgNum += 1
                self.ImgCount += 1



    def image_callback(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
        self.ROI = self.image[226:256, 0:876]
        left_frame = self.image[160:305, 109:329]
        right_frame = self.image[160:305, 547:767]

        ''' ********** Optical Flow 初始帧 *********** '''
        if self.initialize_flag == False:            
            # 左画面
            self.prev_gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            self.prev_left = cv2.goodFeaturesToTrack(self.prev_gray_left, mask = None, **self.feature_params)

            # 右画面
            self.prev_gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
            self.prev_right = cv2.goodFeaturesToTrack(self.prev_gray_right, mask = None, **self.feature_params)
            
            # 长条画面
            self.prev_gray_ROI = cv2.cvtColor(self.ROI, cv2.COLOR_BGR2GRAY)
            self.prev_point = cv2.goodFeaturesToTrack(self.prev_gray_ROI, mask = None, **self.feature_params)

            self.initialize_flag = True
    
            '''  *********** Optical Flow 追踪帧 ************ '''
        else:
            # 左画面
            self.prev_gray_left, self.prev_left, self.LY, self.LX = OpticalFlowToolBox.direction_detect(left_frame, \
                self.prev_gray_left, self.prev_left, self.feature_params, self.lk_params, self.color)
            
            # 右画面
            self.prev_gray_right, self.prev_right, self.RY, self.RX = OpticalFlowToolBox.direction_detect(right_frame, \
                self.prev_gray_right, self.prev_right, self.feature_params, self.lk_params, self.color)

            # 长条画面
            self.prev_gray_ROI, self.prev_point, temp = OpticalFlowToolBox.GetLength(self.ROI, self.prev_gray_ROI, \
                self.prev_point, self.feature_params, self.lk_params, self.color)


            # 计算光流夹角
            if self.LY != 0 and self.LX != 0 and self.RY != 0 and self.RX != 0:
                Lx = np.dot(self.H, self.Left_kf_x.predict())[0]
                self.Left_kf_x.update(self.LX)

                Ly = np.dot(self.H, self.Left_kf_y.predict())[0]
                self.Left_kf_y.update(self.LY)

                Rx = np.dot(self.H, self.Right_kf_x.predict())[0]
                self.Right_kf_x.update(self.RX)

                Ry = np.dot(self.H, self.Right_kf_y.predict())[0]
                self.Right_kf_y.update(self.RY)                

                self.Left_Angle = OpticalFlowToolBox.angular([Ly, Lx])
                self.Right_Angle = OpticalFlowToolBox.angular([Ry, Rx])   

            # 获得前进方向      self.RotateFlag (1:右, 2:左, 3:前, 4:后)
            self.RotateFlag = OpticalFlowToolBox.GetDirectionOfTravel(self.Left_Angle, self.Right_Angle)

            '''******************** 跳过转弯检测的初始5帧 ******************** '''
            if self.dir_flag <= 5:
                self.now_d, self.pre_d, self.pre_d_sub = self.RotateFlag, self.RotateFlag, self.RotateFlag
                self.dir_flag += 1
            
            else:
                # 当前判断和上一阶段不同
                if self.pre_d != self.RotateFlag:
                    
                    # 当前判断和上一帧不同
                    if self.pre_d_sub != self.RotateFlag:
                        self.count = 0
                    
                    # 当前判断和上一帧相同
                    else:
                        self.count += 1
                    self.pre_d_sub = self.RotateFlag
                
                # 当前判断和上一阶段相同
                else:
                    self.count = 0

                if self.now_d == 1 or self.now_d == 2:
                    self.TempOfImg[self.OfImgNum] = self.image
                    self.OfImgNum += 1                

                '''********************  再次滤波 连续出现5次相同的变化 ********************'''
                if self.count >= 5:
                    # 计算和前一节点的时间间隔
                    if self.now_d != 3:
                        self.UnionTimeStart = rospy.get_time()
                        self.UnionTimeInterval = self.UnionTimeStart - self.UnionTimeEnd
                        self.UnionTimeEnd = rospy.get_time()
                    self.now_d = self.RotateFlag
                    
                    if self.now_d == 3:
                        self.cornor_direction = self.pre_d
                        self.SaveFlagOf = 1
                    
                    self.count = 0
                    
                # 把这一阶段的结果赋值
                self.pre_d = self.now_d


                # 计算角度 + 整合方向  返回的 self.angle 即相机旋转的角度
                self.OF_sum, self.DirectionNow, self.Direction, self.__angle_flag, self.angle = OpticalFlowToolBox.GetRotateAng(self.now_d, \
                    self.OF_sum, temp, self.__angle_flag, self.angle)


                # 开启时，保存图像等信息
                if self.SaveFlagOf == 1:
                    self.FileNum, self.TempOfImg, self.OfCount, self.SaveFlagOf, self.OFImgSet, self.Obj, self.MovePartten, self.UnionTimeInterval \
                        = OpticalFlowToolBox.SaveDirectionImg(self.FileNum, \
                        self.angle, self.cornor_direction, self.UnionTimeInterval, self.TempOfImg, self.OfCount, self.SaveFlagOf)
                    self.OfImgNum = 0

                    '''
                    self.OFImgSet: 图像集合
                    self.Obj: 此处物体类型
                    self.MovePartten: 此处运动方式
                    self.UnionTimeTnterval: 距离上一个节点的时间间隔
                    '''
                    # 开始发布消息
                    self.ImgNum_pub.publish(len(self.OFImgSet))
                    for i in range(1, len(self.OFImgSet)+1):
                        self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.OFImgSet[i], "bgr8"))
                    self.OBJ_Info_pub.publish(self.Obj)
                    self.MovePartten_info_pub.publish(self.MovePartten)
                    self.Interval_pub.publish(int(self.UnionTimeInterval))