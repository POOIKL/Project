#!/usr/bin/env python2.7
# -*- coding: utf-8 -*
import cv2
import numpy as np
import os
import MyToolBox



def Gauss_weight(L):
    L.sort()
    mid = len(L) / 2
    for i in range(len(L)):
        L[i] = L[i]*(1/np.sqrt(2*np.pi))*np.exp(-0.5*(i-mid)**2)
    return L



# 光流函数
def direction_detect(new_frame, prev_gray, prev, FP, LP, color):
    xList = []
    yList = []
    
    gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    n, _, _ = prev.shape
    
    if n <= 10:
        Points = cv2.goodFeaturesToTrack(gray, mask = None, **FP)
        if Points is None:
            pass
        else:
            prev = np.vstack((prev, Points))
    
    nex, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **LP)
    good_old = prev[status == 1]
    if nex is None:
        nex = prev
    good_new = nex[status == 1]

    for _, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        
        # 画光流
        # cv2.line(new_frame, (a, b), (c, d), color, 1)
        # cv2.circle(new_frame, (a, b), 1, color, -1)
        
        # 图片水平, 竖直方向坐标差
        y_vol, x_vol = a - c, b - d
        xList.append(x_vol)
        yList.append(y_vol)

    prev_gray = gray.copy()
    prev = good_new.reshape(-1, 1, 2)

    # 高斯加权
    xList = Gauss_weight(xList)
    yList = Gauss_weight(yList)
    BigY = sum(yList)
    BigX = sum(xList)
    return prev_gray, prev, BigY, BigX



def GetLength(new_frame, prev_gray, prev, FP, LP, color):        
    L = []
    gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    n, _, _ = prev.shape
    
    if n <= 10:
        Points = cv2.goodFeaturesToTrack(gray, mask = None, **FP)
        if Points is None:
            pass
        else:
            prev = np.vstack((prev, Points))

    nex, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **LP)
    good_old = prev[status == 1]
    good_new = nex[status == 1]

    for _, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()

        # 画光流
        # cv2.line(new_frame, (a, b), (c, d), color, 1)
        # cv2.circle(new_frame, (a, b), 1, color, -1)    

        OFLength = np.sqrt(np.square(b-d) + np.square(a-c))
        L.append(OFLength)

    L.sort()
    temp = L[len(L) / 2]

    prev_gray = gray.copy()
    prev = good_new.reshape(-1, 1, 2)

    return prev_gray, prev, temp



def angular(a):
    p = []
    p.append(a[0][0])
    p.append(a[1][0])
    stander = np.array([10, 0])
    r = np.dot(p,stander)/(np.linalg.norm(p)*(np.linalg.norm(stander)))
    theta = np.rad2deg(np.arccos(r))
    if np.isnan(theta) == True:
        return 0 
    if p[1] < stander[1]:
        return -int(theta)
    return int(theta)



def GetDirectionOfTravel(Left_Angle, Right_Angle):
    # 判断窗口中的方向
    if Left_Angle >= - 90 and Left_Angle <= 90:
        Left_dir = 0
    else:
        Left_dir = 1

    if Right_Angle >= - 90 and Right_Angle <= 90:
        Right_dir = 0
    else:
        Right_dir = 1

    # 方向组合
    # RF (1:右, 2:左, 3:前, 4:后)
    if Left_dir == 1 and Right_dir == 1:                                
        RotateFlag = 1 # 右

    elif Left_dir == 0 and Right_dir == 0:                                
        RotateFlag = 2 # 左

    elif Left_dir == 1 and Right_dir == 0:                                
        RotateFlag = 3 # 前

    else:                
        RotateFlag = 4 # 后
    
    return RotateFlag



def angular_calculate(ang, Direction):
    ang = int((ang / 876)*360) % 360
    text = Direction + str(ang)
    return text, Direction, ang



def GetRotateAng(now_d, OF_sum, temp, angle_flag, ang):
    if now_d == 3:
        OF_sum = 0
        DirectionNow = 'Forward'
        Direction = 'Forward'
    
    elif now_d == 4:
        OF_sum = 0
        DirectionNow = 'Backward'
        Direction = 'Backward'

    elif now_d == 1:                
        if angle_flag == -1:
            OF_sum += temp
            DirectionNow, Direction, ang = angular_calculate(OF_sum, 'right')
        else:
            OF_sum = 0
            OF_sum += temp
            DirectionNow, Direction, ang = angular_calculate(OF_sum, 'right')     
            angle_flag = -1               

    else:
        if angle_flag == 1:
            OF_sum += temp
            DirectionNow, Direction, ang = angular_calculate(OF_sum, 'left')

        else:
            OF_sum = 0
            OF_sum += temp
            DirectionNow, Direction, ang = angular_calculate(OF_sum, 'left')     
            angle_flag = 1

    return OF_sum, DirectionNow, Direction, angle_flag, ang 



def SaveDirectionImg(FileNum, angle, cornor_direction, UnionTimeInterval, TempOfImg, OfCount, SaveFlagOf):
    SavNum = 20
    FileNum += 1
    OFImgSet = {}
    n = 1
    # 此处类型
    Obj = 'ThisIsCorner'

    # 转角方式
    if angle > 135 and angle < 225:
        MovePartten = 'backward'
    else:
        MovePartten = MyToolBox.DirTransform(cornor_direction)
    
    index = range(1, len(TempOfImg) + 1)
    # 保存图像
    if FileNum != 6:
        if len(index) > SavNum:
            index = MyToolBox.SameInterSampling(index, SavNum)
            for i in index:
                OFImgSet[n] = TempOfImg[i]
                n += 1
        else:
            for i in index:
                OFImgSet[n] = TempOfImg[i]
                n += 1
    else:
        Nindex = [index[i] for i in range(8, 8 - len(index), -1)]
        Nindex.reverse()
        Nindex = MyToolBox.SameInterSampling(Nindex, SavNum)
        for i in Nindex:
            OFImgSet[n] = TempOfImg[i]
            n += 1

    OfCount = 0
    TempOfImg = {}
    SaveFlagOf = 0
    return FileNum, TempOfImg, OfCount, SaveFlagOf, OFImgSet, Obj, MovePartten, UnionTimeInterval 