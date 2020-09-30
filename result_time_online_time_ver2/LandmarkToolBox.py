#!/usr/bin/env python2.7
# -*- coding: utf-8 -*
import numpy as np
import rospy
import os
import cv2
import MyToolBox



def GetObjResult(TrashDict, IndicDict):
    if TrashDict == {}:
        TNum = 0
    else:    
        TNum = max(TrashDict)
    
    if IndicDict == {}:
        INum = 0
    else: 
        INum = max(IndicDict)

    TrashDict = {}
    IndicDict = {}
    return TNum, INum, TrashDict, IndicDict



def SaveLandmarkImg(FileNum, TempObjNum, NodeTnum, NodeInum, TempImgDict):
    SaveNum = 20
    FileNum += 1
    LMImgSet = {}

    TempObjNum = list(set(TempObjNum))
    n = len(TempObjNum)

    # 出现最多的物体数        
    First = TempObjNum[n - 1]
    NoImg = 0

    # 保存图像
    n = 0
    FinalImgDict = {}
    index = []

    for i in TempImgDict:
        if TempImgDict[i][0] == First:
            n += 1
            FinalImgDict[n] = TempImgDict[i][1]
            index.append(n)

    # 多于一定数量 等间隔保存
    if len(index) > SaveNum:
        index = MyToolBox.SameInterSampling(index, SaveNum)

        num = 1
        for i in index:
            LMImgSet[num] = FinalImgDict[i]
            num += 1
    # 多于一定数量 直接保存
    else:
        num = 1
        for i in index:
            LMImgSet[num] = FinalImgDict[i]
            num += 1
    return FileNum, LMImgSet                



def CheckSaveOrPass(ImgCount, TrashDict, IndicDict, ImgNum, TempObjNum, TempImgDict, FileNum, now_d):
    SendFlag = 0
    LMSet = {}
    if ImgCount >= 25:
        NodeTnum, NodeInum, TrashDict, IndicDict = GetObjResult(TrashDict, IndicDict)
        FileNum, LMSet = SaveLandmarkImg(FileNum, TempObjNum, NodeTnum, NodeInum, TempImgDict)
        Obj = str(NodeTnum) + 'TrashCan,'+ str(NodeInum) +'Indictor.'
        SendFlag = 1
    else:
        Obj = 0
        TrashDict = {}
        IndicDict = {}
    
    TempImgDict = {}
    ImgNum = 0
    TempObjNum = []
    BracketNum = 0
    ImgCount = 0
    return TrashDict, IndicDict, TempImgDict, ImgNum, TempObjNum, ImgCount, FileNum, LMSet, Obj, now_d, SendFlag



def NumCount(L):
    cTrash = 0
    cIndicator = 0
    for i in L:
        if i == 'Trash can':
            cTrash += 1
        elif i == 'Indicator panel':
            cIndicator += 1
    return cTrash, cIndicator



def UpdateDict(L, TrashDict, IndicDict):
    if L[0] not in TrashDict.keys():
        TrashDict[L[0]] = 1
    else:
        TrashDict[L[0]] += 1
    
    if L[1] not in IndicDict.keys():
        IndicDict[L[1]] = 1
    else:
        IndicDict[L[1]] += 1
    
    return TrashDict, IndicDict