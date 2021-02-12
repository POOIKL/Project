#!/usr/bin/env python2.7
# -*- coding: utf-8 -*
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import cv2
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import kmc2
import rospy



class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):
        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")
        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)       # x = A x + B u
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q  # P = A P At + Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
        	(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)



def DirTransform(d):
    res = 'forward'
    if d == 1:
        res = 'right'
    elif d == 2:
        res = 'left'
    elif d == 3:
        res = 'forward'
    else:
        res = 'backward'
    return res



def SameInterSampling(index, rho):	
	delete = len(index) % rho
	for _ in range(delete):
		index.pop()
	Bei = len(index) / rho 
	return [index[n] for n in range(0, len(index), Bei)]


def SaveFig(g, dg, num, CurPos, lenDict, savepath):
    path = savepath + 'result/'
    color_map = ['lightskyblue' for _ in range(len(lenDict))]
    color_map[CurPos -1] = 'salmon'

    # 图基础属性
    plt.clf()
    # plt.title("Topology Map")
    plt.title("Step " + str(num))
    plt.ylim(-25, 55)
    plt.xlim(-40,40)
    plt.xlabel('X-axis [second]')
    plt.ylabel('Y-axis [second]')
    
    nx.draw(g, nx.get_node_attributes(g, 'pos'), node_color=color_map, with_labels=True, node_size=180, alpha=1)
    nx.draw(dg, nx.get_node_attributes(dg, 'pos'), with_labels=False, node_size=0, alpha=1,arrowsize=20, edge_color='forestgreen')
    plt.axis('on')

    if not os.path.exists(path):
        os.makedirs(path) 
    plt.savefig(path + str(num) + '.jpg')
    
    plt.ion()
    plt.pause(1)
    plt.show()
    dg.clear()
    return g, dg



def AngleTransform(Curdir, RotateDir):
    if Curdir[0] == 0 and Curdir[1] == 1:
        if RotateDir == 'left':
            angle = 90
        elif RotateDir == 'right':
            angle = - 90
        elif RotateDir == 'forward':
            angle = 0
        else:
            angle = 180

    elif Curdir[0] == 0 and Curdir[1] == -1:
        if RotateDir == 'left':
            angle = 90
        elif RotateDir == 'right':
            angle = - 90
        elif RotateDir == 'forward':
            angle = 0
        else:
            angle = 180

    elif Curdir[0] == -1 and Curdir[1] == 0:
        if RotateDir == 'left':
            angle = 90
        elif RotateDir == 'right':
            angle = - 90
        elif RotateDir == 'forward':
            angle = 0
        else:
            angle = 180

    elif Curdir[0] == 1 and Curdir[1] == 0:
        if RotateDir == 'left':
            angle = 90
        elif RotateDir == 'right':
            angle = - 90
        elif RotateDir == 'forward':
            angle = 0
        else:
            angle = 180
    else:
        print('error')
    return angle



# 回转矩阵函数
def rotaMat(coordXY, DirVec, theta, distance):
    M = np.mat([[np.cos(theta * np.pi/180), -1 * np.sin(theta * np.pi/180)], \
        [np.sin(theta * np.pi/180), np.cos(theta * np.pi/180)]])

    ChangedDirVec = np.ravel(np.dot(M, DirVec))
    ChangedDirVec[0], ChangedDirVec[1] = int(ChangedDirVec[0]), int(ChangedDirVec[1])

    coordXY += ChangedDirVec * distance
    coordXY[0] = round(coordXY[0], 2)
    coordXY[1] = round(coordXY[1], 2)
    return coordXY.tolist(), ChangedDirVec 



def rotaMatArrow(DirVec, Info, arrowlen):
    theta = AngleTransform(DirVec, Info)
    M = np.mat([[np.cos(theta * np.pi/180), -1 * np.sin(theta * np.pi/180)], \
        [np.sin(theta * np.pi/180), np.cos(theta * np.pi/180)]])

    ChangedDirVec = np.ravel(np.dot(M, DirVec))
    ChangedDirVec[0], ChangedDirVec[1] = int(ChangedDirVec[0])*arrowlen, int(ChangedDirVec[1])*arrowlen
    return  ChangedDirVec 



# 计算距离
def calDistance(p1, p2):
    return np.sqrt(np.power((p2[0] - p1[0]), 2) + np.power((p2[1] - p1[1]), 2))



# 查找修正点
def SearchPoint(PointId, CurP, Dirpre, graph, CoorDict):
    dx = CurP[0] - CoorDict[PointId][0]
    dy = CurP[1] - CoorDict[PointId][1]

    if Dirpre[0] == 0 and Dirpre[1] == -1:
        if dx != 0:
            Flag = 'x'
        else:
            Flag = 'n'
    elif Dirpre[0] == 0 and Dirpre[1] == 1:
        if dx != 0:
            Flag = 'x'
        else:
            Flag = 'n'
    elif Dirpre[0] == 1 and Dirpre[1] == 0:
        if dy != 0:
            Flag = 'y'
        else:
            Flag = 'n'
    elif Dirpre[0] == -1 and Dirpre[1] == 0:
        if dy != 0:
            Flag = 'y'
        else:
            Flag = 'n'
    else:
        print('error')

    # 寻找同一直线上的点号
    SameLine = []
    if Flag == 'x':
        for i in range(1, len(CoorDict)+1):
            if CoorDict[i][0] == CurP[0]:
                SameLine.append((CoorDict[i][0],i))
    
    elif Flag == 'y':
        for i in range(1, len(CoorDict)+1):
            if CoorDict[i][1] == CurP[1]:
                SameLine.append((CoorDict[i][1],i))
    else:
        SameLine = []
    
    # 寻找中止条件
    ChangeCoord = []
    if SameLine != []:        
        if Dirpre[0] == 0 and Dirpre[1] == -1:
            SameLine = sorted(SameLine)
            ChangeCoord.append(SameLine[0][1])
            for i in range(1, len(SameLine)):
                n = len(nx.dijkstra_path(graph, source=SameLine[i-1][1], target=SameLine[i][1]))
                if n == 2:
                    ChangeCoord.append(SameLine[i][1])
                else:
                    break
        
        elif Dirpre[0] == 0 and Dirpre[1] == 1:
            SameLine = sorted(SameLine, reverse=True)
            ChangeCoord.append(SameLine[0][1])
            for i in range(1, len(SameLine)):
                n = len(nx.dijkstra_path(graph, source=SameLine[i-1], target=SameLine[i]))
                if n == 2:
                    ChangeCoord.append(SameLine[i][1])
                else:
                    break 

        elif Dirpre[0] == 1 and Dirpre[1] == 0:
            SameLine = sorted(SameLine)
            ChangeCoord.append(SameLine[0][1], reverse=True)
            for i in range(1, len(SameLine)):
                n = len(nx.dijkstra_path(graph, source=SameLine[i-1], target=SameLine[i]))
                if n == 2:
                    ChangeCoord.append(SameLine[i][1])
                else:
                    break 

        elif Dirpre[0] == - 1 and Dirpre[1] == 0:
            SameLine = sorted(SameLine)
            ChangeCoord.append(SameLine[0][1])
            for i in range(1, len(SameLine)):
                n = len(nx.dijkstra_path(graph, source=SameLine[i-1], target=SameLine[i]))
                if n == 2:
                    ChangeCoord.append(SameLine[i][1])
                else:
                    break 
    return ChangeCoord, Flag



def SearchNeighbors(CurPos, NodePre, CoordDict, g):
    neigh = []
    adj = []
    deltx = CurPos[0] - CoordDict[NodePre][0]
    delty = CurPos[1] - CoordDict[NodePre][1]
    adj = [i for i in g.neighbors(NodePre)]

    if deltx > 0:
        for i in adj:
            if (CoordDict[i][0] - CoordDict[NodePre][0]) > deltx:
                neigh.append(i)
    elif deltx < 0:
        for i in adj:
            if (CoordDict[i][0] - CoordDict[NodePre][0]) < deltx:
                neigh.append(i)
    elif delty > 0:
        for i in adj:
            if (CoordDict[i][1] - CoordDict[NodePre][1]) > delty:
                neigh.append(i)
    elif delty < 0:
        for i in adj:
            if (CoordDict[i][1] - CoordDict[NodePre][1]) < delty:
                neigh.append(i)
    return neigh



def InfoExtract(UnionInfo):
    Info = []
    UnionInfo = sorted(UnionInfo)
    Info.append(UnionInfo[0])
    UnionInfo.pop(0)
    for i in UnionInfo:
        if len(i) <= 8:
            Info.append(i)
    for i in UnionInfo:
        if i not in Info:
            Info.append(i)
    return Info



def GetClockAngle(v1, v2):
    # 2个向量模的乘积
    TheNorm = np.linalg.norm(v1)*np.linalg.norm(v2)
    # 叉乘
    rho = np.rad2deg(np.arcsin(np.cross(v1, v2)/TheNorm))
    # 点乘
    theta = np.rad2deg(np.arccos(np.dot(v1,v2)/TheNorm))
    if rho < 0:
        return int(- theta)
    else:
        return int(theta)



def GetNormVec(a, b):
    # 从a出发，指向b
    T = np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    return [(b[0] - a[0]) / T, (b[1] - a[1]) / T]



def DirConvert(deg):
    if deg == 0:
        return 'go forward'
    elif deg == 90:
        return 'turn left'
    elif deg == 180:
        return 'go backward'
    elif deg == -90:
        return 'turn right'
    else:
        print('Error') 



def GetGoPattern(Info):
    GoPattern = []
    for i in range(len(Info)-1):
        GoPattern.append(DirConvert(Info[i]))
    return GoPattern


def Transform(PatternAndVecN, ObjWithVecN):
    objnum = []
    # 查数
    if ObjWithVecN[1] == 'ThisIsCorner':
        return ' the intersection'
    else:
        Obj = ''
        if PatternAndVecN[1][0] == ObjWithVecN[0][0] and PatternAndVecN[1][1] == ObjWithVecN[0][1]:
            objnum.append((int(ObjWithVecN[1][4]), int(ObjWithVecN[1][19]), int(ObjWithVecN[1][33]), int(ObjWithVecN[1][48])))
        else:
            objnum.append((int(ObjWithVecN[1][19]), int(ObjWithVecN[1][4]), int(ObjWithVecN[1][48]), int(ObjWithVecN[1][33])))

        LeftRegionNum = (objnum[0][0], objnum[0][2])   # LeftRegionNum = (垃圾箱数, 指示盘数)
        RightRegionNum = (objnum[0][1], objnum[0][3])  # RightRegionNum = (垃圾箱数, 指示盘数)

        if LeftRegionNum[0] != 0:
            Obj += ' ' + str(LeftRegionNum[0]) + ' trash can'
        if LeftRegionNum[1] != 0:
            Obj += ' ' + str(LeftRegionNum[1]) + ' indicator'
        Obj += ' on the left and '

        if RightRegionNum[0] != 0:
            Obj += ' ' + str(RightRegionNum[0]) + ' trash can'
        if RightRegionNum[1] != 0:
            Obj += ' ' + str(RightRegionNum[1]) + ' indicator'
        Obj += ' on the right.'
        return Obj
                
def Last(Objtuple, Patterntuple):
    objnum = []
    # 查数
    if Objtuple[1] == 'ThisIsCorner':
        return ' the intersection'
    else:
        Obj = ''
        if Patterntuple[1][0] == Objtuple[0][0] and Patterntuple[1][1] == Objtuple[0][1]:
            objnum.append((int(Objtuple[1][4]), int(Objtuple[1][19]), int(Objtuple[1][33]), int(Objtuple[1][48])))
        else:
            objnum.append((int(Objtuple[1][19]), int(Objtuple[1][4]), int(Objtuple[1][48]), int(Objtuple[1][33])))

        LeftRegionNum = (objnum[0][0], objnum[0][2])   # LeftRegionNum = (垃圾箱数, 指示盘数)
        RightRegionNum = (objnum[0][1], objnum[0][3])  # RightRegionNum = (垃圾箱数, 指示盘数)

        if LeftRegionNum[0] != 0:
            Obj += ' ' + str(LeftRegionNum[0]) + ' trash can'
        if LeftRegionNum[1] != 0:
            Obj += ' ' + str(LeftRegionNum[1]) + ' indicator'
        Obj += ' on the left and'

        if RightRegionNum[0] != 0:
            Obj += ' ' + str(RightRegionNum[0]) + ' trash can'
        if RightRegionNum[1] != 0:
            Obj += ' ' + str(RightRegionNum[1]) + ' indicator'
        Obj += ' on the right.'
        return Obj


def Translate(PatternAndVec, route, ObjWithVec):
    SR = ' Firstly, ' + str(PatternAndVec[0][0]) + '. \n'
    for i in range(1, len(PatternAndVec)):
        SR += ' See' + Transform(PatternAndVec[i-1], ObjWithVec[i]) + ' (Node ' + str(route[i]) + '). ' +  str(PatternAndVec[i][0]) + '.\n'
    
    FinalObj = Last(ObjWithVec[-1], PatternAndVec[-1])
    SR += ' Finally, go forward. when see' + str(FinalObj) +' you will reach the target point' + ' (Node ' + str(route[-1]) +'). '
    return SR



def GetSemanticResult(route, PatternAndVec, OBJDict):
    ObjWithVec = []
    for node in route:
        ObjWithVec.append((OBJDict[node][1], OBJDict[node][0]))

    SR = Translate(PatternAndVec, route, ObjWithVec)
    return SR



def Navigation(TopologyMap, start, end, CoordNodeDict, NowVec, OBJDict):
    route = nx.dijkstra_path(TopologyMap, source=start, target=end)
    routeVec = []
    AngInfo = []
    PatternAndVec = []  

    for i in range(1, len(route)):
        checkVec = GetNormVec(CoordNodeDict[start], CoordNodeDict[route[i]])
        routeVec.append(checkVec)
        RotAng = GetClockAngle(NowVec, checkVec)
        
        AngInfo.append(RotAng)
        NowVec = checkVec
        start = route[i]
    
    AngInfo.append(0)
    gopattern = GetGoPattern(AngInfo)

    for i in range(len(gopattern)):
        PatternAndVec.append((gopattern[i], routeVec[i]))

    SR = GetSemanticResult(route, PatternAndVec, OBJDict)
    return SR



def OBJTranslation(ObjName):
    if ObjName == 'ThisIsCorner':
        return 'You have ever turned at here before.'
    elif ObjName[0] == '0':
        return 'There are ' + ObjName[10] + ' Indicators here.'
    elif ObjName[10] == '0':
        return 'There are ' + ObjName[0] + ' Trash can here.'
    else:
        return 'There are ' + ObjName[0] + ' Trash can and ' + ObjName[10] + ' Indicators here.'



def GetBGRvec(TempImgset):
    ColorBin = 30
    count = 1
    for img in TempImgset:
        # BGR直方图
        histB = cv2.calcHist([img], [0], None, [ColorBin], [0, 256])
        histG = cv2.calcHist([img], [1], None, [ColorBin], [0, 256])
        histR = cv2.calcHist([img], [2], None, [ColorBin], [0, 256])

        histB = cv2.normalize(histB, histB, 0, 1, cv2.NORM_MINMAX, -1)
        histG = cv2.normalize(histG, histG, 0, 1, cv2.NORM_MINMAX, -1)
        histR = cv2.normalize(histR, histR, 0, 1, cv2.NORM_MINMAX, -1)
        
        if count == 1:
            Bcombine = histB
            Gcombine = histG
            Rcombine = histR
            count += 1
        else:
            Bcombine = np.hstack((Bcombine,histB))
            Gcombine = np.hstack((Gcombine,histG))
            Rcombine = np.hstack((Rcombine,histR))

    # PCA降维
    pca = PCA(n_components=1)
    
    pca.fit(Bcombine) # B
    Bcombine = pca.transform(Bcombine)
    pca.fit(Gcombine) # G
    Gcombine = pca.transform(Gcombine)
    pca.fit(Rcombine) # R
    Rcombine = pca.transform(Rcombine)

    Bcombine = np.array(Bcombine)
    Gcombine = np.array(Gcombine)
    Rcombine = np.array(Rcombine)

    BGR = np.ravel(np.vstack((np.vstack((Bcombine, Gcombine)), Rcombine)))
    return BGR



def SuperFrame(BriefImgset):
    #设置Flannde参数
    FLANN_INDEX_KDTREE=0
    indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
    searchParams= dict(checks=50)
    flann=cv2.FlannBasedMatcher(indexParams,searchParams)

    MatchedBrief = []
    cou = 1
    for i in range(len(BriefImgset)-1):
        matches=flann.knnMatch(np.asarray(BriefImgset[i+1],np.float32), np.asarray(BriefImgset[i],np.float32),k=2)
        #舍弃大于0.5的匹配结果
        if cou == 1:
            MatchedBrief = [BriefImgset[i+1][j] for j, (m,n) in enumerate(matches) if m.distance< 0.87*n.distance]
            cou += 1
        elif cou != 1:
            MatchedBrief = np.vstack((MatchedBrief, [BriefImgset[i+1][j] for j, (m,n) in enumerate(matches) if m.distance< 0.87*n.distance]))
        else:
            print('Error')
    return MatchedBrief



def kmc2_seed(m, num):
    seeding = kmc2.kmc2(m, num, afkmc2 = True)
    model = MiniBatchKMeans(num, init = seeding).fit(m)
    new_centers = model.cluster_centers_
    return new_centers



def Coslength(p,q):
    r = np.dot(p,q)/(np.linalg.norm(p)*(np.linalg.norm(q)))
    return r



def Kd_VLAD_BGR_sim(BriefImgset, TempImgset, PrevBrief, PrevBGR):
    SeedNum = 10
    NowBrief = SuperFrame(BriefImgset)
    NowBGR = GetBGRvec(TempImgset)
    
    [r1, _] = NowBrief.shape
    # [r2, c2] = PrevBrief.shape

    combine = np.vstack((NowBrief, PrevBrief))
    afk_seed = kmc2_seed(combine, SeedNum)    

    # kmeans 聚类, bow[i] 为每个特征属于哪个组的, cord[i] 为各个中心组分别是第几号的 
    #  distance[i] 为各个特征与所属的中心点的距离差
    distance = KMeans(n_clusters = SeedNum, init = afk_seed).fit_transform(combine)
    bow = KMeans(n_clusters = SeedNum, init = afk_seed).fit(combine).predict(combine)
    cord = KMeans(n_clusters = SeedNum, init = afk_seed).fit(combine).cluster_centers_
    
    point_num = len(bow)
    center_num = len(cord)

    L1 = [0 for i in range(center_num)]
    L2 = [0 for i in range(center_num)]

    for i in range(0, point_num):
        for j in range(center_num):
            if i < r1:
                L1[bow[i]] += distance[i,j]
            else:
                L2[bow[i]] += distance[i,j]

    # 整合
    ImgV1 = np.hstack((L1, NowBGR))
    ImgV2 = np.hstack((L2, PrevBGR))

    return round(Coslength(ImgV1, ImgV2), 2)



def SaveLandmarkImg(FileNum, TempObjNum, TempImgDict):
    SaveNum = 20
    FileNum += 1
    LMImgSet = {}

    TempObjNum = list(set(TempObjNum))
    n = len(TempObjNum)

    # 出现最多的物体数        
    First = TempObjNum[n - 1]

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
        index = SameInterSampling(index, SaveNum)

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


def GetObjResult(TrashLeft, TrashRight, IndicatorLeft, IndicatorRight):
    if TrashLeft == {}:
        TNumLeft = 0
    else:    
        TNumLeft = max(TrashLeft)

    if TrashRight == {}:
        TNumRight = 0
    else:    
        TNumRight = max(TrashRight)
    
    if IndicatorLeft == {}:
        INumLeft = 0
    else:    
        INumLeft = max(IndicatorLeft)

    if IndicatorRight == {}:
        INumRight = 0
    else:    
        INumRight = max(IndicatorRight)

    TrashLeft = {}
    TrashRight = {}
    IndicatorLeft = {}
    IndicatorRight = {}
    return TNumLeft, TNumRight, INumLeft, INumRight, TrashLeft, TrashRight, IndicatorLeft, IndicatorRight



def CheckSaveOrPass(ImgCount, TrashLeft, TrashRight, IndicatorLeft, IndicatorRight, ImgNum, TempObjNum, TempImgDict, FileNum, now_d):
    SendFlag = 0
    LMSet = {}
    if ImgCount >= 25:
        NodeTnumLeft, NodeTnumRight, NodeInumLeft, NodeInumRight, TrashLeft, TrashRight, IndicatorLeft, IndicatorRight = \
            GetObjResult(TrashLeft, TrashRight, IndicatorLeft, IndicatorRight)
        FileNum, LMSet = SaveLandmarkImg(FileNum, TempObjNum, TempImgDict)
        Obj = 'Left' + str(NodeTnumLeft) + 'TrashCan,'+ 'Right' + str(NodeTnumRight) + 'TrashCan,' \
            + 'Left' + str(NodeInumLeft) + 'Indictor,' + 'Right' + str(NodeInumRight) + 'Indictor.'
        SendFlag = 1
    else:
        Obj = 0
        TrashLeft = {}
        TrashRight = {}
        TrashLeft = {}
        TrashRight = {}
    
    TempImgDict = {}
    ImgNum = 0
    TempObjNum = []
    ImgCount = 0
    return TrashLeft, TrashRight, IndicatorLeft, IndicatorRight, TempImgDict, ImgNum, TempObjNum, ImgCount, FileNum, LMSet, Obj, now_d, SendFlag



def NumCount(L):
    LTrashNum = 0
    RTrashNum = 0
    LIndicatorNum = 0
    RIndicatorNum = 0
    for i in L:
        if i == 'Trash can+Left':
            LTrashNum += 1
        elif i == 'Trash can+Right':
            RTrashNum += 1
        elif i == 'Indicator panel+Left':
            LIndicatorNum += 1
        elif i == 'Indicator panel+Right':
            RIndicatorNum += 1
        else:
            print('NumCount function error')
    return LTrashNum, RTrashNum, LIndicatorNum, RIndicatorNum



def UpdateDict(L, TrashLeft, TrashRight, IndicatorLeft, IndicatorRight):
    # 仅当前帧
    # L[左侧垃圾桶数, 右侧垃圾桶数, 左侧指示盘数, 右侧指示盘数]
    if L[0] not in TrashLeft.keys(): # Trash left
        TrashLeft[L[0]] = 1
    else:
        TrashLeft[L[0]] += 1

    if L[1] not in TrashRight.keys(): # Trash right
        TrashRight[L[1]] = 1
    else:
        TrashRight[L[1]] += 1

    if L[2] not in IndicatorLeft.keys(): # Indicator left
        IndicatorLeft[L[2]] = 1
    else:
        IndicatorLeft[L[2]] += 1

    if L[3] not in IndicatorRight.keys(): # Indicator right
        IndicatorRight[L[3]] = 1
    else:
        IndicatorRight[L[3]] += 1
    return TrashLeft, TrashRight, IndicatorLeft, IndicatorRight




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
        MovePartten = DirTransform(cornor_direction)
    
    index = range(1, len(TempOfImg) + 1)
    # 保存图像
    if FileNum != 6:
        if len(index) > SavNum:
            index = SameInterSampling(index, SavNum)
            for i in index:
                OFImgSet[n] = TempOfImg[i]
                n += 1
        else:
            for i in index:
                OFImgSet[n] = TempOfImg[i-1]
                n += 1
    else:
        Nindex = [index[i] for i in range(8, 8 - len(index), -1)]
        Nindex.reverse()
        Nindex = SameInterSampling(Nindex, SavNum)
        for i in Nindex:
            OFImgSet[n] = TempOfImg[i]
            n += 1

    OfCount = 0
    TempOfImg = {}
    SaveFlagOf = 0
    return FileNum, TempOfImg, OfCount, SaveFlagOf, OFImgSet, Obj, MovePartten, UnionTimeInterval 