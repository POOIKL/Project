#!/usr/bin/env python2.7
# -*- coding: utf-8 -*
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import cv2
import os


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
	for i in range(delete):
		index.pop()
	Bei = len(index) / rho 
	return [index[n] for n in range(0, len(index), Bei)]



def SaveFig(g, dg, r, num, CurPos, lenDict, savepath, Nav, route):
    if Nav == False:
        path = savepath + 'result/'
        color_map = ['blue' for i in range(len(lenDict))]
        color_map[CurPos -1] = 'red'

        # 图基础属性
        plt.clf()
        plt.title("Topology Map")
        plt.ylim(-25, 55)
        plt.xlim(-40,40)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        
        nx.draw(g, nx.get_node_attributes(g, 'pos'), node_color=color_map, with_labels=True, node_size=180, alpha=1)
        nx.draw(dg, nx.get_node_attributes(dg, 'pos'), with_labels=False, node_size=0, alpha=1,arrowsize=20, edge_color='g')
        plt.axis('on')

        if not os.path.exists(path):
            os.makedirs(path) 
        plt.savefig(path + str(num) + '.jpg')
        
        plt.ion()
        plt.pause(1)
        plt.show()
        dg.clear()

        return g, dg
    
    else:
        plt.clf()
        plt.title("Topology Map")
        plt.ylim(-25, 55)
        plt.xlim(-40,40)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')

        for i in range(2, len(route)+1):
            r.add_node(i-1, pos=(lenDict[i-1][0], lenDict[i-1][1]))
            r.add_node(i, pos=(lenDict[i][0], lenDict[i][1]))
            r.add_edges_from([(i-1, i)])
        
        print(7777)
        nx.draw(r, nx.get_node_attributes(r, 'pos'), node_color='r', with_labels=True, node_size=180, alpha=0.5, edge_color='r', width=6)
        print(88888)
        plt.ion()     
        plt.show()
        print(99999)
        return r



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



def Remove_duplication(TheList):
    # 返回测试图像去重后的点的索引 query_index 
    temp = [TheList[0]]
    query_index = [TheList[0][1]]
    count = 0
    for i in range(1, len(TheList)):
        if(TheList[i][0] == temp[count][0] and TheList[i][2] < temp[count][2]):
            temp[count] = TheList[i]
            query_index[count] = TheList[i][1]

        elif(TheList[i][0] != temp[count][0]):
            temp.append(TheList[i])
            query_index.append(TheList[i][1])
            count+=1
    return query_index



def GetSimilarity(ImgList1, ImgList2):
    ListSURF2 = {}
    ListSURF1 = {}
    Distance = 0.36
    bf = cv2.BFMatcher_create()

    for Index1 in range(len(ImgList1)):
        ListSURF1[Index1] = ImgList1[Index1]
    
    for Index2 in range(len(ImgList2)):
        ListSURF2[Index2] = ImgList2[Index2]


    BeI = float(len(ListSURF2)) / len(ListSURF1)

    if BeI >= 1 and BeI < 2:
        p0List = []
        if len(ListSURF1) >= len(ListSURF2):
            for i in range(len(ListSURF2)):
                matches = bf.match(ListSURF2[i], ListSURF1[i])
                GoodMatches = [(match.trainIdx, match.queryIdx, match.distance) for match in matches if match.distance < Distance]
                QueryIndex = Remove_duplication(sorted(GoodMatches))
                p0 = float(len(QueryIndex)) / len(ListSURF2[i]) 
                p0List.append(p0)

                matches = bf.match(ListSURF2[i], ListSURF1[len(ListSURF1) - 1 - i])
                GoodMatches = [(match.trainIdx, match.queryIdx, match.distance) for match in matches if match.distance < Distance]
                QueryIndex = Remove_duplication(sorted(GoodMatches))
                p0 = float(len(QueryIndex)) / len(ListSURF2[i]) 
                p0List.append(p0)
            sim = round(np.mean(p0List), 2)
        else:
            for i in range(len(ListSURF1)):
                matches = bf.match(ListSURF1[i], ListSURF2[i])
                GoodMatches = [(match.trainIdx, match.queryIdx, match.distance) for match in matches if match.distance < Distance]
                QueryIndex = Remove_duplication(sorted(GoodMatches))
                p0 = float(len(QueryIndex)) / len(ListSURF1[i]) 
                p0List.append(p0)

                matches = bf.match(ListSURF1[i], ListSURF2[len(ListSURF2) - 1 - i])
                GoodMatches = [(match.trainIdx, match.queryIdx, match.distance) for match in matches if match.distance < Distance]
                QueryIndex = Remove_duplication(sorted(GoodMatches))
                p0 = float(len(QueryIndex)) / len(ListSURF1[i]) 
                p0List.append(p0)
            sim = round(np.mean(p0List), 2)

    elif BeI >= 2:
        p0List = []
        BeI = int(BeI)
        for i in range(len(ListSURF1)):
            matches = bf.match(ListSURF1[i], ListSURF2[i + BeI - 1])
            GoodMatches = [(match.trainIdx, match.queryIdx, match.distance) for match in matches if match.distance < Distance]
            QueryIndex = Remove_duplication(sorted(GoodMatches))
            p0 = float(len(QueryIndex)) / len(ListSURF1[i]) 
            p0List.append(p0)                                     

            matches = bf.match(ListSURF1[i], ListSURF2[len(ListSURF2) - i - BeI + 1])
            GoodMatches = [(match.trainIdx, match.queryIdx, match.distance) for match in matches if match.distance < Distance]
            QueryIndex = Remove_duplication(sorted(GoodMatches))
            p0 = float(len(QueryIndex)) / len(ListSURF1[i]) 
            p0List.append(p0)
        sim = round(np.mean(p0List), 2)

    else:
        p0List = []
        BeI = int(1 / BeI)
        for i in range(len(ListSURF2)):
            matches = bf.match(ListSURF2[i], ListSURF1[i + BeI - 1])
            GoodMatches = [(match.trainIdx, match.queryIdx, match.distance) for match in matches if match.distance < Distance]
            QueryIndex = Remove_duplication(sorted(GoodMatches))
            p0 = float(len(QueryIndex)) / len(ListSURF2[i]) 
            p0List.append(p0)                                     

            matches = bf.match(ListSURF2[i], ListSURF1[len(ListSURF1) - i - BeI])
            GoodMatches = [(match.trainIdx, match.queryIdx, match.distance) for match in matches if match.distance < Distance]
            QueryIndex = Remove_duplication(sorted(GoodMatches))
            p0 = float(len(QueryIndex)) / len(ListSURF2[i]) 
            p0List.append(p0)
        sim = round(np.mean(p0List), 2)
    return sim



# 查找修正点
def SearchPoint(PointId, CurP, Dirpre, graph, CoorDict):
    dx = CurP[0] - CoorDict[PointId][0]
    dy = CurP[1] - CoorDict[PointId][1]

    if Dirpre[0] == 0 and Dirpre[1] == -1:
        if dx != 0:
            Flag = 'x'
        else:
            # Flag = 'y'
            Flag = 'n'
    elif Dirpre[0] == 0 and Dirpre[1] == 1:
        if dx != 0:
            Flag = 'x'
        else:
            # Flag = 'y'
            Flag = 'n'
    elif Dirpre[0] == 1 and Dirpre[1] == 0:
        if dy != 0:
            Flag = 'y'
        else:
            # Flag = 'x'
            Flag = 'n'
    elif Dirpre[0] == -1 and Dirpre[1] == 0:
        if dy != 0:
            Flag = 'y'
        else:
            # Flag = 'x'
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
    rho =  np.rad2deg(np.arcsin(np.cross(v1, v2)/TheNorm))
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



def Translation(Info, rout):
    res = ''
    for i in range(len(Info)-1):
        res += 'At ' + str(rout[i]) + ' ' + DirConvert(Info[i]) + '. '

    res += 'Then go forward, reach the end at ' + str(rout[len(rout)-1]) + '.'
    return res



def Navigation(TopologyMap, start, end, CoordNodeDict, NowVec):
    route = nx.dijkstra_path(TopologyMap, source=start, target=end)
    AngInfo = []  

    for i in range(1, len(route)):
        checkVec = GetNormVec(CoordNodeDict[start], CoordNodeDict[route[i]])
        RotAng = GetClockAngle(NowVec, checkVec)
        
        AngInfo.append(RotAng)
        NowVec = checkVec
        start = route[i]
    
    AngInfo.append(0)
    result = Translation(AngInfo, route)
    return result



def OBJTranslation(ObjName):
    if ObjName == 'ThisIsCorner':
        return 'You have ever turned at here before.'
    else:
        TrashCanNum = ObjName[0]
        IndictorNum = ObjName[10]

    if ObjName[0] == '0':
        return 'There are ' + ObjName[10] + ' Indicators here.'
    elif ObjName[10] == '0':
        return 'There are ' + ObjName[0] + ' Trash can here.'
    else:
        return 'There are ' + ObjName[0] + ' Trash can and ' + ObjName[10] + ' Indicators here.'