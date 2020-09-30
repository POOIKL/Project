#!/usr/bin/env python2.7
# -*- coding: utf-8 -*
import cv2
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import kmc2
import numpy as np




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
    
    [r1, c1] = NowBrief.shape
    [r2, c2] = PrevBrief.shape

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