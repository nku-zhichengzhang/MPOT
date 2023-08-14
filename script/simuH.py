import numpy as np
import cv2
from decompH import DecomposeH2motion, ReconMotion2H
from tqdm import tqdm
from os.path import join as J
from os import listdir as D
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import torch
import torch

rtx = [-8,8]
rty = [-8,8]
rv1 = [-0.0015, 0.0015]
rv2 = [-0.0015, 0.0015]
rtheta = [-0.7, 0.7]
rscale = [1/1.38, 1.38]
rk1 = [-0.1, 0.1]
rk2 = [-0.0015, 0.0015]
ORIQUAD = np.array([-1, 1, 1, 1, 1, -1, -1, -1]).reshape(4,1,2)
SIMULATION_NUMBER_POINT = 10
iH, iW = 500, 500

def getrandM():
    tx = np.random.uniform(rtx[0], rtx[-1])
    ty = np.random.uniform(rty[0], rty[-1])
    v1 = np.random.uniform(rv1[0], rv1[-1])
    v2 = np.random.uniform(rv2[0], rv2[-1])
    theta = np.random.uniform(rtheta[0], rtheta[-1])
    scale = np.random.uniform(rscale[0], rscale[-1])
    k1 = np.random.uniform(rk1[0], rk1[-1])
    k2 = np.random.uniform(rk2[0], rk2[-1])
    return tx, ty, v1, v2, theta, scale, k1, k2

def getSimuQuad(H):
    dst = cv2.perspectiveTransform(ORIQUAD.astype(float), H)
    return dst


def getNQuad():
    Quads = []
    for _ in range(int(SIMULATION_NUMBER_POINT)):
        tx, ty, v1, v2, theta, scale, k1, k2 = getrandM()
        M = {'tx':tx,'ty':ty,'v1':v1,'v2':v2,'theta':theta,'scale':scale,'k1':k1,'k2':k2}
        H = ReconMotion2H(M)
        Quads.append(getSimuQuad(H))
    return np.array(Quads)

centerQuads = getNQuad() + np.array([250,250])
H_cond_map = np.zeros((iH,iW))
H_max_map = np.zeros((iH,iW))
M_max_map = np.zeros((iH,iW))


for x in tqdm(range(iW)):
    for y in range(iH):
        curQuads = getNQuad() + np.array([x,y])
        for centerQuad in centerQuads:
            for curQuad in curQuads:
                H, _ = cv2.findHomography(np.float32(curQuad), np.float32(centerQuad), cv2.RANSAC, 5.0)
                H_cond = np.linalg.cond(H)
                H_max = np.max(H)
                M = DecomposeH2motion(H)
                M['tx']=M['tx']/iW
                M['ty']=M['ty']/iH
                M_max = max(M.values())
                
                H_cond_map[y,x] = H_cond
                H_max_map[y,x] = H_max
                M_max_map[y,x] = M_max
    print(max(H_cond_map[:,x]), min(H_cond_map[:,x]))
    print(max(H_max_map[:,x]), min(H_max_map[:,x]))
    print(max(M_max_map[:,x]), min(M_max_map[:,x]))

np.save('res_H_state/H_cond_map.npy', H_cond_map)
np.save('res_H_state/H_max_map.npy', H_max_map)
np.save('res_H_state/M_max_map.npy', M_max_map)
# fig = plt.figure()  #定义新的三维坐标轴
# ax3 = plt.axes(projection='3d')