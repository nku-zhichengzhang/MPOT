import numpy as np
from math import exp, log, sqrt, ceil
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import time
def cal_sigma(dmax, edge_value):
    return sqrt(- pow(dmax, 2) / log(edge_value))

def gaussian(array_like_hm, mean, sigma):
    """modifyed version normal distribution pdf, vector version"""
    array_like_hm -= mean
    x_term = array_like_hm[:,0] ** 2
    y_term = array_like_hm[:,1] ** 2
    exp_value = - (x_term + y_term) / 2 / pow(sigma, 2)
    return np.exp(exp_value)

def draw_heatmap(width, height, x, y, sigma, array_like_hm):
    m1 = (x, y)
    s1 = np.eye(2) * pow(sigma, 2)
    k1 = multivariate_normal(mean=m1, cov=s1)
    #     zz = k1.pdf(array_like_hm)
    zz = gaussian(array_like_hm, m1, sigma)
    img = zz.reshape((height,width))*1.001
    # if x >= 0 and y >= 0 and x < width and y < height:
    #     img[y,x]=1.0
    img = np.clip(img,0,1)
    return img

def G_map(width, height, x, y, array, dmax=100):
    # dmax = int(height/5)
    edge_value = 0.01
    sigma = cal_sigma(dmax, edge_value)
    
    
    return draw_heatmap(width, height, x, y, sigma, array)

def mesh(width,height):
    xlim = (0, width)
    ylim = (0, height)

    xa = np.arange(width, dtype=np.float)
    ya = np.arange(height, dtype=np.float)
    xx, yy = np.meshgrid(xa,ya)
    xxyy = np.c_[xx.ravel(), yy.ravel()]
    return xxyy.copy()

if __name__ == '__main__':
    import cv2
    import torch
    width = 1280
    height = 720
    m = mesh(width,height)
    s = time.time()
    for _ in range(20):
        gmap = G_map(width, height, -200, 5, m.copy())*255
        # print(gmap[0,0])
        # cv2.imwrite('try/a.jpg',gmap)
        # cv2.waitKey(0)
        img = torch.from_numpy(G_map(width, height, 2, 5, m.copy()))
    print(time.time()-s)