import numpy as np
import cv2
import time

def num2deg(x):
    return x/180*np.pi

def deg2num(d):
    return d*180/np.pi

def DecomposeH2motion(H):
    # t
    tx = H[0][2]
    ty = H[1][2]

    # v
    v1 = H[2][0]
    v2 = H[2][1]

    # A
    A = H[:2,:2]
    sRK = A - H[:2,2:].dot(H[2:,:2])

    # theta
    theta=np.arctan(sRK[1][0]/sRK[0][0])

    # scale
    fenzi = sRK[1][1]*np.cos(theta)-sRK[0][1]*np.sin(theta)
    fenmu = (np.cos(theta)**3)/sRK[0][0] + (np.sin(theta)**3)/sRK[1][0]
    s = np.sqrt(fenzi/fenmu)

    # k
    k1 = sRK[1][0]/s/np.sin(theta)
    k2 = (np.sin(theta)/k1+sRK[0][1]/s) / np.cos(theta)
    
    return {'tx':tx,'ty':ty,'v1':v1,'v2':v2,'theta':theta,'scale':s,'k1':k1,'k2':k2}

def ReconMotion2H(M):
    Hp = np.array([M['scale']*np.cos(M['theta']), -M['scale']*np.sin(M['theta']), M['tx'], M['scale']*np.sin(M['theta']), M['scale']*np.cos(M['theta']), M['ty'], 0, 0, 1]).reshape(3,3)
    Ha = np.array([M['k1'], M['k2'], 0, 0, 1/M['k1'], 0, 0, 0, 1]).reshape(3,3)
    Hs = np.array([1, 0, 0, 0, 1, 0, M['v1'], M['v2'], 1]).reshape(3,3)
    ReconH = Hp.dot(Ha).dot(Hs)
    return ReconH

if __name__ == '__main__':
    src_pts = np.array([100, 93, 238, 82, 300, 501, 122, 399]).reshape(4,1,2)
    dst_pts = np.array([400, 100, 500, 0, 510, 390, 410, 410]).reshape(4,1,2)
    pic1 = np.zeros((720,1280,3),dtype=np.uint8)
    pic2 = np.zeros((720,1280,3),dtype=np.uint8)
    cv2.polylines(pic1, [src_pts.astype(np.int32)], True, (0,255,255))
    cv2.imwrite('test_init.jpg', pic1)
    cv2.polylines(pic2, [dst_pts.astype(np.int32)], True, (0,255,255))
    cv2.imwrite('test_targ.jpg', pic2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    st1 = time.time()
    M = DecomposeH2motion(H)
    ed1 = time.time()
    st2 = time.time()
    rH = ReconMotion2H(M)
    ed2 = time.time()
    print('Decompose')
    print(ed1-st1)
    print(M)
    print()
    print('Reconstruction')
    print(ed2-st2)
    print(rH)