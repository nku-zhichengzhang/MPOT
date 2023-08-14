import torch
import numpy as np
def make_mesh(height, width):
    yv, xv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])

    yv = yv.reshape(height*width).unsqueeze(0).float()
    xv = xv.reshape(height*width).unsqueeze(0).float()
    return yv,xv

def make_gaussian(y_idx, x_idx, yv, xv, sigma=30):
    y_idx = y_idx.transpose(0, 1)
    x_idx = x_idx.transpose(0, 1)

    g = torch.exp(- ((yv-y_idx)**2 + (xv-x_idx)**2) / (2*sigma**2))

    return g

def gaussian_radius(det_size, min_overlap):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    
    return min(r1, r2, r3)


if __name__ == '__main__':
    import cv2
    import torch
    import time
    width = 1280
    height = 720
    yv,xv = make_mesh(height,width)
    y_idx = torch.Tensor([-100,300]).view(1,2)
    x_idx = torch.Tensor([100,600]).view(1,2)
    n=4
    s = time.time()
    g = make_gaussian(y_idx.repeat(1,n), x_idx.repeat(1,n), yv, xv)
    print(time.time()-s)
    s = time.time()
    for _ in range(n*2):
        g = make_gaussian(y_idx[0:1,0:1], x_idx[0:1,0:1], yv, xv)
    print(time.time()-s)
    print(g.size())