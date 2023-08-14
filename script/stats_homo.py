#%%
from tqdm import tqdm
import os
from os.path import join as J
from os import listdir as D
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import math

root = '/home/ubuntu11/zzc/data/MPOT'
mpot_area = np.zeros(20)
mpot_img = np.zeros((720,1280))


def build_test_mpot_coord(frames,txt,init_txt,test_mode):
    lines = np.loadtxt(init_txt, dtype=str)[0::2]
    no = lines.shape[0]
    obj_init = [int(l.split(',')[11]) for l in lines]
    obj_ord_dict = dict(zip(obj_init,range(no)))
    # obj_init: obj id in list     no:num objects     obj_ord_dict:obj_id-order
    
    objs, ptss=[], []
    nf = len(frames)
    top_ptss = np.zeros((nf,no,8),dtype = np.float32)
    f = np.zeros((nf,no),dtype=bool)
    for frame in frames:
        # print(frame)
        pts_frame, _ = build_mpot_coord(frame,txt)
        # pts_frame: int-np8 dict
        for pt in pts_frame.keys():
            top_ptss[frame-1,obj_ord_dict[pt]]=pts_frame[pt]
            f[frame-1,obj_ord_dict[pt]]=True

    return top_ptss, f, obj_init

def build_mpot_coord(frame,txt):
    # frame: img name, int
    lines = np.loadtxt(txt, dtype=str)
    obj = []
    pts = {}
    for line in lines:
        anno = line.split(',')
        if frame == int(anno[0]):
            if anno[11]=='':
                print(anno)
                print(txt)
            pts[int(anno[11])] = np.array(anno[2:10]).astype(float)
            obj.append(int(anno[11]))
    return pts, obj

M_list=[]

for sr in ['train','test','val']:
    files = [x for x in os.listdir(J(root,sr))]
    # print(sr)
    for file in tqdm(files):
        annotxt = os.path.join(root,sr,file, 'gt/gt_obj.txt')
        annoinit = os.path.join(root, sr, file, 'gt/gt_obj_init.txt')
        imgfolder = os.path.join(root,sr,file, 'seq1')
        frames = [name[:6] for name in os.listdir(imgfolder)]
        frames.sort()
        sample_frame = frames
        coord, flag, obj_init = build_test_mpot_coord([int(name) for name in sample_frame],annotxt,annoinit,'first') 

        for obj in range(coord.shape[1]):
            obj_coords = coord[:,obj]
            obj_flags = flag[:,obj]

            st=0
            while obj_flags[st]!=True:
                st+=1
            first_coord = obj_coords[st].reshape(4,1,2)

            for fr in range(st+1, obj_coords.shape[0]):
                if obj_flags[fr]!=True:
                    continue
                cur_coord = obj_coords[fr].reshape(4,1,2)
                M, mask = cv2.findHomography(np.float32(first_coord), np.float32(cur_coord), cv2.RANSAC, 5.0)
                M_list.append(np.max(np.abs(M)))
 
f, ax = plt.subplots(figsize=(20, 18))
sns.histplot(M_list, kde=True, color=sns.xkcd_rgb['blue'], binwidth=1000,  ax=ax, label='number of relative occlusion',stat='probability')
ax.set_ylabel('Proportion',fontsize=40)
ax.set_xlabel('Maximum of Homography Matrix',fontsize=40)
plt.yticks(fontproperties='Times New Roman', size=30)
plt.xticks(fontproperties='Times New Roman', size=30)
plt.savefig('homo.png', dpi=800)