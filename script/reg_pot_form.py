import os
import numpy as np
from tqdm import tqdm
import cv2
import shutil
saveroot = '/home/ubuntu/zzc/data/POT/test'
rawroot = '/home/ubuntu/zzc/data/POT280Data'
annoroot = '/home/ubuntu/zzc/data/annotation'

for scene in os.listdir(rawroot):
    for vid in tqdm(os.listdir(os.path.join(rawroot,scene,scene))):
        vidname = vid.split('.')[0]
        savevid = os.path.join(saveroot,vidname)
        saveanno = os.path.join(savevid,'gt')
        saveannotxt = os.path.join(saveanno,'gt.txt')
        saveinfo = os.path.join(savevid,'seqinfo.ini')
        saveimgdir = os.path.join(savevid,'seq1')
        os.makedirs(savevid, exist_ok=True)
        os.makedirs(saveanno, exist_ok=True)
        os.makedirs(saveimgdir, exist_ok=True)
        # gt
        shutil.copyfile(os.path.join(annoroot,vidname+'_gt_points.txt'), saveannotxt)
        # imgs
        cap = cv2.VideoCapture(os.path.join(rawroot,scene,scene,vid))
        frame_count = 0 # 保存帧的索引
        success=True
        while(success):
            success, frame = cap.read()
            if success:
                cv2.imwrite(os.path.join(saveimgdir,str(frame_count).zfill(6)+'.jpg'), frame)
                frame_count += 1
        assert frame_count==501
        
        with open(saveinfo,'w+')as txt:
            txt.write('[Sequence]\nname = '+vidname+'\nimdir = seq1\nframerate = 30\nseqlength = 501\nimwidth = 1280\nimheight = 720\nimext = .jpg\n\n')
        
