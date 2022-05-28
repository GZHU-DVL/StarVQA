import os
from fvcore.common.file_io import PathManager
import cv2

path = '/home/fengchuang/TimeSformer-main'

path_to_file=os.path.join( path,'test.csv')
data_dir = '/home/fengchuang/zax/LSVQ'
with PathManager.open(path_to_file,'r') as f:
    for clip_idx, path_label in enumerate(f.read().splitlines()):
        path, label= path_label.split(',')
        path=(f'{data_dir}/{path}.mp4')
        frame=cv2.VideoCapture(path)
        if not (frame.isOpened()):
            print(path)