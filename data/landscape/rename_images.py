import numpy as np
import os, glob
from tqdm import tqdm
from natsort import natsorted

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True, help='base directory where data is placed')
opt = parser.parse_args()

modes = ['sky_train', 'sky_test']
for ia, mode in enumerate(modes):

    img_path = opt.data_dir + mode + '/'
    videos = os.listdir(img_path)

    for i, vid in enumerate(tqdm(videos)):
        sub_videos = os.listdir(img_path + vid + '/')
        for j, vjd in enumerate(sub_videos):
            img_start = glob.glob(img_path + vid + '/' + vjd + '/' + '*.jpg')
            images = natsorted([img[-11:] for img in img_start])
            for k, image in enumerate(images):
                os.rename(img_start[0][:-11] + image, f'{img_path}{vid}/{vjd}/frame{k}.jpg')
