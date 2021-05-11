import cv2, torch, torch.nn as nn
import numpy as np
import kornia as k, os, glob
from data.augmentation import Augmentation_random_crop


class Dataset(torch.utils.data.Dataset):

    def __init__(self, opt, mode):

        self.data_path = opt.Data['data_path']
        self.prefix = 'sky_train' if mode != 'test' else 'sky_test'

        self.seq_length = opt.Data['sequence_length']
        self.do_aug = opt.Data['aug']

        path = f'data/landscape/{mode}.txt'
        video_list = open(path).read().split()
        self.videos = []; self.num_frames = []

        for vid in video_list:
            length = len(glob.glob(self.data_path + self.prefix + '/' + vid + '/*.jpg'))

            for _ in range(opt.Data[f'iter_{mode}']):
                self.videos.append(self.prefix + '/' + vid)
                self.num_frames.append(length)

        self.length = len(self.videos)

        if mode == 'train' and self.do_aug:
            self.aug = Augmentation_random_crop(opt.Data['img_size'], opt.Data.Augmentation)
        else:
            self.aug = torch.nn.Sequential(
                        k.Resize(size=(opt.Data['img_size'], opt.Data['img_size'])),
                        k.augmentation.Normalize(0.5, 0.5))

    def __len__(self):
        return self.length

    def load_img(self, video, frame):
        img = cv2.imread(self.data_path + video + '/' + 'frame' + str(int(frame)) + '.jpg')
        return k.image_to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))/255.0

    def __getitem__(self, idx):
        video  = self.videos[idx]
        frames = np.arange(0, self.num_frames[idx])

        ## Sample random starting point in the sequence
        start_rand = np.random.randint(0, len(frames) - self.seq_length + 1)

        seq = torch.stack([self.load_img(video, frames[start_rand + i]) for i in range(self.seq_length)], dim=0)
        return {'seq': self.aug(seq)}

