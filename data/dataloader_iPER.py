import torch, torch.nn as nn
import numpy as np
import kornia as k, os, glob, cv2
from torchvision import transforms
from data.augmentation import Augmentation


class Dataset(torch.utils.data.Dataset):

    def __init__(self, opt, mode):

        self.mode = mode
        self.data_path = opt.Data['data_path']
        self.seq_length = opt.Data['sequence_length']
        self.img_size = opt.Data['img_size']
        self.do_aug = opt.Data['aug']

        self.videos = []; self.num_frames = []

        ## Create list of videos and number of frames per video (train + eval)
        print(f"Setup dataloder {mode}")
        file = 'train.txt' if mode == 'train' else 'val.txt'
        videos = open('data/iPER/' + file, 'r').read().split()
        for i, vid in enumerate(videos):
            vid = vid.replace('/', '_')
            n_frames = len(glob.glob(self.data_path + vid + '/' + '*.png'))
            if n_frames < self.seq_length:
                continue
            for _ in range(opt.Data[f'iter_{mode}']):
                for _ in range(int(vid[-1])):
                    self.videos.append(vid)
                    self.num_frames.append(n_frames)

        self.length = len(self.videos)
        self.aug_train = Augmentation(self.img_size, opt.Data.Augmentation)
        self.aug_test = torch.nn.Sequential(
                        k.Resize(size=(self.img_size, self.img_size)),
                        k.augmentation.Normalize(0.5, 0.5))

    def __len__(self):
        return self.length

    def load_img(self, video, frame):
        img = cv2.imread(self.data_path  + video +  f'/frame_{frame}.png')
        return k.image_to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))/255.0

    def __getitem__(self, idx):
        video = self.videos[idx]
        frames = np.arange(0, self.num_frames[idx])

        ## Load sequence
        start_rand = np.random.randint(0, len(frames) - self.seq_length + 1)
        seq = torch.stack([self.load_img(video, frames[start_rand + i]) for i in range(self.seq_length)], dim=0)
        seq = self.aug_train(seq) if self.mode == 'train' and self.do_aug else self.aug_test(seq)

        sample = {'seq': seq}
        return sample


class DatasetEvaluation(torch.utils.data.Dataset):

    def __init__(self, seq_length, img_size, path):

        self.data_path = path
        self.seq_length = seq_length
        self.img_size = img_size

        self.videos = []; self.num_frames = []
        videos = open('data/iPER/test.txt', 'r').read().split()
        for i, vid in enumerate(videos):
            vid = vid.replace('/', '_')
            n_frames = len(glob.glob(path + vid + '/' + '*.png'))
            if n_frames < self.seq_length:
                continue
            self.videos.append(vid)
            self.num_frames.append(n_frames)

        self.length = 1000
        self.num_videos = len(self.videos)
        self.aug_test = torch.nn.Sequential(
                            k.Resize(size=(self.img_size, self.img_size)),
                            k.augmentation.Normalize(0.5, 0.5))

    def __len__(self):
        return self.length

    def load_img(self, video, frame):
        img = cv2.imread(self.data_path  + video + '/' + 'frame_' + str(int(frame)) + '.png')
        return k.image_to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))/255.0

    def __getitem__(self, idx):
        video = self.videos[idx%self.num_videos]
        frames = np.arange(0, self.num_frames[idx%self.num_videos])

        ## Load sequence
        start_rand = np.random.randint(0, len(frames) - self.seq_length + 1)
        seq = torch.stack([self.load_img(video, frames[start_rand + i]) for i in range(self.seq_length)], dim=0)
        return {'seq': self.aug_test(seq)}
