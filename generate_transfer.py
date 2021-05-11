import argparse, os, glob, cv2, torch, math, imageio, lpips
from tqdm import tqdm
import kornia as k, numpy as np
from natsort import natsorted

from get_model import Model
from utils import auxiliaries as aux

img_suffix = ['jpg', 'png', 'jpeg']

# setup argparser
parser = argparse.ArgumentParser()
parser.add_argument('-gpu', type=str, required=True, help="Define GPU on which to run")
parser.add_argument('-dataset', type=str, required=True, help='Specify dataset')
parser.add_argument('-ckpt_path', type=str, required=False)
parser.add_argument('-seq_length', type=int, default=16)
parser.add_argument('-bs', type=int, default=6, help='Batchsize')
args = parser.parse_args()

assert args.dataset == 'landscape', 'Only implemented for landscape'

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
ckpt_path = f'./models/{args.dataset}/stage2/' if not args.ckpt_path else args.ckpt_path

## Load model from config
model = Model(ckpt_path, args.seq_length, transfer=True)

img_path = f'./assets/GT_samples/{args.dataset}/transfer/'

## Load sequences
video_paths = natsorted(os.listdir(img_path))
img_res = model.config.Data['img_size']
videos = []
for vidp in video_paths:
    ## get all images in folder
    img_list = []
    for suffix in img_suffix:
        img_list.extend(glob.glob(img_path + vidp + '/' + f'*.{suffix}'))

    img_list = natsorted(img_list)[:args.seq_length]
    resize = k.Resize(size=(img_res, img_res))
    normalize = k.augmentation.Normalize(0.5, 0.5)
    seq = [k.image_to_tensor(cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB))/255.0 for name in img_list]
    videos.append(resize(normalize(torch.stack(seq))))

videos = torch.stack(videos)

## Create transfer for each video provided in samples
bs = 6
length = math.ceil(videos.size(0) / bs)

for idx, query in enumerate(videos):
    transfer = []
    with torch.no_grad():
        for i in range(length):

            if i < (length -1):
                batch = videos[i * bs:(i + 1) * bs, 0].cuda()
            else:
                batch = videos[i * bs:, 0]
            transfer.append(model.transfer(query[None, :].cuda(), batch.cuda()).cpu())

    transfer = torch.cat(transfer)
    save_path = f'./assets/results/{args.dataset}/'
    ## Save video as gif
    transfer = torch.cat((query[None, :], transfer), dim=0)
    gif = aux.convert_seq2gif(transfer)
    imageio.mimsave(save_path + f'transfer_{idx}.gif', gif.astype(np.uint8), fps=3)
