import argparse, os, glob, cv2, torch, math, imageio, lpips
from tqdm import tqdm
import kornia as k, numpy as np

from get_model import Model
from utils import auxiliaries as aux

img_suffix = ['jpg', 'png', 'jpeg']

# setup argparser
parser = argparse.ArgumentParser()
parser.add_argument('-gpu', type=str, required=True, help="Define GPU on which to run")
parser.add_argument('-dataset', type=str, required=True, help='Specify dataset')
parser.add_argument('-texture', type=str, help='Specify texture when using DTDB')
parser.add_argument('-ckpt_path', type=str, required=False, help='If ckpt outside of repo')
parser.add_argument('-seq_length', type=int, default=16)
parser.add_argument('-bs', type=int, default=6, help='Batchsize')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

path_ds = f'{args.dataset}/{args.texture}' if args.dataset == 'DTDB' else f'{args.dataset}'
ckpt_path = f'./models/{path_ds}/stage2/' if not args.ckpt_path else args.ckpt_path
img_path = f'./assets/GT_samples/{path_ds}/'

## get all images (jpg, png, jpeg) in folder
img_list = []
for suffix in img_suffix:
    img_list.extend(glob.glob(img_path + f'*.{suffix}'))

## Load model from config
model = Model(ckpt_path, args.seq_length)

## Load images
img_res = model.config.Data['img_size']
resize = k.Resize(size=(img_res, img_res))
normalize = k.augmentation.Normalize(0.5, 0.5)

imgs = [resize(normalize(k.image_to_tensor(cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB))/255.0))
        for name in img_list]
imgs = torch.cat(imgs)

## Generate videos
bs = args.bs
length = math.ceil(imgs.size(0)/bs)
videos = []
with torch.no_grad():
    for i in range(length):

        if i < (length -1):
            batch = imgs[i * bs:(i + 1) * bs].cuda()
        else:
            batch = imgs[i * bs:].cuda()
        videos.append(model(batch).cpu())

videos = torch.cat(videos)

## Save video as gif
save_path = f'./assets/results/{path_ds}/'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
gif = aux.convert_seq2gif(videos)
imageio.mimsave(save_path + f'results.gif', gif.astype(np.uint8), fps=3)
print(f'Animations saved in {save_path}')