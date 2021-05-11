import argparse, os, glob, cv2, torch, math, imageio, lpips
from tqdm import tqdm
import kornia as k, numpy as np, torchvision

from get_model import Model
from utils import auxiliaries as aux
from data.get_dataloder import get_eval_loader

# setup argparser
parser = argparse.ArgumentParser()
parser.add_argument('-gpu', type=str, required=True, help="Define GPU on which to run")
parser.add_argument('-dataset', type=str, required=True, help='Specify dataset')
parser.add_argument('-data_path', type=str, required=False, help="Path to dataset arranged as described in readme")
parser.add_argument('-ckpt_path', type=str, required=False, help='If ckpt outside of repo')
parser.add_argument('-seq_length', type=int, default=16)
parser.add_argument('-n_samples', type=int, default=15, help='How many realizations generated for each test instance')
parser.add_argument('-n_realiz', type=int, default=8, help='How many realizations generated for each test instance')
parser.add_argument('-bs', type=int, default=6, help='Batchsize')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

assert args.dataset == 'bair'
path_ds = f'{args.dataset}/{args.texture}/' if args.dataset == 'DTDB' else f'{args.dataset}'
ckpt_path = f'./models/{path_ds}/stage2_control/' if not args.ckpt_path else args.ckpt_path

## Load model from config
model = Model(ckpt_path, args.seq_length)

# set up dataloader
dataset = get_eval_loader(args.dataset, args.seq_length + 1, args.data_path, model.config, control=True)
dataloader = torch.utils.data.DataLoader(dataset, num_workers=10, batch_size=args.bs, shuffle=False)

# Generate samples
seq_fake = []
with torch.no_grad():
    for _ in range(args.n_realiz):
        seq_fakes = []
        num_samples = 0
        for batch_idx, file_dict in enumerate(dataloader):
            seq = file_dict["seq"].type(torch.FloatTensor).cuda()
            seq_gen = model(seq[:, 0], cond=file_dict["cond"])
            seq_fakes.append(seq_gen.detach().cpu())
            num_samples += seq_gen.size(0)
            if num_samples >= args.n_samples:
                break
        seq_fake.append(torch.cat(seq_fakes))

videos = torch.stack(seq_fake, 1)[:args.n_samples]
del model
torch.cuda.empty_cache()

## Save video as gif
save_path = f'./assets/results/bair_endpoint/'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
for idx, vid in enumerate(videos):
    gif = aux.convert_seq2gif(vid)
    imageio.mimsave(save_path + f'endpoint_{idx}.gif', gif.astype(np.uint8), fps=3)
    torchvision.utils.save_image(vid[:, -1], save_path + f'endpoint_{idx}.png', normalize=True)

print(f'Animations saved in {save_path}')