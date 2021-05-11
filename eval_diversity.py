import argparse, os, torch, numpy as np
from tqdm import tqdm

from data.get_dataloder import get_eval_loader
from get_model import Model
from metrics.Diversity.VGG import compute_vgg_diversity
from metrics.Diversity.I3D import compute_I3D_diversity, compute_DTI3D_diversity
from utils.auxiliaries import set_seed

# setup argparser
parser = argparse.ArgumentParser()
parser.add_argument('-gpu', type=str, required=True, help="Define GPU on which to run")
parser.add_argument('-dataset', type=str, required=True, help='Specify dataset')
parser.add_argument('-texture', type=str, help='Specify texture when using DTDB')
parser.add_argument('-ckpt_path', type=str, required=False)
parser.add_argument('-data_path', type=str, required=True, default='/export/scratch/compvis/datasets/iPER/processed_256_resized/')
parser.add_argument('-seq_length', type=int, default=16, help='Number of frames to predict')
parser.add_argument('-n_realiz', type=int, default=5, help='How many samples should be generated for each test instance')
parser.add_argument('-bs', type=int, default=6, help='Batchsize')
parser.add_argument('-I3D', type=bool, help='Evaluation using kinetics I3D backbone')
parser.add_argument('-VGG', type=bool, help='Evaluation using VGG backbone')
parser.add_argument('-DTI3D', type=bool, help='Evaluation using DTDB I3D backbone')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
set_seed(249)

## Load model from config
path_ds = f'{args.dataset}/{args.texture}/' if args.dataset == 'DTDB' else f'{args.dataset}'
ckpt_path = f'./models/{path_ds}/stage2/' if not args.ckpt_path else args.ckpt_path

model = Model(ckpt_path, args.seq_length)
#
# set up dataloader
dataset = get_eval_loader(args.dataset, args.seq_length, args.data_path, model.config)
dataloader = torch.utils.data.DataLoader(dataset, num_workers=10, batch_size=args.bs, shuffle=False)

# Generate samples
seq_fake = []
with torch.no_grad():
    for _ in range(args.n_realiz):
        seq_fakes = []
        for batch_idx, file_dict in enumerate(tqdm(dataloader)):
            seq = file_dict["seq"].type(torch.FloatTensor).cuda()
            seq_gen = model(seq[:, 0])
            seq_fakes.append(seq_gen.detach().cpu())
        seq_fake.append(torch.cat(seq_fakes))

seq1 = torch.stack(seq_fake, 1)

del model
torch.cuda.empty_cache()

if args.VGG:
    compute_vgg_diversity(seq1)
if args.DTI3D:
    div = compute_DTI3D_diversity(seq1)
if args.I3D:
    compute_I3D_diversity(seq1, args.n_realiz)
