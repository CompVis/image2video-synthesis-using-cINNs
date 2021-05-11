import argparse, os, torch, random
from tqdm import tqdm
import lpips, numpy as np

from data.get_dataloder import get_eval_loader
from get_model import Model
from metrics.FVD.evaluate_FVD import compute_fvd
from metrics.FID.FID_Score import calculate_FID
from metrics.FID.inception import InceptionV3
from metrics.DTFVD import DTFVD_Score
from utils.auxiliaries import set_seed

parser = argparse.ArgumentParser()
parser.add_argument('-gpu', type=str, required=True, help="Define GPU on which to run")
parser.add_argument('-dataset', type=str)
parser.add_argument('-texture', type=str, required=False, help='Specify texture when using DTDB')
parser.add_argument('-ckpt_path', type=str, required=False, help="Specify path if outside of repo for chkpt")
parser.add_argument('-data_path', type=str, required=False, help="Path to dataset arranged as described in readme")
parser.add_argument('-seq_length', type=int, default=16)
parser.add_argument('-bs', type=int, default=6, help='Batchsize')
parser.add_argument('-FID', type=bool)
parser.add_argument('-FVD', type=bool)
parser.add_argument('-DTFVD', type=bool)
parser.add_argument('-LPIPS', type=bool)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
set_seed(249)

## Load model from config
path_ds = f'{args.dataset}/{args.texture}/' if args.dataset == 'DTDB' else f'{args.dataset}'
ckpt_path = f'./models/{path_ds}/stage2/' if not args.ckpt_path else args.ckpt_path
model = Model(ckpt_path, args.seq_length)

# set up dataloader
dataset = get_eval_loader(args.dataset, args.seq_length + 1, args.data_path, model.config)
dataloader = torch.utils.data.DataLoader(dataset, num_workers=10, batch_size=args.bs, shuffle=False)

## Generate samples
seq_real, seq_fake = [], []
with torch.no_grad():
    for batch_idx, file_dict in enumerate(tqdm(dataloader)):
        seq = file_dict["seq"].type(torch.FloatTensor).cuda()
        seq_gen = model(seq[:, 0])
        if args.dataset == 'bair':
            ## Following https://arxiv.org/abs/1812.01717 the evaluation sequence length is of length 16 after
            ## concatenating the conditioning (in our case a single frame)
            seq_gen = torch.cat((seq[:, :1], seq_gen[:, :-1]), dim=1)
            seq_real.append(seq[:, :-1].detach().cpu())
        elif args.dataset == 'iPER':
            ## For fair comparison with other methods which condition on multiple frames we concatenated only the last
            ## conditioning frame to the sequence and used all generated frames for computing FVD on iPER
            seq_gen = torch.cat((seq[:, :1], seq_gen), dim=1)
            seq_real.append(seq.detach().cpu())
        else:
            ## On dynamic textures we evaluated FVD without concatenating GT frames to the generated one
            seq_real.append(seq[:, :-1].detach().cpu())
        seq_fake.append(seq_gen.detach().cpu())

seq2 = torch.cat(seq_real, 0)
seq1 = torch.cat(seq_fake, 0)

del model
torch.cuda.empty_cache()
assert seq2.shape == seq1.shape

if args.FID or args.LPIPS:
    pd_imgs = seq1.reshape(-1, *seq1.shape[2:])
    gt_imgs = seq2.reshape(-1, *seq2.shape[2:])

if args.FID:
    print('Evaluate FID')
    inception = InceptionV3()
    batch_size = 50
    FID, _ = calculate_FID(inception, pd_imgs, gt_imgs, batch_size, 2048)
    del inception
    torch.cuda.empty_cache()
    print(f'FID score of {FID}')

if args.LPIPS:
    print('Evaluate LPIPS')
    LPIPS = 0
    lpips_vgg = lpips.LPIPS(net='vgg').cuda()
    with torch.no_grad():
        for i in range(pd_imgs.size(0)//10):
            pd_batch, gt_batch = pd_imgs[i*10:(i+1)*10], gt_imgs[i*10:(i+1)*10]
            LPIPS += lpips_vgg(pd_batch.cuda(), gt_batch.cuda()).mean().cpu().item()
    _ = lpips_vgg.cpu()
    LPIPS /= pd_imgs.size(0)//10
    del lpips_vgg
    torch.cuda.empty_cache()
    print(f'LPIPS score of {LPIPS}')

## Evaluate Dynamic Texture FVD score
if args.DTFVD:
    print('Evaluate DTFVD')
    batch_size = 40
    if args.seq_length > 16:
        I3D = DTFVD_Score.load_model(length=32).cuda()
        DTFVD = DTFVD_Score.calculate_FVD32(I3D, seq1, seq2, batch_size, True)
    else:
        I3D = DTFVD_Score.load_model(length=16).cuda()
        DTFVD = DTFVD_Score.calculate_FVD(I3D, seq1, seq2, batch_size, True)
    del I3D
    torch.cuda.empty_cache()
    print(f'DTFVD score of {DTFVD}')

if args.FVD:
    print('Evaluate FVD')
    seq1 = seq1[:seq1.size(0) // 16 * 16].reshape(-1, 16, seq1.size(1), 3, seq1.size(-1), seq1.size(-1))
    seq2 = seq2[:seq2.size(0) // 16 * 16].reshape(-1, 16, seq2.size(1), 3, seq2.size(-1), seq2.size(-1))
    fvd = compute_fvd(seq1, seq2)
    print(f'FVD score of {fvd}')