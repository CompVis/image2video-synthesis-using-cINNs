import numpy as np, csv
import torch, os, random
import imageio, wandb
from tqdm import tqdm
from metrics.PyTorch_FVD.FVD_logging import calculate_FVD as calc_FVD
from metrics.DTFVD.DTFVD_Score import calculate_FVD as calc_DTFVD

def get_save_dict(model, optmizer, scheduler, epoch):
    dic = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
           'optim_state_dict': optmizer.state_dict(),
           'scheduler_state_dict': scheduler.state_dict()}
    return dic

### Logging and converting of videos
def convert_seq2gif(sequence):
    img_shape = sequence.shape
    images_orig = denorm(sequence).permute(0, 1, 3, 4, 2).detach().cpu().numpy()
    img_gif = images_orig[0]
    for i in range(1, img_shape[0]):
        img_gif = np.concatenate((img_gif, images_orig[i]), axis=2)
    img_gif = 255 * img_gif / np.max(img_gif)
    return img_gif


def save_video(path, video):
    writer = imageio.get_writer(path, fps=3)
    long_video = np.tile(video, (6, 1, 1, 1))
    for im in long_video:
        writer.append_data(im)
    writer.close()


def plot_vid(opt, sequences, epoch=0, mode='train', path=None, axis=1):
    sequence_orig, sequence_gen = sequences

    ## Save sequences as GIF's
    seq_orig = convert_seq2gif(sequence_orig)
    seq_gen = convert_seq2gif(sequence_gen)
    seq = np.concatenate((seq_gen, seq_orig), axis=axis)
    x, y = seq.shape[1] // 16 * 16, seq.shape[2] // 16 * 16
    seq = seq[:, :x, :y]
    if path is None:
        imageio.mimsave(opt.Training['save_path'] + '/videos/{:03d}_sequence_'.format(epoch + 1) + mode + '.gif',
                        seq.astype(np.uint8), fps=3)
        save_video(opt.Training['save_path'] + '/videos/{:03d}_sequence_'.format(epoch + 1) +
                   mode + '.mp4', seq.astype(np.uint8))
    else:
        imageio.mimsave(path + 'seq.gif',seq.astype(np.uint8), fps=3)
        save_video(path + 'seq.mp4', seq.astype(np.uint8))
    return seq.astype(np.uint8).transpose(0, 3, 1, 2)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

### Evaluate FVD score using PyTorch FVD during training of posterior (1st stage- reconstruction) and prior (2nd stage)
def evaluate_FVD_posterior(dloader, model, encoder, I3D, mode):

    seq_g, seq_o = [], []

    with torch.no_grad():
        for _, file in enumerate(dloader):
            seq = file["seq"].type(torch.FloatTensor).cuda()
            motion, *_ = encoder(seq[:, 1:].transpose(1, 2))
            seq_gen = model(seq[:, 0], motion)
            seq_g.append(seq_gen.cpu())
            seq_o.append(seq[:, 1:].cpu())

    seq_gen  = torch.cat(seq_g, dim=0)
    seq_orig = torch.cat(seq_o, dim=0)

    torch.cuda.empty_cache()

    FVD = calc_FVD(I3D, seq_orig, seq_gen, 20) if mode=='FVD' else calc_DTFVD(I3D, seq_orig, seq_gen, 40, True)
    return FVD


def evaluate_FVD_prior(dloader, cINN, decoder, I3D, z_dim, opt, epoch, mode, control):

    seq_g, seq_o = [], []

    with torch.no_grad():
        for _, file in enumerate(tqdm(dloader)):
            seq = file["seq"].type(torch.FloatTensor).cuda()

            res = torch.randn(seq.size(0), z_dim).cuda()
            cond = [seq[:, 0]] if not control else [seq[:, 0], file["cond"]]
            z   = cINN(res, cond, reverse=True).view(seq.size(0), -1)
            seq_gen = decoder(seq[:, 0], z)
            seq_g.append(seq_gen.cpu())
            seq_o.append(seq[:, 1:].cpu())

    seq_gen  = torch.cat(seq_g, dim=0)
    seq_orig = torch.cat(seq_o, dim=0)

    ## save some random samples
    rand_sel = np.random.randint(0, seq_gen.size(0), 10)
    video = plot_vid(opt, [seq_gen[rand_sel], seq_orig[rand_sel]], epoch, mode='eval')
    wandb.log({"eval_video": wandb.Video(video, fps=3,  format="gif")})
    torch.cuda.empty_cache()

    FVD = calc_FVD(I3D, seq_orig, seq_gen, 20) if mode=='FVD' else calc_DTFVD(I3D, seq_orig, seq_gen, 40, True)
    return FVD


### Offline Logging
class CSVlogger:
    def __init__(self, logname, header_names):
        self.header_names = header_names
        self.logname = logname
        with open(logname, "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(header_names)

    def write(self, inputs):
        with open(self.logname, "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(inputs)


class Logging:
    def __init__(self, keys):
        super(Logging, self).__init__()
        self.keys = keys
        self.dic = {x: np.array([]) for x in self.keys}

    def reset(self):
        self.dic = {x: np.array([]) for x in self.keys}

    def append(self, loss_dic):
        for key, val in self.dic.items():
            self.dic[key] = np.append(val, loss_dic[key])

    def get_iteration_mean(self, horizon=50):
        mean = []
        for key, val in self.dic.items():
            if len(val) < horizon:
                mean.append(np.mean(val))
            else:
                mean.append(np.mean(val[-horizon:]))
        return mean

    def log(self):
        mean = []
        for key, val in self.dic.items():
            mean.append(np.mean(val))
        return mean

