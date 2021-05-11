import torch.nn
import kornia as k
import numpy as np
from tqdm import tqdm
from stage2_cINN.AE.modules.vgg16 import vgg16


def compute_vgg_diversity(videos):
    """
    Computes diversity based on VGG backbone trained on ImageNet

    Input: PyTorch tensor of shape (BatchSize, NumberSamples, Time, Channel, H, W)
        Important input needs to be in range [-1, 1] !

    """

    print('Evaluate Diversity score based on VGG trained on ImageNet')

    resize = k.Resize(size=(224, 224))
    normalize = k.augmentation.Normalize(torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225]))
    vgg = vgg16().cuda()
    n_samples = videos.size(1)

    ## check if videos are in correct range
    assert videos.min() < 0
    assert videos.max() <= 1
    videos = (videos + 1)/2
    img_res = videos.size(-1)
    seq_length = videos.size(2)

    div = []
    with torch.no_grad():
        for video in tqdm(videos):
            fmap = vgg(resize(normalize(video.reshape(-1, 3, img_res, img_res)).cuda()))
            for i in range(n_samples):
                for j in range(n_samples):
                    if i != j:
                        for l in range(5):
                            f = fmap[l].reshape(n_samples, seq_length, *fmap[l].shape[1:])
                            div.append(((f[i] - f[j])**2).mean().cpu().item())

    div = np.asarray(div)
    print(f'Diversity score of {np.mean(div)} using VGG backbone')
    del vgg
    torch.cuda.empty_cache()
