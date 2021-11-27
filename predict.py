import os
import cv2
import math
import tempfile
import imageio
import torch
from pathlib import Path
import cog
import kornia as k
import numpy as np
from get_model import Model
from utils import auxiliaries as aux


class Predictor(cog.Predictor):
    def setup(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        seq_length = 16
        self.models = {'DTDB': {}}
        for key in ['landscape', 'bair', 'iPER']:
            ckpt_path = f'models/{key}/stage2/'
            self.models[key] = Model(ckpt_path, seq_length)
        for texture in ['clouds', 'fire', 'vegetation', 'waterfall']:
            ckpt_path = f'models/DTDB/{texture}/stage2/'
            self.models['DTDB'][texture] = Model(ckpt_path, seq_length)
        print("Models loaded!")

    @cog.input(
        "image",
        type=Path,
        help="input image, support 'jpg', 'png', 'jpeg' files",
    )
    @cog.input(
        "model_type",
        type=str,
        options=['landscape', 'bair', 'iPER', 'DTDB'],
        help="choose model type",
    )
    @cog.input(
        "texture",
        type=str,
        default='clouds',
        options=['clouds', 'fire', 'vegetation', 'waterfall'],
        help="Specify texture when using DTDB, valid only for DTDB",
    )
    def predict(self, image, model_type, texture):
        assert str(image).split('.')[-1] in ['jpg', 'png', 'jpeg'], "input image should be 'jpg', 'png' or 'jpeg' file"
        batch_size = 6
        model = self.models[model_type][texture] if model_type=='DTDB' else self.models[model_type]
        img_res = model.config.Data['img_size']
        resize = k.Resize(size=(img_res, img_res))
        normalize = k.augmentation.Normalize(0.5, 0.5)
        imgs = [resize(normalize(k.image_to_tensor(cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)) / 255.0))]
        imgs = torch.cat(imgs)

        ## Generate videos
        length = math.ceil(imgs.size(0) / batch_size)
        videos = []
        with torch.no_grad():
            for i in range(length):

                if i < (length - 1):
                    batch = imgs[i * batch_size:(i + 1) * batch_size].cuda()
                else:
                    batch = imgs[i * batch_size:].cuda()
                videos.append(model(batch).cpu())

        videos = torch.cat(videos)

        out_path = Path(tempfile.mkdtemp()) / "out.gif"
        gif = aux.convert_seq2gif(videos)
        imageio.mimsave(str(out_path), gif.astype(np.uint8), fps=3)
        imageio.mimsave('000.gif', gif.astype(np.uint8), fps=3)
        return out_path
