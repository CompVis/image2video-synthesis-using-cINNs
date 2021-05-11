import numpy as np, math
import torch, os
from omegaconf import OmegaConf

from stage2_cINN.modules import INN
from stage1_VAE.modules import decoder
from stage1_VAE.modules.resnet3D import Encoder


class Model(torch.nn.Module):
    def __init__(self, model_path, vid_length, transfer=False):
        super().__init__()

        ### Load cINN config if evaluation of final model is intended
        opt = OmegaConf.load(model_path + 'config_stage2.yaml')
        path_stage1 = opt.First_stage_model['model_path'] + opt.First_stage_model['model_name'] + '/'

        ## Load config for first stage model
        config = OmegaConf.load(path_stage1 + 'config_stage1.yaml')

        ## Load VAE
        self.decoder = decoder.Generator(config.Decoder).cuda()
        self.decoder.load_state_dict(torch.load(path_stage1 + opt.First_stage_model['checkpoint_decoder'] +
                                                '.pth')['state_dict'])
        _ = self.decoder.eval()

        if transfer:
            self.encoder = Encoder(dic=config.Encoder).cuda()
            self.encoder.load_state_dict(torch.load(path_stage1 + opt.First_stage_model['checkpoint_encoder'] +
                                                    '.pth.tar')['state_dict'])
            _ = self.encoder.eval()

        ## Load cINN
        flow_mid_channels = config.Decoder["z_dim"] * opt.Flow["flow_mid_channels_factor"]
        self.flow = INN.SupervisedTransformer(flow_in_channels=config.Decoder["z_dim"],
                                              flow_embedding_channels=opt.Conditioning_Model['z_dim'],
                                              n_flows=opt.Flow["n_flows"],
                                              flow_hidden_depth=opt.Flow["flow_hidden_depth"],
                                              flow_mid_channels=flow_mid_channels,
                                              flow_conditioning_option="None",
                                              dic=opt.Conditioning_Model,
                                              control=opt.Training['control']).cuda()
        self.flow.flow.load_state_dict(torch.load(model_path + 'cINN.pth')['state_dict'])

        _ = self.flow.eval()

        self.z_dim = config.Decoder["z_dim"]
        self.vid_length = vid_length
        self.config = opt

    def forward(self, x_0, cond=None):
        """
            Input: x_0 (start frame) should be of shape (BS, C, H, W)

            Output: sequence of shape (BS, T, C, H, W)
        """

        ## Draw a residual from Gaussian Normal Distribution
        residual = torch.randn(x_0.size(0), self.z_dim).cuda()

        ## Define conditioning
        cond = [x_0, cond]

        ## Use cINN with residual and x_0 (start frame) to obtain the video representation z
        z = self.flow(residual, cond, reverse=True).view(x_0.size(0), -1)

        ## Render sequence using generator/decoder
        seq = self.decoder(x_0, z)

        ## Apply multiple times if longer sequence is needed
        while seq.shape[1] < self.vid_length:
            seq1 = self.decoder(seq[:, -1], z)
            seq = torch.cat((seq, seq1), dim=1)

        return seq[:self.vid_length]

    def transfer(self, seq_query, x_0):
        """
            Input
                - query sequence seq_query of shape (BS, T, C, H, W)
                - random starting frame to which the motion should be transferred to of shape (BS, C, H, W)
            Output
                - sequence of shape (BS, T, C, H, W)
        """

        ## Obtain video representation for query sequence
        _, z, _ = self.encoder(seq_query[:, 1:].transpose(1, 2))

        ## Obtain residual using cINN (independent of appearance)
        res, _  = self.flow(z, [seq_query[:, 0]])

        ## Obtain video representation with other appearance but same motion as z
        z_ref = self.flow(res.view(z.size(0), -1).repeat(x_0.size(0), 1), [x_0], reverse=True).view(x_0.size(0), -1)

        ## Animate sequence using decoder/generator
        seq_gen = self.decoder(x_0, z_ref)

        ## Apply multiple times to obtain longer sequence
        while seq_gen.shape[1] < self.vid_length:
            seq1 = self.decoder(seq_gen[:, -1], z_ref)
            seq_gen = torch.cat((seq_gen, seq1), dim=1)

        return seq_gen
