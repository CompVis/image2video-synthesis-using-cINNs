import torch.nn as nn
import torch
from stage2_cINN.AE.modules.LPIPS import LPIPS
import torch.nn.functional as F
import wandb


def calculate_adaptive_weight(nll_loss, g_loss, discriminator_weight, last_layer=None):
    if last_layer is not None:
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    else:
        nll_grads = torch.autograd.grad(nll_loss, last_layer[0], retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer[0], retain_graph=True)[0]

    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    d_weight = d_weight * discriminator_weight
    return d_weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def adopt_weight(weight, epoch, threshold=0, value=0.):
    if epoch < threshold:
        weight = value
    return weight


class Loss(nn.Module):
    def __init__(self, dic):
        super(Loss, self).__init__()
        self.vgg_loss = LPIPS().cuda()
        self.kl_weight = dic['w_kl']
        self.disc_factor = 1
        self.disc_weight = 1
        self.logvar = nn.Parameter(torch.ones(size=()) * 0.0)
        self.disc_start = dic['pretrain']

    def forward(self, inp, generator, discriminator, optimizers, epoch, logger, training=True):

        if training:
            opt_gen, opt_disc = optimizers
        recon, _, p = generator(inp)
        rec_loss = torch.abs(inp.contiguous() - recon.contiguous())
        p_loss = self.vgg_loss(inp.contiguous(), recon.contiguous())
        rec_loss = rec_loss + p_loss

        kl_loss = p.kl()

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        # generator update
        logits_fake = discriminator(recon)
        g_loss = -torch.mean(logits_fake)

        try:
            d_weight = calculate_adaptive_weight(nll_loss, g_loss, self.disc_weight, last_layer=list(generator.parameters())[-1])
        except RuntimeError:
            assert not training
            d_weight = torch.tensor(0.0)

        disc_factor = adopt_weight(self.disc_factor, epoch, threshold=self.disc_start)
        loss = nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

        if training:
            opt_gen.zero_grad()
            loss.backward()
            opt_gen.step()

        logits_real = discriminator(inp.contiguous().detach())
        logits_fake = discriminator(recon.contiguous().detach())

        disc_factor = adopt_weight(self.disc_factor, epoch, threshold=self.disc_start)
        d_loss = disc_factor * hinge_d_loss(logits_real, logits_fake)

        if training and d_loss.item() > 0:
            opt_disc.zero_grad()
            d_loss.backward()
            opt_disc.step()

        loss_dic = {
            "Loss": loss.item(),
            "Loss_recon": rec_loss.mean().item(),
            "Loss_nll": nll_loss.item(),
            "Logvar": self.logvar.detach().item(),
            "L_KL": kl_loss.item(),
            "Loss_G": g_loss.item(),
            "L_disc": d_loss.item(),
            "Logits_real": logits_real.mean().item(),
            "Logits_fake": logits_fake.mean().item(),
            "Disc_weight": d_weight.item(),
            "Disc_factor": disc_factor,
        }

        logger.append(loss_dic)

        prefix = 'train' if training else 'eval'
        loss_dic = {prefix + '_' + key:val for key, val in loss_dic.items()}
        wandb.log(loss_dic)

        return recon.cpu(), rec_loss.mean().item()

