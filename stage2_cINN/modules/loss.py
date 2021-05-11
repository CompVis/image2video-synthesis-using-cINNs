import torch.nn as nn
import torch
import wandb

class FlowLoss(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, sample, logdet, logger, mode='eval'):
        nll_loss = torch.mean(nll(sample))
        assert len(logdet.shape) == 1
        nlogdet_loss = -torch.mean(logdet)
        loss = nll_loss + nlogdet_loss
        reference_nll_loss = torch.mean(nll(torch.randn_like(sample)))
        loss_dic = {
            f"Loss": loss.item(),
            f"reference_nll_loss": reference_nll_loss.item(),
            f"nlogdet_loss": nlogdet_loss.item(),
            f"nll_loss": nll_loss.item(),
        }
        logger.append(loss_dic)
        ## Add description to keys to be identified either as train or eval
        loss_dic = {mode + '_' + key:val for key, val in loss_dic.items()}
        wandb.log(loss_dic)
        return loss

def nll(sample):
    return 0.5 * torch.sum(torch.pow(sample, 2), dim=[1, 2, 3])