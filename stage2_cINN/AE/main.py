import numpy as np, os, sys, ast, time
import torch, torch.nn as nn, pickle as pkl, random
from tqdm import tqdm, trange
import argparse, torchvision
from datetime import datetime
import wandb
from omegaconf import OmegaConf

import stage2_cINN.AE.modules.AE as BigAE
from stage2_cINN.AE.modules.loss import Loss
from stage1_VAE.modules.patch_disc import NLayerDiscriminator
from data.get_dataloder import get_loader
import utils.auxiliaries as aux


"""=========================Trainer Function==================================================="""

def trainer(network, disc, epoch, data_loader, optimizers, loss_func, scheduler, logger, save_path):

    _ = network.train()
    logger.reset()
    data_iter = tqdm(data_loader, position=2)
    inp_string = 'Epoch {} || Loss_recon: --- '.format(epoch)
    data_iter.set_description(inp_string)
    for image_idx, file_dict in enumerate(data_iter):

        img = file_dict["seq"].squeeze(1).type(torch.FloatTensor).cuda()
        img_gen, loss_recon = loss_func(img, network, disc, optimizers, epoch, logger)

        if image_idx % 20 == 0:
            inp_string = 'Epoch {} || Loss_recon: {}'.format(epoch, np.round(loss_recon, 3))
            data_iter.set_description(inp_string)

    ## Empty GPU cache and perform scheduler update
    torch.cuda.empty_cache()
    scheduler[0].step(loss_recon)
    scheduler[1].step(loss_recon)

    ## Log images using wandb
    images = torchvision.utils.make_grid(torch.cat((img.cpu(), img_gen.detach().cpu()), dim=2))
    wandb.log({"images_train": [wandb.Image(images.permute(1, 2, 0).numpy(), caption="Reconstructions")]})
    torchvision.utils.save_image(torch.cat((img.cpu(), img_gen.detach().cpu()), dim=2),
                                 save_path + f'{epoch}_train_recon.jpg', normalize=True)

"""===========================Validation Function==================================================="""

def validator(network, disc, epoch, data_loader, loss_func, logger, save_path):

    _ = network.eval()
    logger.reset()
    data_iter = tqdm(data_loader, position=2)
    inp_string = 'Epoch {} || Loss_recon: ----'.format(epoch)
    data_iter.set_description(inp_string)
    with torch.no_grad():
        for image_idx, file_dict in enumerate(data_iter):

            img = file_dict["seq"].squeeze(1).type(torch.FloatTensor).cuda()

            img_gen, loss_recon = loss_func(img, network, disc, None, epoch, logger, training=False)

            if image_idx % 20 == 0:
                inp_string = 'Epoch {} || Loss_recon: {}'.format(epoch, np.round(loss_recon, 3))
                data_iter.set_description(inp_string)

    ## Save images
    images = torchvision.utils.make_grid(torch.cat((img.cpu(), img_gen.detach().cpu()), dim=2))
    wandb.log({"images_eval": [wandb.Image(images.permute(1, 2, 0).numpy(), caption="Reconstructions")]})
    torchvision.utils.save_image(torch.cat((img.cpu(), img_gen.detach().cpu()), dim=2),
                                 save_path + f'{epoch}_eval_recon.jpg', normalize=True)

    ### Empty GPU cache
    torch.cuda.empty_cache()

def main(opt):
    """================= Create Model, Optimizer and Scheduler =========================="""
    network = BigAE.BigAE(opt.AE).cuda()
    disc = NLayerDiscriminator(opt.Discriminator_Patch).cuda()

    loss_func    = Loss(opt.Training)
    optimizer1   = torch.optim.Adam(network.parameters(), lr=opt.Training['lr'], weight_decay=opt.Training['weight_decay'])
    scheduler1   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='min', factor=0.5, patience=1, min_lr=1e-8,
                                                             threshold=0.0001, threshold_mode='abs')
    optimizer2   = torch.optim.Adam(disc.parameters(), lr=opt.Training['lr'], weight_decay=opt.Training['weight_decay'])
    scheduler2   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='min', factor=0.5, patience=1, min_lr=1e-8,
                                                             threshold=0.0001, threshold_mode='abs')

    optimizers = [optimizer1, optimizer2]
    schedulers = [scheduler1, scheduler2]

    """==================== Set up Dataloaders ========================"""
    dataset       = get_loader(opt.Data['dataset'])
    train_dataset = dataset.Dataset(opt, mode='train')
    eval_dataset  = dataset.Dataset(opt, mode='eval')

    train_data_loader = torch.utils.data.DataLoader(train_dataset, num_workers=opt.Training['workers'],
                                                    batch_size=opt.Training['bs'], shuffle=True, drop_last=True)
    eval_data_loader  = torch.utils.data.DataLoader(eval_dataset, num_workers=opt.Training['workers'],
                                                    batch_size=opt.Training['bs'], shuffle=True)

    """======================Set Logging Files======================"""
    dt = datetime.now()
    dt = '{}-{}-{}-{}-{}-{}'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    run_name = 'Stage2_AE_' + opt.Data['dataset'] + '_Date-' + dt + '_' + opt.Training['savename']
    save_path = opt.Training['save_path'] + "/" + run_name

    ## Make the saving directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    opt.Training['save_path'] = save_path

    # Make summary plots, images, segmentation and videos folder
    save_path_img = save_path + '/images/'
    if not os.path.exists(save_path_img):
        os.makedirs(save_path_img)

    # Create weightsbiases logger
    log_dic = opt.Logging
    wandb.init(entity=log_dic['entitiy'], config=opt, dir=save_path, project=log_dic['project'],
               name=opt.Training['savename'], mode=log_dic['mode'])
    # wandb.watch(model, log="all")

    # save yaml config
    OmegaConf.save(config=opt, f=save_path + "/config_stage2_AE.yaml")

    ## Offline logging
    logging_keys = ["Loss", "Loss_recon", 'Loss_nll', "Logvar", "L_KL", "Loss_G", "L_disc", 'Logits_real',
                    'Logits_fake', 'Disc_weight', 'Disc_factor']

    logger_train = aux.Logging(logging_keys)
    logger_eval = aux.Logging(logging_keys)

    ### Setting up CSV writers
    full_log_train = aux.CSVlogger(save_path + "/log_per_epoch_train.csv", ["Epoch", "Time", "LR"] + logging_keys)
    full_log_eval = aux.CSVlogger(save_path + "/log_per_epoch_test.csv", ["Epoch", "Time", "LR"] + logging_keys)

    """=================== Start training ! ==========================="""
    best_val = 99
    epoch_iterator = tqdm(range(0, opt.Training['n_epochs']), ascii=True, position=1)
    for epoch in epoch_iterator:
        epoch_time = time.time()
        lr = [group['lr'] for group in optimizers[0].param_groups][0]

        ##### Training ########
        epoch_iterator.set_description("Training with lr={}".format(np.round(lr, 6)))
        trainer(network, disc, epoch, train_data_loader, optimizers, loss_func, schedulers, logger_train, save_path_img)

        ###### Validation #########
        epoch_iterator.set_description('Validating...')
        validator(network, disc, epoch, eval_data_loader, loss_func, logger_eval, save_path_img)

        ## Best Validation Score
        if logger_eval.log()[1] < best_val:
            ###### SAVE CHECKPOINTS ########
            save_dict = {'state_dict': network.encoder.state_dict()}
            torch.save(save_dict, save_path + f'/encoder_stage2.pth')

        ###### Logging Epoch Data
        epoch_time = time.time() - epoch_time
        full_log_train.write([epoch, epoch_time, lr, *logger_train.log()])
        full_log_eval.write([epoch, epoch_time, lr, *logger_eval.log()])

### Start Training ###
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--config", type=str, default='stage2_cINN/AE/configs/bair_config.yaml',
                        help="Define config file")
    parser.add_argument("-gpu", type=str, required=True)
    args = parser.parse_args()

    conf = OmegaConf.load(args.config)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    aux.set_seed(42)

    main(conf)

