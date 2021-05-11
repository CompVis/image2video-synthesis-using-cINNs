#!/usr/bin/env python3
# Code adapted from https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
"""Calculates the Frechet Inception Distance (FID) to evalulate GANs
The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.
When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).
The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.
See --help to see further details.
Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow
Copyright 2018 Institute of Bioinformatics, JKU Linz
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import numpy as np
import torch
from scipy import linalg
from metrics.PyTorch_FVD.I3D import I3D

import torch.nn.functional as F

def calculate_frechet_distance_dic(act1, act2, eps=1e-6):
    act1 = np.stack(act1, 0)
    act2 = np.stack(act2, 0)
    mu1, sigma1 = np.mean(act1, 0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, 0), np.cov(act2, rowvar=False)

    return calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6)/3.3

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def get_activations(data, model, batch_size=50, cuda=False, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- data        : Tensor of images
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()
    n_samples = data.size(0)

    if n_samples % batch_size != 0:
        pass
        # print(('Warning: number of images is not a multiple of the '
        #        'batch size. Some samples are going to be ignored.'))
    if batch_size > n_samples:
        # print(('Warning: batch size is bigger than the data size. '
        #        'Setting batch size to data size'))
        batch_size = n_samples

    n_batches = n_samples // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, 400))
    for i in range(n_batches):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)

        start = i * batch_size
        end = start + batch_size

        batch = data[start:end]
        if cuda:
            batch = batch.cuda()
        with torch.no_grad():
            pred = model(batch.permute(0, 2, 1, 3, 4))[1]
        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

    if verbose:
        print(' done')
    return pred_arr


def calculate_activation_statistics(data, model, batch_size=50, cuda=True, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(data, model, batch_size, cuda, verbose)

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_FVD(model, data_gen, data_orig, batch_size, cuda=True):
    """Calculates the FID for two tensors"""
    data_gen, data_orig = preprocess(data_gen, data_orig)
    m1, s1 = calculate_activation_statistics(data_gen, model, batch_size, cuda)
    m2, s2 = calculate_activation_statistics(data_orig, model, batch_size, cuda)
    FVD = calculate_frechet_distance(m1, s1, m2, s2)

    return FVD

def compute_activations(model, data_gen, data_orig, batch_size, cuda=True):
    data_gen, data_orig = preprocess(data_gen, data_orig)
    return get_activations(data_orig, model, batch_size, cuda), get_activations(data_gen, model, batch_size, cuda)

def preprocess(data_gen, data_orig):

    data_gen = F.interpolate(data_gen.reshape(-1, *data_gen.shape[2:]), mode='bilinear', size=(224, 224),
                             align_corners=True).reshape(*data_gen.shape[:2], 3, 224, 224)
    data_orig = F.interpolate(data_orig.reshape(-1, *data_orig.shape[2:]), mode='bilinear', size=(224, 224),
                              align_corners=True).reshape(*data_orig.shape[:2], 3, 224, 224)

    if data_gen.min() < 0:
        data_gen = denorm(data_gen)

    if data_orig.min() < 0:
        data_orig = denorm(data_orig)

    return data_gen, data_orig

def denorm(x):
    return (x + 1.0)/2.0

def load_model():
    model = I3D(400, 'rgb')
    state_dict = torch.load('./models/PI3D/model_rgb.pth', map_location="cpu")
    model.load_state_dict(state_dict)
    _ = model.eval()

    return model
