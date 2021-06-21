# Stochastic Image-to-Video Synthesis using cINNs

Official PyTorch implementation of Stochastic Image-to-Video Synthesis using cINNs accepted to CVPR2021.

https://user-images.githubusercontent.com/24416143/117287893-2aef6d80-ae6b-11eb-8d6a-f0678426ff58.mp4

## [Arxiv](https://arxiv.org/abs/2105.04551) | [Project Page](https://compvis.github.io/image2video-synthesis-using-cINNs/) | [Supplemental](https://compvis.github.io/image2video-synthesis-using-cINNs/results/222_supp.zip) | [Pretrained Models](https://drive.google.com/drive/folders/12-PccC8jKz4UGpaE9GS0aOi23PHjJLzW?usp=sharing) | [BibTeX](#bibtex)
[Michael Dorkenwald](https://mdork.github.io/),
[Timo Milbich](https://timomilbich.github.io/),
[Andreas Blattmann](https://www.linkedin.com/in/andreas-blattmann-479038186/?originalSubdomain=de),
[Robin Rombach](https://github.com/pesser),
[Kosta Derpanis](https://www.cs.ryerson.ca/kosta/)\*,
[Björn Ommer](https://hci.iwr.uni-heidelberg.de/Staff/bommer)\*,
[CVPR 2021](http://cvpr2021.thecvf.com/)<br/>

**tl;dr** We present a framework for both stochastic and controlled image-to-video synthesis. We bridge the gap between the image and video domain using conditional invertible neural networks and account for the inherent ambiguity with a learned, dedicated scene dynamics representation.

![teaser](https://compvis.github.io/image2video-synthesis-using-cINNs/paper/method.png)

For any questions, issues, or recommendations, please contact Michael at m.dorkenwald(at)gmail.com. If our project is helpful for your research, please consider [citing](#bibtex).
# Table of Content

1. [Requirements](#Requirements)
2. [Running pretrained models](#pretrained_models)
3. [Data preparation](#data)
4. [Evaluation](#evaluation)
    1. [Synthesis quality](#synthesis_quality)
    2. [Diversity](#diversity)
5. [Training](#training)
    1. [Stage1: Video-to-Video synthesis](#stage1)
    2. [Stage2: cINN for Image-to-Video synthesis](#stage2)
6. [Shout-outs](#shoutouts)
7. [BibTeX](#bibtex)


## Requirements <a name="Requirements"></a>
A suitable [conda](https://conda.io/) environment named `i2v` can be created and activated with

```
conda env create -f environment.yaml
conda activate i2v
```
For this repository cuda verion 11.1 is used. To suppress the annoying warnings from kornia please run all python scripts with `-W ignore`.
## Running pretrained models <a name="pretrained_models"></a>
One can test our method using the scripts below on images placed in `assets/GT_samples` after placing the [pre-trained model weights](https://drive.google.com/drive/folders/12-PccC8jKz4UGpaE9GS0aOi23PHjJLzW?usp=sharing) for the corresponding datasets e.g. bair in the models folder like `models/bair/`.  
```bash
python -W ignore generate_samples.py -dataset landscape -gpu <gpu_id> -seq_length <sequence_length>
```
![teaser](https://compvis.github.io/image2video-synthesis-using-cINNs/results/Landscape_more.gif)

Moreoever, one can also transfer an observed dynamic from a given video (first row) to an arbitrary starting frame using
```bash
python -W ignore generate_transfer.py -dataset landscape -gpu <gpu_id> 
```
![teaser](https://compvis.github.io/image2video-synthesis-using-cINNs/results/transfer.gif)
![teaser](https://compvis.github.io/image2video-synthesis-using-cINNs/results/transfer2.gif)

```bash
python -W ignore generate_samples.py -dataset bair -gpu <gpu_id> 
```
![teaser](https://compvis.github.io/image2video-synthesis-using-cINNs/results/BAIR_more.gif)

Our model can be extended to control specific factors e.g. the endpoint location of the robot arm. Note, to run this script you need to download the BAIR dataset.
```bash
python -W ignore visualize_endpoint.py -dataset bair -gpu <gpu_id> -data_path <path2data>
```

Sample 1             |  Sample 2
:-------------------------:|:-------------------------:
![](https://compvis.github.io/image2video-synthesis-using-cINNs/results/endpoint_15.gif) |  ![](https://compvis.github.io/image2video-synthesis-using-cINNs/results/endpoint_17.gif)


or look only on the last frame of the generated sequence, which is similar since all videos were conditioned on the same endpoint

Sample 1             |  Sample 2
:-------------------------:|:-------------------------:
![](https://compvis.github.io/image2video-synthesis-using-cINNs/results/endpoint_15.png) |  ![](https://compvis.github.io/image2video-synthesis-using-cINNs/results/endpoint_17.png)

```bash
python -W ignore generate_samples.py -dataset iPER -gpu <GPU_ID>
```
![teaser](https://compvis.github.io/image2video-synthesis-using-cINNs/results/iPER_more.gif)
```bash
python -W ignore generate_samples.py -dataset DTDB -gpu <GPU_ID> -texture fire
```
![teaser](https://compvis.github.io/image2video-synthesis-using-cINNs/results/fire_more.gif)
```bash
python -W ignore generate_samples.py -dataset DTDB -gpu <GPU_ID> -texture vegetation
```
![teaser](https://compvis.github.io/image2video-synthesis-using-cINNs/results/vegetation_more.gif)
```bash
python -W ignore generate_samples.py -dataset DTDB -gpu <GPU_ID> -texture clouds
```
![teaser](https://compvis.github.io/image2video-synthesis-using-cINNs/results/clouds_more.gif)
```bash
python -W ignore generate_samples.py -dataset DTDB -gpu <GPU_ID> -texture waterfall
```
![teaser](https://compvis.github.io/image2video-synthesis-using-cINNs/results/waterfall_more.gif)

## Data preparation <a name="data"></a>
### BAIR

To download the dataset to a given target directory `<TARGETDIR>`, run the following command
```bash
sh data/bair/download_bair.sh <TARGETDIR>
```
In order to convert the tensorflow records file run the following command
```bash
python data/bair/convert_bair.py --data_dir <DATADIR> --output_dir <TARGETDIR>
```
`traj_256_to_511` is used for validation and `traj_0_to_255` for testing. 
The resulting folder structure should be the following
```
$bair/train/
├── traj_512_to_767
│   ├── 1
|   ├── ├── 0.png
|   ├── ├── 1.png
|   ├── ├── 2.png
|   ├── ├── ...
│   ├── 2
│   ├── ...
├── ...
$bair/eval/
├── traj_256_to_511
│   ├── 1
|   ├── ├── 0.png
|   ├── ├── 1.png
|   ├── ├── 2.png
|   ├── ├── ...
│   ├── 2
│   ├── ...
$bair/test/
├── traj_0_to_255
│   ├── 1
|   ├── ├── 0.png
|   ├── ├── 1.png
|   ├── ├── 2.png
|   ├── ├── ...
│   ├── 2
│   ├── ...
```
Please [cite](https://dblp.org/rec/conf/corl/EbertFLL17.html?view=bibtex) the corresponding [paper](https://arxiv.org/abs/1710.05268) if you use the data.

### Landscape
Download the corresponding dataset from [here](https://drive.google.com/file/d/1xWLiU-MBGN7MrsFHQm4_yXmfHBsMbJQo/view) using e.g. [gdown](https://github.com/wkentaro/gdown). To use our provided data loader all images need to be renamed to frame0 to frameX to alleviate the problem of missing frames. Therefore the following script can be used
```bash
python data/landscape/rename_images.py --data_dir <DATADIR> 
```
In `data/landscape` we provide a list of videos that were used for training and testing. Please [cite](https://dblp.org/rec/conf/cvpr/XiongL00L18.html?view=bibtex) the corresponding [paper](https://arxiv.org/abs/1709.07592) if you use the data.

### iPER
Download the dataset from [here](https://svip-lab.github.io/dataset/iPER_dataset.html)  and run
```bash
python data/iPER/extract_iPER.py --raw_dir <DATADIR> --processed_dir <TARGETDIR>
```
to extract the frames. In `data/iPER` we provide a list of videos that were used for train, eval, and test. Please [cite](https://dblp.org/rec/conf/iccv/LiuPML0G19.html?view=bibtex) the corresponding [paper](https://arxiv.org/abs/1909.12224) if you use the data. 

### Dynamic Textures
Download the corrsponding dataset from [here](https://drive.google.com/file/d/1sbzZdaWpNCMcCMEcUYFjygLo-f8sdsY6/view?usp=sharing) and unzip it. Please [cite](https://dblp.org/rec/conf/eccv/HadjiW18.html?view=bibtex) the corresponding [paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Isma_Hadji_A_New_Large_ECCV_2018_paper.pdf) if you use the data. The original mp4 files from DTDB can be downloaded from [here](http://www.cse.yorku.ca/~hadjisma/dtdb_website/dtdb.html).
 
## Evaluation <a name="evaluation"></a>
After storing the data as described, the evaluation script for each dataset can be used. 

### Synthesis quality <a name="synthesis_quality"></a>
We use the following metrics to measure synthesis quality: [LPIPS](https://github.com/richzhang/PerceptualSimilarity), [FID](https://github.com/mseitzer/pytorch-fid), [FVD](https://github.com/google-research/google-research/tree/master/frechet_video_distance), DTFVD. The latter was introduced in this work and is a specific FVD for dynamic textures. Therefore, please download the weights of the I3D model from [here](https://drive.google.com/drive/folders/12-PccC8jKz4UGpaE9GS0aOi23PHjJLzW?usp=sharing) and place it in the models folder like `/models/DTI3D/`. For more details on DTFVD please see Sec. C3 in supplemental. To compute the mentioned metrics for a given dataset please run
```bash
python -W ignore eval_synthesis_quality.py -gpu <gpu_id> -dataset <dataset> -data_path <path2data> -FVD True -LPIPS True -FID True -DTFVD True
```
for DTDB please specify the dynamic texture you want to evalute e.g. fire
```bash
python -W ignore eval_synthesis_quality.py -gpu <gpu_id> -dataset DTDB -data_path <path2data> -texture fire -FVD True -LPIPS True -FID True -DTFVD True
```
Please [cite](#bibtex) our work if you use DTFVD in your work. If you place the chkpts outside this repository please specify the location using the argument `-chkpt <path_to_chkpt>`.   
### Diversity <a name="diversity"></a>
We measure diversity by comparing different realizations of an example using a pretrained VGG, I3D and DTI3D backbone. The last two consider the temporal property of the data whereas for the VGG diversity score compared images framewise. To evaluate diversity for a given dataset please run
```bash
python -W ignore eval_diversity.py -gpu <gpu_id> -dataset <dataset> -data_path <path2data> -DTI3D True -VGG True -I3D True -seq_length <length>
```
for DTDB please specify the dynamic texture you want to evalute e.g. fire
```bash
python -W ignore eval_diversity.py -gpu <gpu_id> -dataset DTDB -data_path <path2data> -texture fire -DTI3D True -VGG True -I3D True 
```
## Training
The training of our models is divided into two consecutive stages. In stage 1, we learn an information preserving video latent representation using a conditional generative model which reconstructs the given input video as best as possible. After that, we learn a conditional INN to map the video latent representation to a residual space depicting the scene dynamics conditioned on the starting frame and additional control factors. During inference, we now can sample new scene dynamics from the residual distribution and synthesize novel videos due to the bijective nature of the cINN. For more details please check out our paper. 

For logging our runs we used and recommend [wandb](https://wandb.ai/). Please create a free account and add your username to the config. If you don't want to use it, the metrics are also logged in a csv file and samples are written out in the specified chkpt folder. Therefore, please set logging mode to `offline`. For logging (PyTorch) FVD please download the weights of a PyTorch I3D from [here](https://github.com/hassony2/kinetics_i3d_pytorch/blob/master/model/model_rgb.pth) and place it in models like `/models/PI3D/`. For logging DTFVD please download the weights of the DTI3D model from [here](https://drive.google.com/drive/folders/12-PccC8jKz4UGpaE9GS0aOi23PHjJLzW?usp=sharing) and place it in the models folder like `/models/DTI3D/`. Depending on the dataset please specify either FVD or DTFVD under `FVD` in the config. For each provided pretrained model we left the corresponding config file in the corresponding folder. If you want to run our model on a dataset we did not provide please create a new config. Before you start a run please specify the data path, save path, and the name of the run in the config.

### Stage 1: Video-to-Video synthesis <a name="stage1"></a>
To train the conditional generative model for video-to-video synthesis run the following command
```bash
python -W ignore -m stage1_VAE.main -gpu <gpu_id> -cf stage1_VAE/configs/<config>
```
### Stage 2: cINN for Image-to-Video synthesis <a name="stage2"></a>
Before we can train the cINN, we need to train an AE to obtain an encoder to embed the starting frame for the cINN. You can use the on provided or train your own by running 
```bash
python -W ignore -m stage2_cINN.AE.main -gpu <gpu_id> -cf stage2_cINN/AE/configs/<config>
```
To train the cINN, we need to specify the location of the trained encoder as well as the first stage model in the config. After that, training of the cINN can be started by
```bash
python -W ignore -m stage2_cINN.main -gpu <gpu_id> -cf stage2_cINN/configs/<config>
```
To reproduce the controlled video synthesis experiment, one can specify the `control True` in the `bair_config.yaml` to additional condition the cINN on the endpoint location.

## Shout-outs <a name="shoutouts"></a>
Thanks to everyone who makes their code and models available. In particular,

- The decoder architecture is inspired by [SPADE](https://github.com/NVlabs/SPADE)
- The great work and code of Stochastic Latent Residual Video Prediction [SRVP](https://github.com/edouardelasalles/srvp)
- The 3D encoder and discriminator are based on [3D-Resnet](https://github.com/tomrunia/PyTorchConv3D) and spatial discriminator is adapted from [PatchGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- The metrics which were used [LPIPS](https://github.com/richzhang/PerceptualSimilarity) [PyTorch FID](https://github.com/mseitzer/pytorch-fid) [FVD](https://github.com/google-research/google-research/tree/master/frechet_video_distance)

## BibTeX

```
@InProceedings{Dorkenwald_2021_CVPR,
    author    = {Dorkenwald, Michael and Milbich, Timo and Blattmann, Andreas and Rombach, Robin and Derpanis, Konstantinos G. and Ommer, Bjorn},
    title     = {Stochastic Image-to-Video Synthesis Using cINNs},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {3742-3753}
}
```
