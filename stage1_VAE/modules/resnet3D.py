import torch.nn as nn, torch
import torch.nn.functional as F
import math
from torch.nn.utils import spectral_norm

######################################################################################################
########## 3D-ConvNet Implementation adapted from https://github.com/tomrunia/PyTorchConv3D ##########

def resnet10():
    """Constructs a ResNet-18 model.
    """
    return BasicBlock, [1, 1, 1, 1]

def resnet18():
    """Constructs a ResNet-18 model.
    """
    return BasicBlock, [2, 2, 2, 2]

def resnet34():
    """Constructs a ResNet-34 model.
    """
    return BasicBlock, [3, 4, 6, 3]

def resnet50():
    """Constructs a ResNet-50 model.
    """
    return Bottleneck, [3, 4, 6, 3]

def resnet101():
    """Constructs a ResNet-101 model.
    """
    return Bottleneck, [3, 4, 23, 3]


def conv3x3x3(in_planes, out_planes, stride=1, stride_t=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(3, 3, 3),
        stride=[stride_t, stride, stride],
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = torch.cat([out.data, zero_pads], dim=1)

    return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, stride_t=1, downsample=None, spectral=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(num_groups=16, num_channels=planes)
        self.conv2 = conv3x3x3(planes, planes, stride, stride_t)
        self.bn2 = nn.GroupNorm(num_groups=16, num_channels=planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(num_groups=16, num_channels=planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if spectral:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
            self.conv3 = spectral_norm(self.conv3)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, stride_t=1, downsample=None, spectral=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride, stride_t)
        self.bn1 = nn.GroupNorm(num_groups=16, num_channels=planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.GroupNorm(num_groups=16, num_channels=planes)
        self.downsample = downsample
        self.stride = stride

        if spectral:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Encoder(nn.Module):

    def __init__(self, dic):
        self.inplanes = 64
        super(Encoder, self).__init__()

        __possible_resnets = {
            'resnet18':  resnet18,
            'resnet34':  resnet34,
            'resnet50':  resnet50,
            'resnet101': resnet101
        }

        type_ = dic["res_type_encoder"]
        self.use_spectral_norm = False
        self.use_max_pool = dic["use_max_pool"]
        z_dim = dic["z_dim"]
        self.use_max_pool = dic["use_max_pool"]
        channels = dic["channels"]
        stride_s = dic["stride_s"]
        stride_t = dic["stride_t"]
        assert len(channels) - 1 == len(stride_t)
        assert len(channels) - 1 == len(stride_s)
        block, layers = __possible_resnets[type_]()

        self.conv1   = nn.Conv3d(3, channels[0], kernel_size=(3, 7, 7), stride=(2, 2, 2), padding=(1, 3, 3), bias=False)
        self.norm1   = nn.GroupNorm(num_groups=16, num_channels=channels[0])
        self.relu    = nn.ReLU(inplace=True)
        if self.use_max_pool:
            self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)

        self.layer = []
        for i, ch in enumerate(channels[1:]):
            self.layer.append(self._make_layer(block, ch, layers[i], stride=stride_s[i], stride_t=stride_t[i]))
        self.layer = nn.Sequential(*self.layer)

        self.conv_mu = nn.Conv2d(channels[-1], z_dim, 4, 1, 0)
        self.conv_var = nn.Conv2d(channels[-1], z_dim, 4, 1, 0)


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def _make_layer(self, block, planes, blocks, stride=1, stride_t=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=[3, 3, 3], ### Difference to cinemagrpah model!!!!!!!!
                    stride=[stride_t, stride, stride],
                    padding=[1, 1, 1],
                    bias=False),
                nn.GroupNorm(num_channels=planes * block.expansion, num_groups=16))

        layers = [block(self.inplanes, planes, stride, stride_t, downsample, self.use_spectral_norm)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def reparameterize(self, emb):
        mu, logvar = self.conv_mu(emb).reshape(emb.size(0), -1), self.conv_var(emb).reshape(emb.size(0), -1)
        eps = torch.FloatTensor(logvar.size()).normal_().cuda()
        std = logvar.mul(0.5).exp_()
        return eps.mul(std).add_(mu), mu, logvar

    def forward(self, x):
        if x.size(1) > x.size(2):
            x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        if self.use_max_pool:
            x = self.maxpool(x)

        x = self.layer(x)

        return self.reparameterize(x.squeeze(2))


class Discriminator(nn.Module):

    def __init__(self, dic):
        self.inplanes = 64
        super().__init__()

        __possible_resnets = {
            'resnet18':  resnet18,
            'resnet34':  resnet34,
            'resnet50':  resnet50,
            'resnet101': resnet101
        }

        type_ = dic["res_type_encoder"]
        self.use_spectral_norm = dic["spectral_norm"]
        self.use_max_pool = dic["use_max_pool"]
        channels = dic["channels"]
        stride_s = dic["stride_s"]
        stride_t = dic["stride_t"]
        assert len(channels) - 1 == len(stride_t)
        assert len(channels) - 1 == len(stride_s)
        block, layers = __possible_resnets[type_]()

        self.conv1   = nn.Conv3d(3, channels[0], kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.norm1   = nn.GroupNorm(num_groups=16, num_channels=channels[0])
        self.relu    = nn.ReLU(inplace=True)
        if self.use_max_pool:
            self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)

        self.layer = []
        for i, ch in enumerate(channels[1:]):
            self.layer.append(self._make_layer(block, ch, layers[i], stride=stride_s[i], stride_t=stride_t[i]))
        self.layer = nn.Sequential(*self.layer)

        self.avgpool = nn.AvgPool3d((1, 4, 4), stride=1)
        self.fc = nn.Linear(channels[-1] * block.expansion, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.orthogonal_(m.weight)

    def _make_layer(self, block, planes, blocks, stride=1, stride_t=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or stride_t != 1:
            downsample = nn.Sequential(
                spectral_norm(nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=[3, 3, 3],
                    stride=[stride_t, stride, stride],
                    padding=[1, 1, 1],
                    bias=False)),
                nn.GroupNorm(num_channels=planes * block.expansion, num_groups=16))

        layers = [block(self.inplanes, planes, stride, stride_t, downsample, self.use_spectral_norm)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if x.size(1) > x.size(2):
            x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        if self.use_max_pool:
            x = self.maxpool(x)

        out = []
        for lay in self.layer:
            x = lay(x)
            out.append(x)

        x = self.avgpool(x)
        x = self.fc(x.reshape(x.size(0), -1))

        return x, out

