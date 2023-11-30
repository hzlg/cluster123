from __future__ import print_function
import argparse, os, sys, csv, shutil, time, random, operator, pickle, ast
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
import pickle
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import models.cifar as models

sys.path.insert(0, "/./../utils/")
from logger import *
from eval import *
from misc import *
import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def ResNet18(**kwargs):
    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet34(**kwargs):
    return _resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


def ResNet50(**kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def ResNet101(**kwargs):
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def ResNet152(**kwargs):
    return _resnet(Bottleneck, [3, 8, 36, 3], **kwargs)


class cifar_mlp(nn.Module):
    def __init__(self, ninputs=3 * 32 * 32, num_classes=10):
        self.ninputs = ninputs
        self.num_classes = num_classes
        super(cifar_mlp, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(self.ninputs, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(-1, self.ninputs)
        hidden_out = self.features(x)
        return self.classifier(hidden_out)


def get_model(config, parallel=False, cuda=True, device=0):
    # print("==> creating model '{}'".format(config['arch']))
    if config["arch"].startswith("resnext"):
        model = models.__dict__[config["arch"]](
            cardinality=config["cardinality"],
            num_classes=config["num_classes"],
            depth=config["depth"],
            widen_factor=config["widen-factor"],
            dropRate=config["drop"],
        )
    elif config["arch"].startswith("densenet"):
        model = models.__dict__[config["arch"]](
            num_classes=config["num_classes"],
            depth=config["depth"],
            growthRate=config["growthRate"],
            compressionRate=config["compressionRate"],
            dropRate=config["drop"],
        )
    elif config["arch"].startswith("wrn"):
        model = models.__dict__[config["arch"]](
            num_classes=config["num_classes"],
            depth=config["depth"],
            widen_factor=config["widen-factor"],
            dropRate=config["drop"],
        )
    elif config["arch"].endswith("resnet"):
        model = models.__dict__[config["arch"]](
            num_classes=config["num_classes"],
            depth=config["depth"],
        )
    elif config["arch"].endswith("convnet"):
        model = models.__dict__[config["arch"]](num_classes=config["num_classes"])
    else:
        model = models.__dict__[config["arch"]](
            num_classes=config["num_classes"],
        )

    if parallel:
        model = torch.nn.DataParallel(model)

    if cuda:
        model.cuda()

    return model


def return_model(model_name, lr, momentum, parallel=False, cuda=True, device=0):
    if model_name == "dc":
        arch_config = {
            "arch": "Dc",
            "num_classes": 10,
        }
        model = get_model(arch_config, parallel=parallel, cuda=cuda, device=device)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=momentum)

    elif model_name == "alexnet":
        arch_config = {
            "arch": "alexnet",
            "num_classes": 10,
        }
        model = get_model(arch_config, parallel=parallel, cuda=cuda, device=device)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=momentum)
    elif model_name == "densenet-bc-100-12":
        arch_config = {
            "arch": "densenet",
            "depth": 100,
            "growthRate": 12,
            "compressionRate": 2,
            "drop": 0,
            "num_classes": 10,
        }
        model = get_model(arch_config, parallel=parallel, cuda=cuda, device=device)
        # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=momentum,weight_decay=1e-4)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    elif model_name == "densenet-bc-L190-k40":
        arch_config = {
            "arch": "densenet",
            "depth": 190,
            "growthRate": 40,
            "compressionRate": 2,
            "drop": 0,
            "num_classes": 10,
        }
        model = get_model(arch_config, parallel=parallel, cuda=cuda, device=device)
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=1e-4
        )
    elif model_name == "preresnet-110":
        arch_config = {
            "arch": "preresnet",
            "depth": 110,
            "num_classes": 10,
        }
        model = get_model(arch_config, parallel=parallel, cuda=cuda, device=device)
        # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=momentum, weight_decay=1e-4)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    elif model_name == "resnet-110":
        arch_config = {
            "arch": "resnet",
            "depth": 110,
            "num_classes": 10,
        }
        model = get_model(arch_config, parallel=parallel, cuda=cuda, device=device)
        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=1e-4)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    elif model_name == "resnext-16x64d":
        arch_config = {
            "arch": "resnext",
            "depth": 29,
            "cardinality": 16,
            "widen-factor": 4,
            "drop": 0,
            "num_classes": 10,
        }
        model = get_model(arch_config, parallel=parallel, cuda=cuda, device=device)
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4
        )
    elif model_name == "resnext-8x64d":
        arch_config = {
            "arch": "resnext",
            "depth": 29,
            "cardinality": 8,
            "widen-factor": 4,
            "drop": 0,
            "num_classes": 10,
        }
        model = get_model(arch_config, parallel=parallel, cuda=cuda, device=device)
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4
        )
    elif model_name.startswith("vgg"):
        arch_config = {
            "arch": model_name,
            "num_classes": 10,
        }
        model = get_model(arch_config, parallel=parallel, cuda=cuda, device=device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif model_name == "WRN-28-10-drop":
        arch_config = {
            "arch": "wrn",
            "depth": 28,
            "widen-factor": 10,
            "drop": 0.3,
            "num_classes": 10,
        }
        model = get_model(arch_config, parallel=parallel, cuda=cuda, device=device)
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4
        )
    else:
        assert False, "Model not found!"

    return model, optimizer


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class mnist_conv(nn.Module):
    def __init__(self):
        super(mnist_conv, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x, noise=torch.Tensor()):
        x = x.reshape(-1, 1, 28, 28)

        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 32 * 7 * 7)  # reshape Variable
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class mnist_conv_large(nn.Module):
    def __init__(self):
        super(mnist_conv, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 62)

    def forward(self, x, noise=torch.Tensor()):
        x = x.reshape(-1, 1, 28, 28)

        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 64 * 7 * 7)  # reshape Variable
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(in_channels + i * growth_rate),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        in_channels + i * growth_rate,
                        growth_rate,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                )
            )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
        return torch.cat(features, dim=1)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.bn(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv(x)
        x = self.pool(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_classes=10):
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        in_channels = 64
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                in_channels=in_channels, growth_rate=growth_rate, num_layers=num_layers
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            in_channels = in_channels + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = Transition(
                    in_channels=in_channels, out_channels=in_channels // 2
                )
                self.features.add_module("transition%d" % (i + 1), trans)
                in_channels = in_channels // 2
        self.features.add_module("norm5", nn.BatchNorm2d(in_channels))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
