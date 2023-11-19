#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
# profile是一个用于统计模型参数数量和浮点运算数（FLOPs）的工具
from thop import profile

from copy import deepcopy

__all__ = [
    "fuse_conv_and_bn",
    "fuse_model",
    "get_model_info",
    "replace_module",
]


def get_model_info(model, tsize):
    # 获取模型的参数数量和浮点运算数（FLOPs）信息
    stride = 64
    # 创建了一个大小为 (1, 3, 64, 64) 的零张量 img，表示模型的输入图像
    img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)

    flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    #  将参数数量转换为以百万（M）为单位。
    #  将 FLOPs 转换为以十亿（G）为单位。
    params /= 1e6
    flops /= 1e9
    flops *= tsize[0] * tsize[1] / stride / stride * 2  # Gflops,乘2因为通常每个FLOP包括一个乘法和一个加法操作
    info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
    return info


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False) # 不需要计算梯度
        .to(conv.weight.device)
    )

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    # 计算批归一化层权重的对角矩阵，用于调整权重
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    # 将卷积层的权重与批归一化层的权重相乘，得到融合后的权重
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = (
        torch.zeros(conv.weight.size(0), device=conv.weight.device)
        if conv.bias is None
        else conv.bias
    )
    # 计算批归一化层的偏置，用于调整偏置
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps)
    )
    # 将卷积层的偏置与批归一化层的偏置相加，得到融合后的偏置
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def fuse_model(model):
    """
    Fuse model to accelerate inference speed.
    Args:
        model (nn.Module): model to fuse.
    """
    from yolox.models.network_blocks import BaseConv

    for m in model.modules():
        if type(m) is BaseConv and hasattr(m, "bn"):
            m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
            delattr(m, "bn")  # remove batchnorm
            m.forward = m.fuseforward  # update forward
    return model


def replace_module(module, replaced_module_type, new_module_type, replace_func=None):
    """
    Replace given type in module to a new type. mostly used in deploy.

    Args:
        module (nn.Module): model to apply replace operation.
        replaced_module_type (Type): module type to be replaced.
        new_module_type (Type)
        replace_func (function): python function to describe replace logic. Defalut value None.

    Returns:
        model (nn.Module): module that already been replaced.
    """

    def default_replace_func(replaced_module_type, new_module_type):
        return new_module_type()

    if replace_func is None:
        replace_func = default_replace_func

    model = module
    if isinstance(module, replaced_module_type):
        model = replace_func(replaced_module_type, new_module_type)
    else:  # recurrsively replace
    # 如果输入的模型不是指定类型的模块，递归地遍历模型的子模块，对每个子模块应用相同的替换逻辑
        for name, child in module.named_children():
            new_child = replace_module(child, replaced_module_type, new_module_type)
            if new_child is not child:  # child is already replaced
                model.add_module(name, new_child)

    return model
