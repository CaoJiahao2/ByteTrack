#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from loguru import logger

import torch

import os
import shutil


def load_ckpt(model, ckpt):
    # 获取目标模型的当前状态字典，包含了模型的所有权重参数
    """
    遍历目标模型的状态字典：
    如果当前权重参数在checkpoint中不存在，发出警告并跳过。
    如果当前权重参数的形状与checkpoint中对应的权重参数的形状不一致，发出警告并跳过。
    否则，将checkpoint中对应的权重参数加入load_dict中。
    """
    model_state_dict = model.state_dict()
    load_dict = {}
    for key_model, v in model_state_dict.items():
        if key_model not in ckpt:
            logger.warning(
                "{} is not in the ckpt. Please double check and see if this is desired.".format(
                    key_model
                )
            )
            continue
        v_ckpt = ckpt[key_model]
        if v.shape != v_ckpt.shape:
            logger.warning(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                    key_model, v_ckpt.shape, key_model, v.shape
                )
            )
            continue
        load_dict[key_model] = v_ckpt

    model.load_state_dict(load_dict, strict=False)
    return model


def save_checkpoint(state, is_best, save_dir, model_name=""):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, model_name + "_ckpt.pth.tar")
    torch.save(state, filename)
    if is_best:
    #如果当前模型在验证集上是最佳的 (is_best 为 True)，
    # 则构造最佳模型文件的路径，将当前模型文件拷贝到最佳模型文件中。
        best_filename = os.path.join(save_dir, "best_ckpt.pth.tar")
        shutil.copyfile(filename, best_filename)
