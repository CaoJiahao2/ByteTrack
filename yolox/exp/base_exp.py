#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
from torch.nn import Module

from yolox.utils import LRScheduler

# 用于抽象语法树（Abstract Syntax Tree）的处理
# 语法树是源代码的结构化表示，可以用于分析、转换和生成代码
import ast
# 提供了 "pretty-printing"（漂亮打印）的功能，用于格式化输出数据结构，
# 使其更易读。pprint 模块通常用于打印复杂的数据结构，如字典、列表、嵌套的数据结构等。
import pprint
# 提供了 Abstract Base Classes（抽象基类）的支持
# 抽象基类是一种在面向对象编程中表示接口的方式，它定义了一组必须由继承类实现的抽象方法
# abc 模块的主要用途是帮助开发者创建和使用抽象基类
from abc import ABCMeta, abstractmethod
# 用于将数据转换为表格形式
from tabulate import tabulate
from typing import Dict


class BaseExp(metaclass=ABCMeta):
    """Basic class for any experiment."""

    def __init__(self):
        self.seed = None
        self.output_dir = "./YOLOX_outputs"
        self.print_interval = 100
        self.eval_interval = 10

    @abstractmethod
    # @abstractmethod 装饰器可以确保子类必须实现这些
    # 抽象方法，否则会在实例化时引发 TypeError
    def get_model(self) -> Module:
        pass

    @abstractmethod
    def get_data_loader(
        self, batch_size: int, is_distributed: bool
    ) -> Dict[str, torch.utils.data.DataLoader]:
        pass

    @abstractmethod
    def get_optimizer(self, batch_size: int) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def get_lr_scheduler(
        self, lr: float, iters_per_epoch: int, **kwargs
    ) -> LRScheduler:
        pass

    @abstractmethod
    def get_evaluator(self):
        pass

    @abstractmethod
    def eval(self, model, evaluator, weights):
        pass

    def __repr__(self):
        """定义了类的 __repr__ 方法，该方法用于返回对象的字符串表示形式,通常用于调试和开发目的
        在这个特定的实现中，__repr__ 方法使用了 tabulate 函数来创建一个漂亮的表格表示
        展示对象的属性键和对应的值"""
        table_header = ["keys", "values"]
        exp_table = [
            (str(k), pprint.pformat(v))
            for k, v in vars(self).items()
            if not k.startswith("_")
        ]
        return tabulate(exp_table, headers=table_header, tablefmt="fancy_grid")

    def merge(self, cfg_list):
        """用于将一组键值对的配置参数合并到当前对象的属性中"""
        assert len(cfg_list) % 2 == 0
        for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
            # only update value with same key
            if hasattr(self, k):
                src_value = getattr(self, k)
                src_type = type(src_value)
                if src_value is not None and src_type != type(v):
                # 如果属性值不为 None 且属性值的类型与新值 v 的类型不同
                # 尝试将新值 v 转换为与属性值相同的类型
                    try:
                        v = src_type(v)
                    except Exception:
                        v = ast.literal_eval(v)
                setattr(self, k, v)
