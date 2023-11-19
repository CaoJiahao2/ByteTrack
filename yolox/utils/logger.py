#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from loguru import logger

import inspect
# inspect is a built-in module in Python 3.8, used to get information about live objects
import os
import sys


def get_caller_name(depth=0):
    """
    Args:
        depth (int): Depth of caller conext, use 0 for caller depth. Default value: 0.

    Returns:
        str: module name of the caller
    通过追溯调用堆栈来获取调用者的信息
    函数返回调用者的模块名称,此函数对于在日志记录中标识消息来源的调试和错误处理非常有用
    """
    # the following logic is a little bit faster than inspect.stack() logic
    frame = inspect.currentframe().f_back
    for _ in range(depth):
        frame = frame.f_back

    return frame.f_globals["__name__"]


class StreamToLoguru:
    """
    stream object that redirects writes to a logger instance.
    可以将特定模块的输出重定向到 loguru 实例，实现更灵活和可定制的日志记录
    """

    def __init__(self, level="INFO", caller_names=("apex", "pycocotools")):
        """
        Args:
            level(str): log level string of loguru. Default value: "INFO".
            caller_names(tuple): caller names of redirected module.
                Default value: (apex, pycocotools).
        """
        self.level = level
        self.linebuf = ""
        self.caller_names = caller_names

    def write(self, buf):
        full_name = get_caller_name(depth=1)
        module_name = full_name.rsplit(".", maxsplit=-1)[0]
        if module_name in self.caller_names:
            for line in buf.rstrip().splitlines():
                # use caller level log
                logger.opt(depth=2).log(self.level, line.rstrip())
        else:
            sys.__stdout__.write(buf)

    def flush(self):
        """
        flush method is required for stream object.
        """
        pass


def redirect_sys_output(log_level="INFO"):
    """
    redirect stdout and stderr to loguru.
    """
    redirect_logger = StreamToLoguru(log_level)
    sys.stderr = redirect_logger
    sys.stdout = redirect_logger


def setup_logger(save_dir, distributed_rank=0, filename="log.txt", mode="a"):
    """setup logger for training and testing.
    Args:
        save_dir(str): location to save log file
        distributed_rank(int): device rank when multi-gpu environment
        filename (string): log save name.
        mode(str): log file write mode, `append` or `override`. default is `a`.
        在训练和测试过程中设置日志记录器，以便更好地追踪和记录程序运行过程中的信息
    Return:
        logger instance.
    """
    loguru_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    logger.remove() #  清除之前的日志记录器
    save_file = os.path.join(save_dir, filename)
    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)
    # only keep logger in rank0 process
    if distributed_rank == 0:
        # 在排名为0的进程中，将日志同时输出到标准错误流（stderr）和指定的日志文件中
        logger.add(
            sys.stderr,
            format=loguru_format,
            level="INFO",
            enqueue=True,
        )
        logger.add(save_file)

    # redirect stdout/stderr to loguru
    redirect_sys_output("INFO")
