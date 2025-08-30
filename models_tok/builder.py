# Copyright (c) OpenMMLab. All rights reserved.
# from mmengine import build_from_cfg
from torch import nn

from mmcv.utils import build_from_cfg
from mmpose.models.builder import BACKBONES, HEADS, LOSSES, NECKS, POSENETS

#此函数接受一个配置字典 cfg、一个注册表以及可选的默认参数，
# 并相应地构建一个模块。它支持构建单个模块或模块列表。
def build(cfg, registry, default_args=None):
    """Build a module.
    Args:
        cfg (dict, list[dict]): The config of modules, it is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.
    Returns:
        nn.Module: A built nn module.
    """

    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)

    return build_from_cfg(cfg, registry, default_args)

#此函数根据提供的配置构建一个 backbone 模块。
def build_backbone(cfg):
    """Build backbone."""
    return build(cfg, BACKBONES)

#此函数根据提供的配置构建一个 neck 模块。
def build_neck(cfg):
    """Build neck."""
    return build(cfg, NECKS)

#此函数根据提供的配置构建一个 head 模块。
def build_head(cfg):
    """Build head."""
    return build(cfg, HEADS)

#此函数根据提供的配置构建一个 loss 模块。
def build_loss(cfg):
    """Build loss."""
    return build(cfg, LOSSES)

#此函数根据提供的配置构建一个 posenet 模块。
def build_posenet(cfg):
    """Build posenet."""
    return build(cfg, POSENETS)