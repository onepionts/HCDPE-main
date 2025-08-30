# Copyright (c) OpenMMLab. All rights reserved.
import os
os.environ['CUDA_VISIBLE_DEVICES']="0"
import argparse
import copy

import os.path as osp
import time

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import get_git_hash


from mmpose import __version__
from mmpose.apis import train_model
from mmpose.utils import collect_env, get_root_logger
from mmpose.datasets import build_dataset

from models import build_posenet



# import torch.distributed as dist
# dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)
# import torch.distributed as dist
# dist.init_process_group('gloo', init_method='file:///temp/somefile', rank=0, world_size=1)

# os.environ['CUDA_VISIBLE_DEVICES']='0'
# device=torch.device('cuda:0')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a pose model')
    parser.add_argument('--config',
                        # default='configs/hcdpe_base_tokenizer.py',
                        default='D:/code/HCDPE/utils/configs/hcdpe_base_tokenizer.py',
                        help='train config file path')
    parser.add_argument('--work-dir',
                        # default='output/to1.3/to',
                        default='output/tokenizer',
                        help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from',
        help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():

    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed

    model = build_posenet(cfg.model)
    datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    if cfg.checkpoint_config is not None:
        # save mmpose version, config file content
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmpose_version=__version__ + get_git_hash(digits=7),
            config=cfg.pretty_text,
        )
    print("..........................................................................")
    print(model)
    print("//////////////////////")
    # 获取模型的状态字典
    # model_state_dict = model.state_dict()
    # for key in model_state_dict.keys():
    #     print("之前"+key)
    #     # if key.startswith('keypoint_head.') and key not in [
    #     if key not in[
    #         'keypoint_head.mixer_norm_layer.ff.0.weight',
    #         'keypoint_head.mixer_norm_layer.ff.0.bias',
    #         'keypoint_head.mixer_norm_layer.ff.1.weight',
    #         'keypoint_head.mixer_norm_layer.ff.1.bias',
    #         'keypoint_head.cls_pred_layer.weight',
    #         'keypoint_head.cls_pred_layer.bias'
    #     ]:
    #         print("dongj"+key)
    #         model_state_dict[key].requires_grad = False
    #
    #
    # # 将修改后的状态字典加载回模型
    # model.load_state_dict(model_state_dict)
    # 修改学习率为更大的值
    # model_name = ['cls_pred_layer.weight', 'cls_pred_layer.bias',
    #               'mixer_norm_layer.ff.0.weight', 'mixer_norm_layer.ff.0.bias',
    #               'mixer_norm_layer.ff.1.weight', 'mixer_norm_layer.ff.1.bias'
    #               ]

    # for name, params in self.keypoint_head.named_parameters():
    #     print("zhiqiande" + name)
    #     if (name not in model_name):
    #         print(name + "名称")
    #         params.requires_grad = False
    print("/......")
    print(model)
    # print(model_state_dict)
    print("............................/")
    # print(datasets)
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
