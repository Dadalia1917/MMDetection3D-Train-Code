# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, ConfigDict, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet3d.utils import replace_ceph_backend


def parse_args():
    parser = argparse.ArgumentParser(description='Test SMOKE mono3D model')
    parser.add_argument(
        '--dataset', 
        choices=['kitti-mono3d'],
        default='kitti-mono3d',
        help='choose dataset config')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file path')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved')
    parser.add_argument(
        '--score-thr', type=float, default=0.1, help='bbox score threshold')
    parser.add_argument(
        '--task',
        type=str,
        default='mono_det',
        choices=[
            'mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg',
            'multi-modality_det'
        ],
        help='Determine the visualization method depending on the task.')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def get_config_file(dataset):
    """根据数据集选择配置文件"""
    config_map = {
        'kitti-mono3d': 'configs/smoke/smoke_dla34_dlaneck_gn-all_4xb8-6x_kitti-mono3d.py',
    }
    return config_map.get(dataset, config_map['kitti-mono3d'])


def get_checkpoint_file(dataset):
    """根据数据集选择检查点文件"""
    checkpoint_map = {
        'kitti-mono3d': 'runs/outputs/SMOKE_kitti-mono3d/latest.pth',
    }
    return checkpoint_map.get(dataset, checkpoint_map['kitti-mono3d'])


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualization_hook['test_out_dir'] = args.show_dir
        visualization_hook['vis_task'] = args.task
        visualization_hook['score_thr'] = args.score_thr
    else:
        cfg.default_hooks['visualization'] = dict(
            type='Det3DVisualizationHook',
            draw=True,
            show=args.show,
            wait_time=args.wait_time,
            test_out_dir=args.show_dir,
            vis_task=args.task,
            score_thr=args.score_thr
        )
    return cfg


def main():
    args = parse_args()

    # load config
    if args.config:
        cfg = Config.fromfile(args.config)
    else:
        config_file = get_config_file(args.dataset)
        cfg = Config.fromfile(config_file)

    # set checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = get_checkpoint_file(args.dataset)
        
    if not osp.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        model_name = f'SMOKE_{args.dataset}'
        cfg.work_dir = osp.join('./runs/test_outputs', model_name)

    cfg.load_from = checkpoint_path

    # enable visualization
    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    # build the runner from config
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
