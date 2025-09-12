# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Test MonoFlex mono3D detector')
    parser.add_argument(
        '--dataset', 
        choices=['kitti-mono3d'],
        default='kitti-mono3d',
        help='choose dataset config')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='output result file in pickle format')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--tta', action='store_true', help='Test time augmentation')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def get_config_file(dataset):
    """根据数据集选择配置文件"""
    config_map = {
        'kitti-mono3d': 'configs/monoflex/monoflex_dla34_pytorch_dlaneck_gn-all_2x4_6x_kitti-mono3d.py',
    }
    return config_map.get(dataset, config_map['kitti-mono3d'])


def get_checkpoint_file(dataset):
    """根据数据集选择检查点文件"""
    checkpoint_map = {
        'kitti-mono3d': 'runs/outputs/MonoFlex_kitti-mono3d/latest.pth',
    }
    return checkpoint_map.get(dataset, checkpoint_map['kitti-mono3d'])


def main():
    args = parse_args()

    # load config
    if args.config:
        cfg = Config.fromfile(args.config)
    else:
        config_file = get_config_file(args.dataset)
        cfg = Config.fromfile(config_file)
        
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint or get_checkpoint_file(args.dataset)

    if args.show or args.show_dir:
        cfg.default_hooks.visualization.draw = True
        cfg.default_hooks.visualization.wait_time = args.wait_time
    if args.show:
        cfg.default_hooks.visualization.show = True
        cfg.default_hooks.visualization.vis_backends = [
            dict(type='LocalVisBackend')
        ]
    if args.show_dir:
        cfg.default_hooks.visualization.vis_backends = [
            dict(type='LocalVisBackend', 
                 save_dir=args.show_dir if osp.isabs(args.show_dir) 
                 else osp.join(cfg.work_dir, args.show_dir))
        ]

    if args.tta:
        if 'tta_model' not in cfg:
            cfg.tta_model = dict(type='Det3DTTAModel')
        if 'tta_pipeline' not in cfg:
            cfg.tta_pipeline = cfg.test_pipeline
        cfg.model = cfg.tta_model

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
