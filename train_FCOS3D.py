# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
from pathlib import Path

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet3d.utils import replace_ceph_backend


def parse_args():
    parser = argparse.ArgumentParser(description='Train FCOS3D mono3D detector')
    parser.add_argument(
        '--dataset', 
        choices=['nus-mono3d', 'nus-mono3d-finetune'],
        default='nus-mono3d',
        help='choose dataset config')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--sync_bn',
        type=str,
        choices=['none', 'pytorch', 'apex'],
        default='none',
        help='the type of syncBN method')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
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
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--ceph', action='store_true', help='Use ceph as data backend.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def download_file(url, local_path, force=False):
    """下载文件到指定路径"""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    if local_path.exists() and not force:
        print_log(f"Model already exists: {local_path}", logger='current')
        return str(local_path)
    
    print_log(f"Downloading {url} to {local_path}", logger='current')
    
    try:
        import requests
        from tqdm import tqdm
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(local_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print_log(f"Successfully downloaded: {local_path}", logger='current')
        return str(local_path)
        
    except ImportError:
        print_log("Installing required packages for download...", logger='current')
        os.system("pip install requests tqdm")
        return download_file(url, local_path, force)
    
    except Exception as e:
        print_log(f"Download failed: {str(e)}", logger='current', level=logging.ERROR)
        if local_path.exists():
            local_path.unlink()
        return None


def ensure_fcos3d_models():
    """确保FCOS3D所需的预训练模型已下载"""
    models_dir = Path('./models/fcos3d')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # FCOS3D nuScenes预训练权重
    fcos3d_url = 'https://download.openmmlab.com/mmdetection3d/v0.1.0_models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_20210715_235813-4bed5239.pth'
    fcos3d_path = models_dir / 'fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_20210715_235813-4bed5239.pth'
    
    # FCOS3D nuScenes finetune预训练权重
    fcos3d_finetune_url = 'https://download.openmmlab.com/mmdetection3d/v0.1.0_models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth'
    fcos3d_finetune_path = models_dir / 'fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth'
    
    print_log("Checking FCOS3D model dependencies...", logger='current')
    
    # 可选：下载FCOS3D预训练权重（用于推理）
    fcos3d_result = download_file(fcos3d_url, fcos3d_path)
    fcos3d_finetune_result = download_file(fcos3d_finetune_url, fcos3d_finetune_path)
    
    return fcos3d_result, fcos3d_finetune_result


def get_config_file(dataset):
    """根据数据集选择配置文件"""
    config_map = {
        'nus-mono3d': 'configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d.py',
        'nus-mono3d-finetune': 'configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py',
    }
    return config_map.get(dataset, config_map['nus-mono3d'])


def main():
    args = parse_args()

    # load config
    if args.config:
        cfg = Config.fromfile(args.config)
    else:
        config_file = get_config_file(args.dataset)
        cfg = Config.fromfile(config_file)
    
    # 确保预训练模型已下载（可选，主要用于推理）
    fcos3d_path, fcos3d_finetune_path = ensure_fcos3d_models()

    # TODO: We will unify the ceph support approach with other OpenMMLab repos
    if args.ceph:
        cfg = replace_ceph_backend(cfg)

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        model_name = f'FCOS3D_{args.dataset}'
        cfg.work_dir = osp.join('./runs/outputs', model_name)

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # convert BatchNorm layers
    if args.sync_bn != 'none':
        cfg.sync_bn = args.sync_bn

    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    cfg.resume = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
