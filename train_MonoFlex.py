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


def init_cuda_context():
    """Initialize CUDA context to avoid numba CUDA JIT errors"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.init()
            _ = torch.zeros(1).cuda()
            torch.cuda.synchronize()
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description='Train MonoFlex mono3D detector')
    parser.add_argument(
        '--dataset', 
        choices=['kitti-mono3d'],
        default='kitti-mono3d',
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


def ensure_monoflex_models():
    """确保MonoFlex所需的预训练模型已下载"""
    models_dir = Path('./models/monoflex')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # DLA34骨干网络权重
    dla34_url = 'http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth'
    dla34_path = models_dir / 'dla34-ba72cf86.pth'
    
    # MonoFlex KITTI预训练权重
    monoflex_url = 'https://download.openmmlab.com/mmdetection3d/v0.1.0_models/monoflex/monoflex_dla34_pytorch_dlaneck_gn-all_2x4_6x_kitti-mono3d_20211228_027553-d46d9bb0.pth'
    monoflex_path = models_dir / 'monoflex_dla34_pytorch_dlaneck_gn-all_2x4_6x_kitti-mono3d_20211228_027553-d46d9bb0.pth'
    
    print_log("Checking MonoFlex model dependencies...", logger='current')
    
    # 下载DLA34权重
    dla34_result = download_file(dla34_url, dla34_path)
    
    # 可选：下载MonoFlex预训练权重（用于推理）
    monoflex_result = download_file(monoflex_url, monoflex_path)
    
    return dla34_result, monoflex_result


def get_config_file(dataset):
    """根据数据集选择配置文件"""
    config_map = {
        'kitti-mono3d': 'configs/monoflex/monoflex_dla34_pytorch_dlaneck_gn-all_2x4_6x_kitti-mono3d.py',
    }
    return config_map.get(dataset, config_map['kitti-mono3d'])


def main():
    args = parse_args()
    
    init_cuda_context()

    # load config
    if args.config:
        cfg = Config.fromfile(args.config)
    else:
        config_file = get_config_file(args.dataset)
        cfg = Config.fromfile(config_file)
    
    # 确保预训练模型已下载
    dla34_path, monoflex_path = ensure_monoflex_models()
    
    # 使用本地下载的DLA34预训练权重
    if dla34_path and os.path.exists(dla34_path):
        print_log(f"Using local DLA34 backbone: {dla34_path}", logger='current', level=logging.INFO)
        cfg.model.backbone.init_cfg.checkpoint = dla34_path
    else:
        print_log("Failed to download DLA34 backbone, will use default URL", logger='current', level=logging.WARNING)

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
        model_name = f'MonoFlex_{args.dataset}'
        cfg.work_dir = osp.join('./runs/outputs', model_name)

    # MonoFlex模型暂时不建议使用AMP训练，可能出现数据类型不匹配错误
    if args.amp is True:
        print_log(
            'AMP training might cause issues with MonoFlex model.',
            logger='current',
            level=logging.WARNING)
        print_log(
            'Training will continue without AMP to avoid potential errors.',
            logger='current',
            level=logging.INFO)

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
