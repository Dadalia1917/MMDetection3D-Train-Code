# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import glob
from argparse import ArgumentParser

from mmengine.logging import print_log

from mmdet3d.apis import LidarDet3DInferencer


def parse_args():
    parser = ArgumentParser(description='SECOND inference demo')
    parser.add_argument(
        '--pcd', 
        help='Point cloud file path. If not specified, will use a sample from data/kitti')
    parser.add_argument(
        '--dataset', 
        choices=['kitti-3class', 'kitti-car', 'waymo'],
        default='kitti-3class',
        help='choose dataset config')
    parser.add_argument('--config', help='Config file path')
    parser.add_argument('--checkpoint', help='Checkpoint file path')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=0.3,
        help='bbox score threshold')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='.',
        help='Output directory of prediction and visualization results.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show online visualization results')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=-1,
        help='The interval of show (s). Demo will be blocked in showing'
        'results, if wait_time is -1. Defaults to -1.')
    parser.add_argument(
        '--no-save-vis',
        action='store_true',
        help='Do not save detection visualization results')
    parser.add_argument(
        '--no-save-pred',
        action='store_true',
        help='Do not save detection prediction results')
    parser.add_argument(
        '--print-result',
        action='store_true',
        default=True,
        help='Whether to print the results.')
    
    return parser.parse_args()


def get_config_file(dataset):
    """根据数据集选择配置文件"""
    config_map = {
        'kitti-3class': 'configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-3class.py',
        'kitti-car': 'configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-car.py',
        'waymo': 'configs/second/second_hv_secfpn_sbn-all_16xb2-2x_waymoD5-3d-3class.py'
    }
    return config_map.get(dataset, config_map['kitti-3class'])


def get_checkpoint_file(dataset):
    """根据数据集选择检查点文件"""
    checkpoint_map = {
        'kitti-3class': 'runs/outputs/SECOND_kitti-3class/latest.pth',
        'kitti-car': 'runs/outputs/SECOND_kitti-car/latest.pth',
        'waymo': 'runs/outputs/SECOND_waymo/latest.pth'
    }
    return checkpoint_map.get(dataset, checkpoint_map['kitti-3class'])


def find_sample_pcd():
    """从数据集中找一个示例点云文件"""
    pcd_patterns = [
        'data/kitti/training/velodyne_reduced/*.bin',
        'data/kitti/training/velodyne/*.bin',
        'demo/data/*.bin'
    ]
    
    for pattern in pcd_patterns:
        pcd_files = glob.glob(pattern)
        if pcd_files:
            return pcd_files[0]
    
    raise FileNotFoundError(
        "No point cloud files found. Please specify --pcd argument or "
        "ensure KITTI data is available in data/kitti/training/velodyne_reduced/"
    )


def main():
    args = parse_args()
    
    # 获取点云文件路径
    if args.pcd:
        pcd_file = args.pcd
    else:
        pcd_file = find_sample_pcd()
        print_log(f"Using sample point cloud: {pcd_file}", logger='current')
    
    if not os.path.exists(pcd_file):
        raise FileNotFoundError(f"Point cloud file not found: {pcd_file}")
    
    # 获取配置文件
    if args.config:
        config_file = args.config
    else:
        config_file = get_config_file(args.dataset)
    
    # 获取检查点文件
    if args.checkpoint:
        checkpoint_file = args.checkpoint
    else:
        checkpoint_file = get_checkpoint_file(args.dataset)
        
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_file}\n"
            f"Please train the model first using: python train_SECOND.py --dataset {args.dataset} --amp"
        )
    
    print_log(f"Point cloud file: {pcd_file}", logger='current')
    print_log(f"Config file: {config_file}", logger='current')
    print_log(f"Checkpoint file: {checkpoint_file}", logger='current')
    
    # 初始化推理器
    inferencer = LidarDet3DInferencer(
        model=config_file,
        weights=checkpoint_file,
        device=args.device
    )
    
    # 准备调用参数
    call_args = {
        'inputs': {'points': pcd_file},
        'pred_score_thr': args.pred_score_thr,
        'out_dir': args.out_dir,
        'show': args.show,
        'wait_time': args.wait_time,
        'no_save_vis': args.no_save_vis,
        'no_save_pred': args.no_save_pred,
        'print_result': args.print_result
    }
    
    if call_args['no_save_vis'] and call_args['no_save_pred']:
        call_args['out_dir'] = ''
        
    # NOTE: 如果您的操作环境没有显示设备，建议添加以下代码
    # 为了在无头环境中可视化
    if not args.show:
        call_args['show'] = False
    
    # 运行推理
    print_log("Starting inference...", logger='current')
    results = inferencer(**call_args)
    
    # 保存结果图像到项目根目录
    if not args.no_save_vis:
        import shutil
        vis_files = glob.glob(os.path.join(args.out_dir, 'vis_data', '*.png')) + \
                   glob.glob(os.path.join(args.out_dir, 'vis_data', '*.jpg'))
        
        if vis_files:
            # 找到最新的可视化文件
            latest_vis = max(vis_files, key=os.path.getctime)
            result_filename = f"result_SECOND_{args.dataset}.png"
            shutil.copy2(latest_vis, result_filename)
            print_log(f"Visualization result saved to: {result_filename}", logger='current')
    
    if args.out_dir and not (args.no_save_vis and args.no_save_pred):
        print_log(f"All results have been saved to: {args.out_dir}", logger='current')


if __name__ == '__main__':
    main()
