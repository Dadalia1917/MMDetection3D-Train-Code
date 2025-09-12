# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from glob import glob

from mmengine.config import Config
from mmengine.logging import print_log

from mmdet3d.apis import MonoDet3DInferencer


def parse_args():
    parser = argparse.ArgumentParser(description='MonoFlex mono3D inference')
    parser.add_argument(
        '--dataset', 
        choices=['kitti-mono3d'],
        default='kitti-mono3d',
        help='choose dataset config')
    parser.add_argument('--config', help='config file')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--img', help='image file path')
    parser.add_argument(
        '--out-dir', 
        default='./outputs_monoflex', 
        help='output directory')
    parser.add_argument(
        '--score-thr', 
        type=float, 
        default=0.3, 
        help='score threshold')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=0.3,
        help='bbox score threshold')
    args = parser.parse_args()
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


def get_sample_img(dataset):
    """根据数据集获取示例图像文件"""
    if dataset == 'kitti-mono3d':
        # 优先查找test1.png，然后查找其他图像文件
        img_patterns = [
            'test1.png',
            './test1.png', 
            'demo/data/test1.png',
            'data/kitti/training/image_2/*.png',
            'demo/data/*.jpg',
            'demo/data/*.png'
        ]
        
        for pattern in img_patterns:
            if '*' in pattern:
                imgs = glob(pattern)
                if imgs:
                    return imgs[0]
            else:
                if os.path.exists(pattern):
                    return pattern
    
    return None


def create_simple_info_file(img_file):
    """为图像文件创建简单的信息文件（MonoFlex需要）"""
    import numpy as np
    import pickle
    import tempfile
    
    # 创建临时信息文件
    temp_file = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
    
    # 简单的相机内参 (KITTI典型值)
    cam_intrinsic = np.array([
        [721.5377, 0., 609.5593],
        [0., 721.5377, 172.8540],
        [0., 0., 1.]
    ])
    
    # 创建信息字典
    info = {
        'image': {
            'image_path': img_file,
            'image_shape': (375, 1242, 3),  # KITTI典型尺寸
        },
        'calib': {
            'P2': cam_intrinsic.flatten(),
            'R0_rect': np.eye(3).flatten(),
            'Tr_velo_to_cam': np.eye(4).flatten()
        }
    }
    
    # 保存信息文件
    with open(temp_file.name, 'wb') as f:
        pickle.dump([info], f)
    
    return temp_file.name


def main():
    args = parse_args()
    
    # 获取配置文件
    if args.config:
        config_file = args.config
    else:
        config_file = get_config_file(args.dataset)
    
    if not os.path.exists(config_file):
        print_log(f"Config file not found: {config_file}", logger='current')
        print_log("Please ensure MMDetection3D is properly installed", logger='current')
        return
    
    # 获取检查点文件
    if args.checkpoint:
        checkpoint_file = args.checkpoint
    else:
        checkpoint_file = get_checkpoint_file(args.dataset)
    
    # 检查本地模型
    local_monoflex_model = './models/monoflex/monoflex_dla34_pytorch_dlaneck_gn-all_2x4_6x_kitti-mono3d_20211228_027553-d46d9bb0.pth'
    
    if os.path.exists(local_monoflex_model):
        checkpoint_file = local_monoflex_model
        print_log(f"Using local MonoFlex model: {checkpoint_file}", logger='current')
    elif not os.path.exists(checkpoint_file):
        checkpoint_file = None
        print_log("Using pretrained weights (will download automatically)", logger='current')
    
    # 获取图像文件
    if args.img:
        img_file = args.img
    else:
        img_file = get_sample_img(args.dataset)
        
    if not img_file or not os.path.exists(img_file):
        print_log(f"Image file not found: {img_file}", logger='current')
        print_log("Please specify an image file with --img", logger='current')
        return
    
    print_log(f"Using config: {config_file}", logger='current')
    print_log(f"Processing image: {img_file}", logger='current')
    
    try:
        # 创建信息文件
        info_file = create_simple_info_file(img_file)
        
        # 初始化推理器
        # 如果没有本地权重文件，使用None让MMDetection3D自动下载预训练权重
        inferencer = MonoDet3DInferencer(
            model=config_file,
            weights=checkpoint_file if checkpoint_file and os.path.exists(checkpoint_file) else None,
            device=args.device
        )
        
        # 准备输入数据
        inputs = {
            'img': img_file,
            'infos': info_file
        }
        
        # 运行推理
        print_log("Starting MonoFlex 3D detection inference...", logger='current')
        results = inferencer(
            inputs=inputs,
            pred_score_thr=args.pred_score_thr,
            return_vis=True,
            print_result=True,
            out_dir=args.out_dir
        )
        
        # 保存可视化结果
        img_name = osp.splitext(osp.basename(img_file))[0]
        result_file = f'result_MonoFlex_{img_name}.png'
        
        if results['visualization'] is not None:
            import cv2
            cv2.imwrite(result_file, results['visualization'][0])
            print_log(f"Visualization result saved to: {result_file}", logger='current')
        
        print_log("MonoFlex inference completed successfully!", logger='current')
        
        # 清理临时文件
        if os.path.exists(info_file):
            os.unlink(info_file)
            
    except Exception as e:
        print_log(f"Error during inference: {str(e)}", logger='current')
        if 'info_file' in locals() and os.path.exists(info_file):
            os.unlink(info_file)


if __name__ == '__main__':
    main()
