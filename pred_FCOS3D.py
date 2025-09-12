# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from glob import glob

from mmengine.config import Config
from mmengine.logging import print_log

from mmdet3d.apis import MonoDet3DInferencer


def parse_args():
    parser = argparse.ArgumentParser(description='FCOS3D mono3D inference')
    parser.add_argument(
        '--dataset', 
        choices=['nus-mono3d', 'nus-mono3d-finetune'],
        default='nus-mono3d',
        help='choose dataset config')
    parser.add_argument('--config', help='config file')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--img', help='image file path')
    parser.add_argument(
        '--out-dir', 
        default='./outputs_fcos3d', 
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
        'nus-mono3d': 'configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d.py',
        'nus-mono3d-finetune': 'configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py',
    }
    return config_map.get(dataset, config_map['nus-mono3d'])


def get_checkpoint_file(dataset):
    """根据数据集选择检查点文件"""
    checkpoint_map = {
        'nus-mono3d': 'runs/outputs/FCOS3D_nus-mono3d/latest.pth',
        'nus-mono3d-finetune': 'runs/outputs/FCOS3D_nus-mono3d-finetune/latest.pth',
    }
    return checkpoint_map.get(dataset, checkpoint_map['nus-mono3d'])


def get_sample_img(dataset):
    """根据数据集获取示例图像文件"""
    # 优先查找test1.png，然后查找其他图像文件
    img_patterns = [
        'test1.png',
        './test1.png', 
        'demo/data/test1.png',
        'data/nuscenes/samples/CAM_FRONT/*.jpg',
        'demo/data/*.jpg',
        'demo/data/*.png',
        'data/kitti/training/image_2/*.png'  # 备用KITTI图像
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
    """为图像文件创建简单的信息文件（FCOS3D需要）"""
    import numpy as np
    import pickle
    import tempfile
    
    # 创建临时信息文件
    temp_file = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
    
    # 简单的相机内参 (nuScenes典型值)
    cam_intrinsic = np.array([
        [1252.8131021997343, 0., 826.588114781398],
        [0., 1252.8131021997343, 469.9846626224581],
        [0., 0., 1.]
    ])
    
    # 创建信息字典
    info = {
        'image': {
            'image_path': img_file,
            'image_shape': (900, 1600, 3),  # nuScenes典型尺寸
        },
        'calib': {
            'P2': cam_intrinsic.flatten(),
            'R0_rect': np.eye(3).flatten(),
            'Tr_velo_to_cam': np.eye(4).flatten()
        },
        'cam_intrinsic': cam_intrinsic,
        'lidar2img': np.eye(4)  # FCOS3D可能需要这个
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
    local_fcos3d_models = [
        './models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_20210715_235813-4bed5239.pth',
        './models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth'
    ]
    
    for local_model in local_fcos3d_models:
        if os.path.exists(local_model):
            checkpoint_file = local_model
            print_log(f"Using local FCOS3D model: {checkpoint_file}", logger='current')
            break
    else:
        if not checkpoint_file or not os.path.exists(checkpoint_file):
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
        print_log("Note: FCOS3D is primarily designed for nuScenes dataset", logger='current')
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
        print_log("Starting FCOS3D 3D detection inference...", logger='current')
        results = inferencer(
            inputs=inputs,
            pred_score_thr=args.pred_score_thr,
            return_vis=True,
            print_result=True,
            out_dir=args.out_dir
        )
        
        # 保存可视化结果
        img_name = osp.splitext(osp.basename(img_file))[0]
        result_file = f'result_FCOS3D_{img_name}.png'
        
        if results['visualization'] is not None:
            import cv2
            cv2.imwrite(result_file, results['visualization'][0])
            print_log(f"Visualization result saved to: {result_file}", logger='current')
        
        print_log("FCOS3D inference completed successfully!", logger='current')
        
        # 清理临时文件
        if os.path.exists(info_file):
            os.unlink(info_file)
            
    except Exception as e:
        print_log(f"Error during inference: {str(e)}", logger='current')
        print_log("Note: FCOS3D requires nuScenes-style data format", logger='current')
        if 'info_file' in locals() and os.path.exists(info_file):
            os.unlink(info_file)


if __name__ == '__main__':
    main()
