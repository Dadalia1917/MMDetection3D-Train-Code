# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from glob import glob

from mmengine.config import Config
from mmengine.logging import print_log

from mmdet3d.apis import LidarDet3DInferencer


def parse_args():
    parser = argparse.ArgumentParser(description='ImVoxelNet 3D inference')
    parser.add_argument(
        '--dataset', 
        choices=['kitti-3d-car', 'sunrgbd-3d-10class'],
        default='kitti-3d-car',
        help='choose dataset config')
    parser.add_argument('--config', help='config file')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--img', help='image file path')
    parser.add_argument('--pcd', help='point cloud file path (optional for ImVoxelNet)')
    parser.add_argument(
        '--out-dir', 
        default='./outputs_imvoxelnet', 
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
        'kitti-3d-car': 'configs/imvoxelnet/imvoxelnet_8xb4_kitti-3d-car.py',
        'sunrgbd-3d-10class': 'configs/imvoxelnet/imvoxelnet_2xb4_sunrgbd-3d-10class.py',
    }
    return config_map.get(dataset, config_map['kitti-3d-car'])


def get_checkpoint_file(dataset):
    """根据数据集选择检查点文件"""
    checkpoint_map = {
        'kitti-3d-car': 'runs/outputs/ImVoxelNet_kitti-3d-car/latest.pth',
        'sunrgbd-3d-10class': 'runs/outputs/ImVoxelNet_sunrgbd-3d-10class/latest.pth',
    }
    return checkpoint_map.get(dataset, checkpoint_map['kitti-3d-car'])


def get_sample_files(dataset):
    """根据数据集获取示例文件"""
    if dataset == 'kitti-3d-car':
        # 优先查找test1.png
        img_patterns = [
            'test1.png',
            './test1.png', 
            'demo/data/test1.png',
            'data/kitti/training/image_2/*.png',
            'demo/data/*.jpg',
            'demo/data/*.png'
        ]
        pcd_patterns = [
            'data/kitti/training/velodyne_reduced/*.bin',
            'data/kitti/training/velodyne/*.bin',
            'demo/data/*.bin'
        ]
    else:  # sunrgbd
        img_patterns = [
            'test1.png',
            './test1.png', 
            'demo/data/test1.png',
            'data/sunrgbd/image/*.jpg',
            'demo/data/*.jpg',
            'demo/data/*.png'
        ]
        pcd_patterns = [
            'demo/data/*.bin'
        ]
    
    img_file = None
    for pattern in img_patterns:
        if '*' in pattern:
            imgs = glob(pattern)
            if imgs:
                img_file = imgs[0]
                break
        else:
            if os.path.exists(pattern):
                img_file = pattern
                break
    
    pcd_file = None
    for pattern in pcd_patterns:
        pcds = glob(pattern)
        if pcds:
            pcd_file = pcds[0]
            break
    
    return img_file, pcd_file


def create_simple_info_file(img_file, pcd_file=None):
    """为图像文件创建简单的信息文件（ImVoxelNet需要）"""
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
        },
        'cam_intrinsic': cam_intrinsic,
        'lidar2img': np.eye(4),
        'sample_idx': 0
    }
    
    # 如果有点云文件，添加点云信息
    if pcd_file:
        info['lidar_points'] = {
            'lidar_path': pcd_file,
            'num_pts_feats': 4
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
    local_imvoxelnet_model = './models/imvoxelnet/imvoxelnet_4x8_kitti-3d-car_20210830_003014-3d0ffdf4.pth'
    
    if os.path.exists(local_imvoxelnet_model):
        checkpoint_file = local_imvoxelnet_model
        print_log(f"Using local ImVoxelNet model: {checkpoint_file}", logger='current')
    elif not checkpoint_file or not os.path.exists(checkpoint_file):
        checkpoint_file = None
        print_log("Using pretrained weights (will download automatically)", logger='current')
    
    # 获取输入文件
    if args.img:
        img_file = args.img
        pcd_file = args.pcd
    else:
        img_file, pcd_file = get_sample_files(args.dataset)
        
    if not img_file or not os.path.exists(img_file):
        print_log(f"Image file not found: {img_file}", logger='current')
        print_log("Please specify an image file with --img", logger='current')
        return
    
    print_log(f"Using config: {config_file}", logger='current')
    print_log(f"Processing image: {img_file}", logger='current')
    if pcd_file:
        print_log(f"Processing point cloud: {pcd_file}", logger='current')
    else:
        print_log("No point cloud file provided (ImVoxelNet can work with image only)", logger='current')
    
    try:
        # 创建信息文件
        info_file = create_simple_info_file(img_file, pcd_file)
        
        # 对于ImVoxelNet，使用LidarDet3DInferencer
        # 因为它是基于体素的方法，即使主要使用图像
        # 如果没有本地权重文件，使用None让MMDetection3D自动下载预训练权重
        inferencer = LidarDet3DInferencer(
            model=config_file,
            weights=checkpoint_file if checkpoint_file and os.path.exists(checkpoint_file) else None,
            device=args.device
        )
        
        # 准备输入数据
        if pcd_file and os.path.exists(pcd_file):
            inputs = pcd_file
        else:
            # 如果没有点云，使用图像（需要适当的信息文件）
            inputs = {
                'points': None,  # ImVoxelNet可以只用图像
                'img': img_file,
                'infos': info_file
            }
        
        # 运行推理
        print_log("Starting ImVoxelNet 3D detection inference...", logger='current')
        results = inferencer(
            inputs=inputs,
            pred_score_thr=args.pred_score_thr,
            return_vis=True,
            print_result=True,
            out_dir=args.out_dir
        )
        
        # 保存可视化结果
        img_name = osp.splitext(osp.basename(img_file))[0]
        result_file = f'result_ImVoxelNet_{img_name}.png'
        
        if results['visualization'] is not None:
            import cv2
            cv2.imwrite(result_file, results['visualization'][0])
            print_log(f"Visualization result saved to: {result_file}", logger='current')
        
        print_log("ImVoxelNet inference completed successfully!", logger='current')
        
        # 清理临时文件
        if os.path.exists(info_file):
            os.unlink(info_file)
            
    except Exception as e:
        print_log(f"Error during inference: {str(e)}", logger='current')
        print_log("Note: ImVoxelNet requires specific data format and may need both image and point cloud", logger='current')
        if 'info_file' in locals() and os.path.exists(info_file):
            os.unlink(info_file)


if __name__ == '__main__':
    main()
