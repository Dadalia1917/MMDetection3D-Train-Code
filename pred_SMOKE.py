# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import glob
from argparse import ArgumentParser

from mmengine.logging import print_log

from mmdet3d.apis import MonoDet3DInferencer


def parse_args():
    parser = ArgumentParser(description='SMOKE mono3D inference demo')
    parser.add_argument(
        '--img', 
        help='Image file path. If not specified, will use test1.png or find a sample from data/kitti')
    parser.add_argument(
        '--dataset', 
        choices=['kitti-mono3d'],
        default='kitti-mono3d',
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
        'kitti-mono3d': 'configs/smoke/smoke_dla34_dlaneck_gn-all_4xb8-6x_kitti-mono3d.py',
    }
    return config_map.get(dataset, config_map['kitti-mono3d'])


def get_checkpoint_file(dataset):
    """根据数据集选择检查点文件"""
    checkpoint_map = {
        'kitti-mono3d': 'runs/outputs/SMOKE_kitti-mono3d/latest.pth',
    }
    return checkpoint_map.get(dataset, checkpoint_map['kitti-mono3d'])


def find_sample_image():
    """找一个示例图像文件"""
    # 优先使用项目根目录的test1.png
    if os.path.exists('test1.png'):
        return 'test1.png'
    
    # 寻找KITTI数据集中的图像文件
    img_patterns = [
        'data/kitti/training/image_2/*.png',
        'demo/data/*.png',
        'demo/data/*.jpg'
    ]
    
    for pattern in img_patterns:
        img_files = glob.glob(pattern)
        if img_files:
            return img_files[0]
    
    raise FileNotFoundError(
        "No image files found. Please specify --img argument or "
        "place a test1.png file in the project root directory, or "
        "ensure KITTI data is available in data/kitti/training/image_2/"
    )


def find_kitti_info_file():
    """找到对应的KITTI信息文件"""
    info_patterns = [
        'data/kitti/kitti_infos_val.pkl',
        'data/kitti/kitti_infos_test.pkl',
        'data/kitti/kitti_infos_train.pkl'
    ]
    
    for info_file in info_patterns:
        if os.path.exists(info_file):
            return info_file
    
    # 如果找不到info文件，返回None，使用预设的相机参数
    return None


def main():
    args = parse_args()
    
    # 获取图像文件路径
    if args.img:
        img_file = args.img
    else:
        img_file = find_sample_image()
        print_log(f"Using sample image: {img_file}", logger='current')
    
    if not os.path.exists(img_file):
        raise FileNotFoundError(f"Image file not found: {img_file}")
    
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
        # 尝试使用预训练权重
        print_log(
            f"Local checkpoint not found: {checkpoint_file}\n"
            f"Will try to use pretrained weights from OpenMMLab",
            logger='current'
        )
        checkpoint_file = None  # 让推理器自动下载预训练权重
    
    # 找KITTI信息文件（用于相机参数）
    info_file = find_kitti_info_file()
    
    print_log(f"Image file: {img_file}", logger='current')
    print_log(f"Config file: {config_file}", logger='current')
    if checkpoint_file:
        print_log(f"Checkpoint file: {checkpoint_file}", logger='current')
    else:
        print_log("Using pretrained weights (will download automatically)", logger='current')
    
    if info_file:
        print_log(f"KITTI info file: {info_file}", logger='current')
    else:
        print_log("No KITTI info file found, using default camera parameters", logger='current')
    
    # 初始化推理器
    # 如果没有本地权重文件，使用None让MMDetection3D自动下载预训练权重
    inferencer = MonoDet3DInferencer(
        model=config_file,
        weights=checkpoint_file if checkpoint_file and os.path.exists(checkpoint_file) else None,
        device=args.device
    )
    
    # 准备输入数据
    if info_file:
        # 使用KITTI数据集的信息文件
        inputs = {
            'img': img_file,
            'infos': info_file
        }
    else:
        # 仅使用图像文件（推理器会使用默认相机参数）
        inputs = {
            'img': img_file,
            'infos': img_file  # 对于单张图像，可以重复使用
        }
    
    # 准备调用参数
    call_args = {
        'inputs': inputs,
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
    print_log("Starting SMOKE mono3D inference...", logger='current')
    try:
        results = inferencer(**call_args)
        
        # 保存结果图像到项目根目录
        if not args.no_save_vis:
            import shutil
            vis_files = glob.glob(os.path.join(args.out_dir, 'vis_data', '*.png')) + \
                       glob.glob(os.path.join(args.out_dir, 'vis_data', '*.jpg'))
            
            if vis_files:
                # 找到最新的可视化文件
                latest_vis = max(vis_files, key=os.path.getctime)
                result_filename = f"result_SMOKE_3d.png"
                shutil.copy2(latest_vis, result_filename)
                print_log(f"3D detection result saved to: {result_filename}", logger='current')
            
            # 如果输入是test1.png，同时复制一份作为说明
            if 'test1.png' in img_file.lower():
                try:
                    shutil.copy2(img_file, "result_user_image.png")
                    print_log(f"User image copied to: result_user_image.png", logger='current')
                except:
                    pass
        
        if args.out_dir and not (args.no_save_vis and args.no_save_pred):
            print_log(f"All results have been saved to: {args.out_dir}", logger='current')
            
    except Exception as e:
        print_log(f"Inference failed: {str(e)}", logger='current', level=logging.ERROR)
        print_log(
            "This might be due to:\n"
            "1. Missing camera calibration information\n"
            "2. Image format not compatible with KITTI\n"
            "3. Model checkpoint not available\n"
            "Please check the image and try again.",
            logger='current', level=logging.WARNING
        )
        raise


if __name__ == '__main__':
    main()
