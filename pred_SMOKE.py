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
        'kitti-mono3d': 'models/smoke/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_20210929_015553-d46d9bb0.pth',
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


def find_bbox_positions(img):
    """找到图像中检测框的位置"""
    import cv2
    import numpy as np
    
    # 转换为HSV以便更好地检测红色
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 扩大红色检测范围，确保能检测到所有红色框
    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 100, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # 创建红色掩码
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    # 形态学操作，连接断开的线条
    kernel = np.ones((2,2), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.dilate(red_mask, kernel, iterations=1)
    
    # 找到轮廓
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 过滤并收集所有有效的边界框
    valid_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # 降低面积阈值以检测更多框
            x, y, w, h = cv2.boundingRect(contour)
            # 更宽松的形状检查
            if w > 20 and h > 20 and w < img.shape[1] and h < img.shape[0]:
                # 计算边界框的中心点
                center_x = x + w // 2
                center_y = y + h // 2
                valid_boxes.append((center_x, center_y, x, y, w, h, area))
    
    # 按照从左到右、从远到近的顺序排序（这样更符合检测结果的顺序）
    # 主要按X坐标排序，然后按面积（距离的反向指标）排序
    valid_boxes.sort(key=lambda box: (box[0], -box[6]))
    
    # 返回边界框位置信息
    bbox_positions = []
    for center_x, center_y, x, y, w, h, area in valid_boxes:
        label_x = max(5, x)  # 确保标签不会超出左边界
        label_y = max(25, y - 10)  # 标签在框的上方，确保不会超出上边界
        bbox_positions.append((center_x, center_y, label_x, label_y, w, h))
    
    return bbox_positions


def enhance_3d_visualization(vis_img_path, labels, scores, bboxes):
    """在3D可视化图像上添加标签和置信度信息"""
    import cv2
    import numpy as np
    
    # 读取可视化图像
    img = cv2.imread(vis_img_path)
    if img is None:
        return False
    
    # 定义KITTI类别名称
    class_names = {
        0: 'Car',
        1: 'Pedestrian', 
        2: 'Car',  # SMOKE中label=2是Car
        3: 'Cyclist'
    }
    
    # 文本样式设置
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # 使用绿色文字
    text_color = (0, 255, 0)  # 绿色
    bg_color = (0, 0, 0)      # 黑色背景
    
    # 找到检测框的位置
    bbox_positions = find_bbox_positions(img)
    
    print(f"Found {len(bbox_positions)} bbox positions for {len(bboxes)} detections")
    
    # 如果检测到的框数量与检测结果数量一致，直接按顺序匹配
    if len(bbox_positions) == len(bboxes):
        for i, (label, score, bbox) in enumerate(zip(labels, scores, bboxes)):
            class_name = class_names.get(int(label), f"Class_{int(label)}")
            label_text = f"{class_name} {score:.2f}"
            
            center_x, center_y, label_x, label_y = bbox_positions[i][:4]
            
            # 计算文本尺寸
            (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
            
            # 添加背景矩形
            cv2.rectangle(img, 
                         (label_x - 3, label_y - text_h - 3), 
                         (label_x + text_w + 3, label_y + 3), 
                         bg_color, -1)
            
            # 添加文字
            cv2.putText(img, label_text, (label_x, label_y), font, font_scale, text_color, thickness)
    
    # 如果检测框数量不匹配，使用更智能的匹配方法
    elif len(bbox_positions) > 0:
        # 根据3D检测结果的距离来匹配检测框（距离越近，框应该越大）
        detection_distances = []
        for bbox in bboxes:
            x, y, z = bbox[0], bbox[1], bbox[2]
            distance = np.sqrt(x**2 + y**2 + z**2)
            detection_distances.append(distance)
        
        # 按距离排序检测结果的索引
        sorted_indices = sorted(range(len(detection_distances)), key=lambda i: detection_distances[i])
        
        # 按面积排序检测框（面积越大，距离可能越近）
        sorted_bbox_positions = sorted(bbox_positions, key=lambda pos: pos[4] * pos[5], reverse=True)
        
        # 匹配并添加标签
        for i, det_idx in enumerate(sorted_indices):
            if i < len(sorted_bbox_positions):
                label = labels[det_idx]
                score = scores[det_idx]
                class_name = class_names.get(int(label), f"Class_{int(label)}")
                label_text = f"{class_name} {score:.2f}"
                
                center_x, center_y, label_x, label_y = sorted_bbox_positions[i][:4]
                
                # 计算文本尺寸
                (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
                
                # 添加背景矩形
                cv2.rectangle(img, 
                             (label_x - 3, label_y - text_h - 3), 
                             (label_x + text_w + 3, label_y + 3), 
                             bg_color, -1)
                
                # 添加文字
                cv2.putText(img, label_text, (label_x, label_y), font, font_scale, text_color, thickness)
    
    # 保存增强后的图像
    result_path = "result_SMOKE_3d.png"
    cv2.imwrite(result_path, img)
    return True


def create_simple_info_file(img_file):
    """为图像文件创建简单的信息文件（SMOKE需要）"""
    import numpy as np
    import pickle
    import tempfile
    import os.path as osp
    
    # 创建临时信息文件
    temp_file = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
    
    # KITTI数据集典型的相机内参
    cam_intrinsic = np.array([
        [721.5377, 0., 609.5593],
        [0., 721.5377, 172.854],
        [0., 0., 1.]
    ])
    
    # 相机到雷达的变换矩阵
    lidar2cam = np.array([
        [7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
        [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
        [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01],
        [0., 0., 0., 1.]
    ], dtype=np.float32)
    
    # 计算lidar2img变换矩阵
    lidar2img = cam_intrinsic @ lidar2cam[:3, :]
    
    # 获取图像基本信息
    img_name = osp.basename(img_file)
    
    # 创建符合MMDetection3D格式的数据信息
    data_info = {
        'sample_idx': 0,
        'images': {
            'CAM2': {
                'img_path': img_name,
                'lidar2cam': lidar2cam,
                'cam2img': cam_intrinsic,
                'lidar2img': lidar2img,
                'height': 375,  # KITTI标准高度
                'width': 1242   # KITTI标准宽度
            }
        }
    }
    
    # 创建完整的信息字典
    info_dict = {
        'metainfo': {
            'dataset': 'kitti',
            'version': '1.0'
        },
        'data_list': [data_info]
    }
    
    # 保存信息文件
    with open(temp_file.name, 'wb') as f:
        pickle.dump(info_dict, f)
    
    return temp_file.name


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
    
    print_log(f"Image file: {img_file}", logger='current')
    print_log(f"Config file: {config_file}", logger='current')
    if checkpoint_file:
        print_log(f"Checkpoint file: {checkpoint_file}", logger='current')
    else:
        print_log("Using pretrained weights (will download automatically)", logger='current')
    
    # 初始化推理器
    # 如果没有本地权重文件，使用None让MMDetection3D自动下载预训练权重
    inferencer = MonoDet3DInferencer(
        model=config_file,
        weights=checkpoint_file if checkpoint_file and os.path.exists(checkpoint_file) else None,
        device=args.device
    )
    
    # 创建单图像信息文件
    try:
        info_file = create_simple_info_file(img_file)
        print_log(f"Created info file: {info_file}", logger='current')
    except Exception as e:
        print_log(f"Failed to create info file: {str(e)}", logger='current', level=logging.ERROR)
        raise
    
    # 准备输入数据
    inputs = {
        'img': img_file,
        'infos': info_file
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
        
        # 解析和显示检测结果
        if results and 'predictions' in results and results['predictions']:
            predictions = results['predictions'][0]  # 取第一个图像的结果
            
            if 'bboxes_3d' in predictions and len(predictions['bboxes_3d']) > 0:
                labels = predictions.get('labels_3d', [])
                scores = predictions.get('scores_3d', [])
                bboxes = predictions['bboxes_3d']
                
                print_log("="*60, logger='current')
                print_log("SMOKE 3D Detection Results:", logger='current')
                print_log("="*60, logger='current')
                print_log(f"Detected {len(bboxes)} objects:", logger='current')
                
                for i, (bbox, label, score) in enumerate(zip(bboxes, labels, scores)):
                    # bbox格式: [x, y, z, w, h, l, yaw]
                    x, y, z, w, h, l, yaw = bbox
                    label_name = "Car" if label == 2 else f"Class_{label}"
                    
                    print_log(f"\nObject {i+1}:", logger='current')
                    print_log(f"  类别: {label_name}", logger='current')
                    print_log(f"  置信度: {score:.3f}", logger='current')
                    print_log(f"  3D中心: ({x:.2f}, {y:.2f}, {z:.2f})", logger='current')
                    print_log(f"  尺寸 (长×宽×高): {l:.2f} × {w:.2f} × {h:.2f}", logger='current')
                    print_log(f"  旋转角度: {yaw:.3f} rad ({yaw*180/3.14159:.1f}°)", logger='current')
                
                print_log("="*60, logger='current')
            else:
                print_log("No 3D objects detected in the image.", logger='current')
        else:
            print_log("No detection results returned.", logger='current')
        
        # 保存并增强可视化结果
        if not args.no_save_vis:
            import shutil
            # 查找可视化文件
            vis_patterns = [
                os.path.join(args.out_dir, 'vis_camera', 'CAM2', '*.png'),
                os.path.join(args.out_dir, 'vis_camera', 'CAM2', '*.jpg'),
                os.path.join(args.out_dir, 'vis_data', '*.png'),
                os.path.join(args.out_dir, 'vis_data', '*.jpg'),
                os.path.join(args.out_dir, '*.png'),
                os.path.join(args.out_dir, '*.jpg')
            ]
            
            vis_files = []
            for pattern in vis_patterns:
                vis_files.extend(glob.glob(pattern))
            
            if vis_files:
                # 找到最新的可视化文件
                latest_vis = max(vis_files, key=os.path.getctime)
                
                # 如果有检测结果，增强可视化图像
                if results and 'predictions' in results and results['predictions']:
                    predictions = results['predictions'][0]
                    if 'bboxes_3d' in predictions and len(predictions['bboxes_3d']) > 0:
                        labels = predictions.get('labels_3d', [])
                        scores = predictions.get('scores_3d', [])
                        bboxes = predictions['bboxes_3d']
                        
                        # 增强可视化图像（添加标签和置信度）
                        if enhance_3d_visualization(latest_vis, labels, scores, bboxes):
                            print_log(f"Enhanced 3D detection visualization saved to: result_SMOKE_3d.png", logger='current')
                        else:
                            # 如果增强失败，直接复制原图
                            shutil.copy2(latest_vis, "result_SMOKE_3d.png")
                            print_log(f"3D detection visualization saved to: result_SMOKE_3d.png", logger='current')
                    else:
                        # 没有检测结果，直接复制
                        shutil.copy2(latest_vis, "result_SMOKE_3d.png")
                        print_log(f"3D detection visualization saved to: result_SMOKE_3d.png", logger='current')
                else:
                    # 没有预测结果，直接复制
                    shutil.copy2(latest_vis, "result_SMOKE_3d.png")
                    print_log(f"3D detection visualization saved to: result_SMOKE_3d.png", logger='current')
            else:
                print_log("Warning: No visualization files found", logger='current')
                print_log(f"Searched in: {args.out_dir}", logger='current')
        
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
    finally:
        # 清理临时文件
        try:
            if 'info_file' in locals() and os.path.exists(info_file):
                os.unlink(info_file)
                print_log(f"Cleaned up temporary info file: {info_file}", logger='current')
        except Exception as e:
            print_log(f"Failed to clean up temporary file: {str(e)}", logger='current')


if __name__ == '__main__':
    main()
