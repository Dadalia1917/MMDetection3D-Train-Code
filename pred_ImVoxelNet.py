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
    
    # 定义类别名称
    class_names = {
        0: 'Car',
        1: 'Pedestrian', 
        2: 'Car',  
        3: 'Cyclist',
        4: 'Van',
        5: 'Truck',
        6: 'Bus'
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
    result_path = "result_ImVoxelNet_3d.png"
    cv2.imwrite(result_path, img)
    return True


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
        
        # 保存并增强可视化结果
        if results['visualization'] is not None:
            import cv2
            import tempfile
            
            # 先保存临时可视化文件
            temp_vis_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            cv2.imwrite(temp_vis_file.name, results['visualization'][0])
            
            # 如果有检测结果，增强可视化图像
            if results and 'predictions' in results and results['predictions']:
                predictions = results['predictions'][0]
                if 'bboxes_3d' in predictions and len(predictions['bboxes_3d']) > 0:
                    labels = predictions.get('labels_3d', [])
                    scores = predictions.get('scores_3d', [])
                    bboxes = predictions['bboxes_3d']
                    
                    # 增强可视化图像（添加标签和置信度）
                    if enhance_3d_visualization(temp_vis_file.name, labels, scores, bboxes):
                        print_log(f"Enhanced 3D detection visualization saved to: result_ImVoxelNet_3d.png", logger='current')
                    else:
                        # 如果增强失败，直接复制原图
                        import shutil
                        shutil.copy2(temp_vis_file.name, "result_ImVoxelNet_3d.png")
                        print_log(f"3D detection visualization saved to: result_ImVoxelNet_3d.png", logger='current')
                else:
                    # 没有检测结果，直接复制
                    import shutil
                    shutil.copy2(temp_vis_file.name, "result_ImVoxelNet_3d.png")
                    print_log(f"3D detection visualization saved to: result_ImVoxelNet_3d.png", logger='current')
            else:
                # 没有预测结果，直接复制
                import shutil
                shutil.copy2(temp_vis_file.name, "result_ImVoxelNet_3d.png")
                print_log(f"3D detection visualization saved to: result_ImVoxelNet_3d.png", logger='current')
            
            # 清理临时文件
            import os
            os.unlink(temp_vis_file.name)
        
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
