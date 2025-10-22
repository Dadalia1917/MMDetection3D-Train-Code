backend_args = None
class_names = [
    'Pedestrian',
    'Cyclist',
    'Car',
]
data_root = 'data/kitti/'
dataset_type = 'KittiDataset'
default_hooks = dict(
    checkpoint=dict(interval=-1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
eval_pipeline = [
    dict(backend_args=None, type='LoadImageFromFileMono3D'),
    dict(keys=[
        'img',
    ], type='Pack3DDetInputs'),
]
find_unused_parameters = True
input_modality = dict(use_camera=True, use_lidar=False)
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 72
metainfo = dict(classes=[
    'Pedestrian',
    'Cyclist',
    'Car',
])
model = dict(
    backbone=dict(
        depth=34,
        in_channels=3,
        init_cfg=dict(
            checkpoint='models\\smoke\\dla34-ba72cf86.pth', type='Pretrained'),
        norm_cfg=dict(num_groups=32, type='GN'),
        type='DLANet'),
    bbox_head=dict(
        attr_branch=(),
        bbox_code_size=7,
        bbox_coder=dict(
            base_depth=(
                28.01,
                16.32,
            ),
            base_dims=(
                (
                    0.88,
                    1.73,
                    0.67,
                ),
                (
                    1.78,
                    1.7,
                    0.58,
                ),
                (
                    3.88,
                    1.63,
                    1.53,
                ),
            ),
            code_size=7,
            type='SMOKECoder'),
        cls_branch=(256, ),
        conv_bias=True,
        dcn_on_last_conv=False,
        diff_rad_by_sin=False,
        dim_channel=[
            3,
            4,
            5,
        ],
        dir_branch=(),
        dir_offset=0,
        feat_channels=64,
        group_reg_dims=(8, ),
        in_channels=64,
        loss_attr=None,
        loss_bbox=dict(
            loss_weight=0.0033333333333333335,
            reduction='sum',
            type='mmdet.L1Loss'),
        loss_cls=dict(loss_weight=1.0, type='mmdet.GaussianFocalLoss'),
        loss_dir=dict(
            loss_weight=1.0, type='mmdet.CrossEntropyLoss', use_sigmoid=False),
        num_attrs=0,
        num_classes=3,
        ori_channel=[
            6,
            7,
        ],
        pred_attrs=False,
        pred_velo=False,
        reg_branch=((256, ), ),
        stacked_convs=0,
        strides=None,
        type='SMOKEMono3DHead',
        use_direction_classifier=False),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='Det3DDataPreprocessor'),
    neck=dict(
        end_level=5,
        in_channels=[
            16,
            32,
            64,
            128,
            256,
            512,
        ],
        norm_cfg=dict(num_groups=32, type='GN'),
        start_level=2,
        type='DLANeck'),
    test_cfg=dict(local_maximum_kernel=3, max_per_img=100, topK=100),
    train_cfg=None,
    type='SMOKEMono3D')
optim_wrapper = dict(
    clip_grad=None,
    loss_scale='dynamic',
    optimizer=dict(lr=0.00025, type='Adam'),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=72,
        gamma=0.1,
        milestones=[
            50,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='kitti_infos_val.pkl',
        backend_args=None,
        box_type_3d='Camera',
        data_prefix=dict(img='training/image_2'),
        data_root='data/kitti/',
        load_type='fov_image_based',
        metainfo=dict(classes=[
            'Pedestrian',
            'Cyclist',
            'Car',
        ]),
        modality=dict(use_camera=True, use_lidar=False),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFileMono3D'),
            dict(down_ratio=4, img_scale=(
                1280,
                384,
            ), type='AffineResize'),
            dict(keys=[
                'img',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='KittiDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/kitti/kitti_infos_val.pkl',
    backend_args=None,
    metric='bbox',
    type='KittiMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFileMono3D'),
    dict(down_ratio=4, img_scale=(
        1280,
        384,
    ), type='AffineResize'),
    dict(keys=[
        'img',
    ], type='Pack3DDetInputs'),
]
train_cfg = dict(max_epochs=300, type='EpochBasedTrainLoop', val_interval=5)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='kitti_infos_train.pkl',
        backend_args=None,
        box_type_3d='Camera',
        data_prefix=dict(img='training/image_2'),
        data_root='data/kitti/',
        load_type='fov_image_based',
        metainfo=dict(classes=[
            'Pedestrian',
            'Cyclist',
            'Car',
        ]),
        modality=dict(use_camera=True, use_lidar=False),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFileMono3D'),
            dict(
                type='LoadAnnotations3D',
                with_attr_label=False,
                with_bbox=True,
                with_bbox_3d=True,
                with_bbox_depth=True,
                with_label=True,
                with_label_3d=True),
            dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
            dict(
                aug_prob=0.3,
                shift_scale=(
                    0.2,
                    0.4,
                ),
                type='RandomShiftScale'),
            dict(down_ratio=4, img_scale=(
                1280,
                384,
            ), type='AffineResize'),
            dict(
                keys=[
                    'img',
                    'gt_bboxes',
                    'gt_bboxes_labels',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                    'centers_2d',
                    'depths',
                ],
                type='Pack3DDetInputs'),
        ],
        test_mode=False,
        type='KittiDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFileMono3D'),
    dict(
        type='LoadAnnotations3D',
        with_attr_label=False,
        with_bbox=True,
        with_bbox_3d=True,
        with_bbox_depth=True,
        with_label=True,
        with_label_3d=True),
    dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
    dict(aug_prob=0.3, shift_scale=(
        0.2,
        0.4,
    ), type='RandomShiftScale'),
    dict(down_ratio=4, img_scale=(
        1280,
        384,
    ), type='AffineResize'),
    dict(
        keys=[
            'img',
            'gt_bboxes',
            'gt_bboxes_labels',
            'gt_bboxes_3d',
            'gt_labels_3d',
            'centers_2d',
            'depths',
        ],
        type='Pack3DDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='kitti_infos_val.pkl',
        backend_args=None,
        box_type_3d='Camera',
        data_prefix=dict(img='training/image_2'),
        data_root='data/kitti/',
        load_type='fov_image_based',
        metainfo=dict(classes=[
            'Pedestrian',
            'Cyclist',
            'Car',
        ]),
        modality=dict(use_camera=True, use_lidar=False),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFileMono3D'),
            dict(down_ratio=4, img_scale=(
                1280,
                384,
            ), type='AffineResize'),
            dict(keys=[
                'img',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='KittiDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/kitti/kitti_infos_val.pkl',
    backend_args=None,
    metric='bbox',
    type='KittiMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './runs/outputs\\SMOKE_kitti-mono3d'
