# dataset settings
dataset_type = 'ShiftDataset'
data_split_type = 'images_dis_train'
data_root = '/data/SHIFT/continuous/videos/1x/'
eval_data_root = '/data/SHIFT/continuous/videos/1x/'
csv_root = csv_root = eval_data_root + 'val/front/seq.csv'
eval_csv_root = eval_data_root + 'val/front/seq.csv'


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (1024, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 500),
        # img_scale=(1024, 1024),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75,2.0],
        flip=False,
        transforms=[
            # dict(type='Resize', img_scale=(1024,1024), keep_ratio=False),
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/front/img',
        ann_dir='train/front/semseg',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/front/img/',
        ann_dir='test/front/semseg/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/front/img/',
        ann_dir='test/front/semseg/',
        pipeline=test_pipeline))