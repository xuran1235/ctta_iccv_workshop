_base_ = [
    '../_base_/models/setr_sam_pup.py',
    '../_base_/datasets/shiftdataset.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint='/home/user/xr/cotta/pretrain/pureSAM.pth')),
    decode_head=dict(img_size=1024, align_corners=False, num_conv=4, upsampling_method='bilinear',
                     num_upsampe_layer=4),
    auxiliary_head=[dict(
        type='VisionTransformerUpHead',
        in_channels=1024,
        channels=512,
        in_index=0,
        img_size=768,
        embed_dim=1024,
        num_classes=14,
        norm_cfg=norm_cfg,
        num_conv=2,
        upsampling_method='bilinear',
        num_upsampe_layer=2,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
        type='VisionTransformerUpHead',
        in_channels=1024,
        channels=512,
        in_index=1,
        img_size=768,
        embed_dim=1024,
        num_classes=14,
        norm_cfg=norm_cfg,
        num_conv=2,
        upsampling_method='bilinear',
        num_upsampe_layer=2,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
        type='VisionTransformerUpHead',
        in_channels=1024,
        channels=512,
        in_index=2,
        img_size=768,
        embed_dim=1024,
        num_classes=14,
        norm_cfg=norm_cfg,
        num_conv=2,
        upsampling_method='bilinear',
        num_upsampe_layer=2,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
        type='VisionTransformerUpHead',
        in_channels=1024,
        channels=512,
        in_index=3,
        img_size=768,
        embed_dim=1024,
        num_classes=14,
        norm_cfg=norm_cfg,
        num_conv=2,
        upsampling_method='bilinear',
        num_upsampe_layer=2,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
    ])

optimizer = dict(lr=0.01, weight_decay=0.0,
                 paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)})
                 )

crop_size = (1024, 1024)
test_cfg = dict(mode='slide', crop_size=crop_size, stride=(512, 512))
find_unused_parameters = True
data = dict(samples_per_gpu=2)
