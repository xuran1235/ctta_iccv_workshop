from .pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from .transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder

from torch import nn
from ..builder import HEADS

from .decode_head import BaseDecodeHead

from ..losses.mask2former_loss import dice_loss, sigmoid_ce_loss, sigmoid_focal_loss
from ..losses.criterion import SetCriterion
from ..utils.matcher import HungarianMatcher
import torch

@HEADS.register_module()
class Mask2FormerHead(nn.Module):
    """
    From mask2former offical
    SEM_SEG_HEAD:
        NAME: "MaskFormerHead"
        IGNORE_VALUE: 255
        NUM_CLASSES: 150
        LOSS_WEIGHT: 1.0
        CONVS_DIM: 256
        MASK_DIM: 256
        NORM: "GN"
        # pixel decoder
        PIXEL_DECODER_NAME: "MSDeformAttnPixelDecoder"
        IN_FEATURES: ["res2", "res3", "res4", "res5"]
        DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES: ["res3", "res4", "res5"]
        COMMON_STRIDE: 4
        TRANSFORMER_ENC_LAYERS: 4
    MASK_FORMER:
        TRANSFORMER_DECODER_NAME: "MultiScaleMaskedTransformerDecoder"
        TRANSFORMER_IN_FEATURE: "multi_scale_pixel_decoder"
        DEEP_SUPERVISION: True
        NO_OBJECT_WEIGHT: 0.1
        CLASS_WEIGHT: 2.0
        MASK_WEIGHT: 5.0
        DICE_WEIGHT: 5.0
        HIDDEN_DIM: 256
        NUM_OBJECT_QUERIES: 100
        NHEADS: 8
        DROPOUT: 0.0
        DIM_FEEDFORWARD: 2048
        ENC_LAYERS: 0
        PRE_NORM: False
        ENFORCE_INPUT_PROJ: False
        SIZE_DIVISIBILITY: 32
        DEC_LAYERS: 10  # 9 decoder layers, add one for the loss on learnable query
        TRAIN_NUM_POINTS: 12544
        OVERSAMPLE_RATIO: 3.0
        IMPORTANCE_SAMPLE_RATIO: 0.75
    TEST:
        SEMANTIC_ON: True
        INSTANCE_ON: True
        PANOPTIC_ON: True
        OVERLAP_THRESHOLD: 0.8
        OBJECT_MASK_THRESHOLD: 0.8


    # From mmseg.mask2former
    decode_head=dict(in_channels=[192, 384, 768, 1536])
    type='Mask2FormerHead',
        in_channels=[256, 512, 1024, 2048],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_classes=num_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type='mmdet.MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(  # DeformableDetrTransformerEncoder
                num_layers=6,
                layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                    self_attn_cfg=dict(  # MultiScaleDeformableAttention
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=True,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True))),
                init_cfg=None),
            positional_encoding=dict(  # SinePositionalEncoding
                num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(  # SinePositionalEncoding
            num_feats=128, normalize=True),
        transformer_decoder=dict(  # Mask2FormerTransformerDecoder
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True)),
            init_cfg=None),
    """
    def __init__(self, **kwargs):        
        super(Mask2FormerHead, self).__init__()        
        self.pixel_decoder = self.pixel_decoder_init(**kwargs)
        self.predictor = self.predictor_init(**kwargs)
        self.align_corners = kwargs["align_corners"]
        self.num_classes = kwargs["num_classes"]
        #matcher = kwargs["train_cfg"]["assigner"]
        matcher = HungarianMatcher(
            cost_class = kwargs["loss_cls"]["loss_weight"],
            cost_mask  = kwargs["loss_mask"]["loss_weight"],
            cost_dice  = kwargs["loss_dice"]["loss_weight"],
            num_points = kwargs["train_cfg"]["num_points"]
        )
        weight_dict = weight_dict = {"loss_cls": kwargs["loss_cls"]["loss_weight"], "loss_mask": kwargs["loss_mask"]["loss_weight"], "loss_dice": kwargs["loss_dice"]["loss_weight"]}
        no_object_weight = 0.1 # original cfg do not have this para
        losses = ["labels", "masks"]

        self.criterion = SetCriterion(
            self.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=kwargs["train_cfg"]["num_points"],
            oversample_ratio=kwargs["train_cfg"]["oversample_ratio"],
            importance_sample_ratio=kwargs["train_cfg"]["importance_sample_ratio"],
            device=torch.device("cuda", 0) # TODO: verify the local_rank
        )
    
    def pixel_decoder_init(self, **kwargs):
        common_stride = 4 
        transformer_dropout = kwargs["pixel_decoder"]["encoder"]["layer_cfg"]["self_attn_cfg"]["dropout"]
        transformer_nheads = kwargs["pixel_decoder"]["encoder"]["layer_cfg"]["self_attn_cfg"]["num_heads"]
        transformer_dim_feedforward = kwargs["pixel_decoder"]["encoder"]["layer_cfg"]["ffn_cfg"]["feedforward_channels"]
        transformer_enc_layers = kwargs["pixel_decoder"]["encoder"]["num_layers"]
        conv_dim = kwargs["pixel_decoder"]["encoder"]["layer_cfg"]["self_attn_cfg"]["embed_dims"]
        mask_dim = kwargs["pixel_decoder"]["encoder"]["layer_cfg"]["self_attn_cfg"]["embed_dims"]
        transformer_in_features = ["res3", "res4", "res5"] # PD: In local config, original controled by num_outs = 3

        pixel_decoder = MSDeformAttnPixelDecoder(kwargs["input_shape"],
                                                transformer_dropout,
                                                transformer_nheads,
                                                transformer_dim_feedforward,
                                                transformer_enc_layers,
                                                conv_dim,
                                                mask_dim,
                                                transformer_in_features,
                                                common_stride)
        return pixel_decoder

    def predictor_init(self, **kwargs):
        in_channels = kwargs["transformer_decoder"]["layer_cfg"]["self_attn_cfg"]["embed_dims"]
        num_classes = kwargs["num_classes"]
        hidden_dim = kwargs["transformer_decoder"]["layer_cfg"]["self_attn_cfg"]["embed_dims"]
        num_queries = kwargs["num_queries"]
        nheads = kwargs["transformer_decoder"]["layer_cfg"]["self_attn_cfg"]["num_heads"]
        dim_feedforward = kwargs["transformer_decoder"]["layer_cfg"]["ffn_cfg"]["feedforward_channels"]
        dec_layers = kwargs["transformer_decoder"]["num_layers"]
        pre_norm = False # PD: no matching hyper parameters
        mask_dim = kwargs["transformer_decoder"]["layer_cfg"]["self_attn_cfg"]["embed_dims"]
        enforce_input_project = False
        mask_classification = True
        predictor = MultiScaleMaskedTransformerDecoder(in_channels, 
                                                        num_classes, 
                                                        mask_classification,
                                                        hidden_dim,
                                                        num_queries,
                                                        nheads,
                                                        dim_feedforward,
                                                        dec_layers,
                                                        pre_norm,
                                                        mask_dim,
                                                        enforce_input_project)
        return predictor

    def forward(self, features, mask=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)       
        predictions = self.predictor(multi_scale_features, mask_features, mask)        
        return predictions

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits,gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        return self.forward(inputs)
    
    def losses(self, segs, gts):
        return self.criterion(segs, gts)

    # TODO
    def init_weights(self,):
        pass
    
