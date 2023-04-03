import torch.nn as nn
from model.resnet import ResNet
from model.fpn import FPN
from model.mask_feature import MaskFeatHead
from model.head import SOLOv2Head


class SOLOv2(nn.Module):
    def __init__(self, cfg):
        super(SOLOv2, self).__init__()
        self.backbone = ResNet(depth=cfg.resnet_depth, frozen_stages=1)
        self.neck = FPN(cfg.fpn_in_c)
        self.mask_feat_head = MaskFeatHead(num_classes=cfg.mask_feat_num_classes)

        self.bbox_head = SOLOv2Head(num_classes=cfg.num_classes, stacked_convs=cfg.head_stacked_convs,
                                    scale_ranges=cfg.head_scale_ranges, seg_feat_channels=cfg.head_seg_feat_c,
                                    ins_out_channels=cfg.head_ins_out_c)
        self.postprocess_cfg = cfg.postprocess_para

        if cfg.mode == 'train':
            self.init_weights(pretrained=cfg.pretrained)

        self.mode = cfg.mode
        self.onnx_trans = cfg.onnx_trans
        self.onnx_shape = cfg.onnx_shape

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print('load model from: {}'.format(pretrained))

        self.backbone.init_weights(pretrained=pretrained)
        self.neck.init_weights()
        self.mask_feat_head.init_weights()
        self.bbox_head.init_weights()

    def forward(self, img, gt_labels=None, gt_bboxes=None, gt_masks=None, ori_shape=None, resize_shape=None,
                post_mode='detect'):
        if self.mode == 'onnx':
            img = self.onnx_trans(img)

        x = self.backbone(img)
        x = self.neck(x)  # (bs, fpn_c_out, H/4, W/4), down sample ratio 4, 8, 16, 32, 64
        # (bs, mask_feat_num_classes, H/4, W/4)
        mask_feat_pred = self.mask_feat_head(x[self.mask_feat_head.start_level:self.mask_feat_head.end_level + 1])
        cate_preds, kernel_preds = self.bbox_head(x)  # (bs, S, S, num_classes), (bs, out_c, S, S)

        if self.training:
            return self.bbox_head.loss(cate_preds, kernel_preds, mask_feat_pred, gt_bboxes, gt_labels, gt_masks)
        else:
            if self.mode == 'onnx':
                resize_shape = self.onnx_shape

            # if self.mode == 'onnx':
            #     seg_result = get_seg_scripted(cate_preds, kernel_preds, mask_feat_pred)
            # else:

            seg_result = self.bbox_head.get_seg(cate_preds, kernel_preds, mask_feat_pred, ori_shape,
                                                resize_shape, self.postprocess_cfg, post_mode)

            return seg_result

# class SOLOv2(nn.Module):
#     def __init__(self, cfg):
#         super(SOLOv2, self).__init__()
#         self.backbone = ResNet(depth=cfg.resnet_depth, frozen_stages=1)
#         self.neck = FPN(cfg.fpn_in_c)
#         self.mask_feat_head = MaskFeatHead(num_classes=cfg.mask_feat_num_classes)
#
#         self.bbox_head = SOLOv2Head(num_classes=cfg.num_classes, stacked_convs=cfg.head_stacked_convs,
#                                     scale_ranges=cfg.head_scale_ranges, seg_feat_channels=cfg.head_seg_feat_c,
#                                     ins_out_channels=cfg.head_ins_out_c, in_detect=(cfg.mode == 'detect'))
#         self.postprocess_cfg = cfg.postprocess_para
#
#         if cfg.mode == 'train':
#             self.init_weights(pretrained=cfg.pretrained)
#
#         self.onnx_trans = cfg.onnx_trans
#         self.onnx_shape = cfg.onnx_shape
#
#     def init_weights(self, pretrained=None):
#         if pretrained is not None:
#             print('load model from: {}'.format(pretrained))
#
#         self.backbone.init_weights(pretrained=pretrained)
#         self.neck.init_weights()
#         self.mask_feat_head.init_weights()
#         self.bbox_head.init_weights()
#
#     def forward(self, img):
#         img = self.onnx_trans(img)
#         x = self.backbone(img)
#         x = self.neck(x)
#         mask_feat_pred = self.mask_feat_head(x[self.mask_feat_head.start_level:self.mask_feat_head.end_level + 1])
#         cate_preds, kernel_preds = self.bbox_head(x)
#         get_seg = torch.jit.script(get_seg_scripted)
#         seg_result = get_seg(cate_preds, kernel_preds, mask_feat_pred)
#         print(seg_result)
#         return seg_result
