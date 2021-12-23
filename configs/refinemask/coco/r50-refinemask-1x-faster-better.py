_base_ = './r50-refinemask-1x.py'

model = dict(
    roi_head=dict(
        type='SimpleRefineRoIHead',
        mask_head=dict(
            _delete_=True,
            type='SimpleRefineMaskHead',
            num_convs_instance=2,
            num_convs_semantic=4,
            conv_in_channels_instance=256,
            conv_in_channels_semantic=256,
            conv_kernel_size_instance=3,
            conv_kernel_size_semantic=3,
            conv_out_channels_instance=256,
            conv_out_channels_semantic=256,
            conv_cfg=None,
            norm_cfg=None,
            fusion_type='MultiBranchFusionAvg', # slighly better than w/o global avg feature
            dilations=[1, 3, 5],
            semantic_out_stride=4,
            stage_num_classes=[80, 80, 80, 1],  # use class-agnostic classifier in the last stage
            stage_sup_size=[14, 28, 56, 112],
            pre_upsample_last_stage=False,      # compute logits and then upsample them in the last stage
            upsample_cfg=dict(type='bilinear', scale_factor=2),
            loss_cfg=dict(
                type='BARCrossEntropyLoss',
                stage_instance_loss_weight=[0.5, 0.75, 0.75, 1.0],
                boundary_width=2,
                start_stage=1))
    )
)

data = dict(samples_per_gpu=2)  # Train RefineMask 2 images per gpu with less than 11G memory cost.
