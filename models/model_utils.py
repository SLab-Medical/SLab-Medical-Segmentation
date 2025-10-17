import numpy as np
import torch
import torch.nn as nn
import pdb

def get_model(args):
    
    if args.model.dimension == '2d':
        if args.model == 'unet':
            from .dim2 import UNet
            if pretrain:
                raise ValueError('No pretrain model available')
            return UNet(args.in_chan, args.classes, args.base_chan, block=args.block)
        if args.model == 'unet++':
            from .dim2 import UNetPlusPlus
            if pretrain:
                raise ValueError('No pretrain model available')
            return UNetPlusPlus(args.in_chan, args.classes, args.base_chan)
        if args.model == 'attention_unet':
            from .dim2 import AttentionUNet
            if pretrain:
                raise ValueError('No pretrain model available')
            return AttentionUNet(args.in_chan, args.classes, args.base_chan)

        elif args.model == 'resunet':
            from .dim2 import UNet 
            if pretrain:
                raise ValueError('No pretrain model available')
            return UNet(args.in_chan, args.classes, args.base_chan, block=args.block)
        elif args.model == 'daunet':
            from .dim2 import DAUNet
            if pretrain:
                raise ValueError('No pretrain model available')
            return DAUNet(args.in_chan, args.classes, args.base_chan, block=args.block)

        elif args.model in ['medformer']:
            from .dim2 import MedFormer
            if pretrain:
                raise ValueError('No pretrain model available')
            return MedFormer(args.in_chan, args.classes, args.base_chan, conv_block=args.conv_block, conv_num=args.conv_num, trans_num=args.trans_num, num_heads=args.num_heads, fusion_depth=args.fusion_depth, fusion_dim=args.fusion_dim, fusion_heads=args.fusion_heads, map_size=args.map_size, proj_type=args.proj_type, act=nn.ReLU, expansion=args.expansion, attn_drop=args.attn_drop, proj_drop=args.proj_drop, aux_loss=args.aux_loss)


        elif args.model == 'transunet':
            from .dim2 import VisionTransformer as ViT_seg
            from .dim2.transunet import CONFIGS as CONFIGS_ViT_seg
            config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
            config_vit.n_classes = args.classes
            config_vit.n_skip = 3
            config_vit.patches.grid = (int(args.training_size[0]/16), int(args.training_size[1]/16))
            net = ViT_seg(config_vit, img_size=args.training_size[0], num_classes=args.classes)

            if pretrain:
                net.load_from(weights=np.load(args.init_model))

            return net
        
        elif args.model == 'swinunet':
            from .dim2 import SwinUnet
            from .dim2.swin_unet import SwinUnet_config
            config = SwinUnet_config()
            net = SwinUnet(config, img_size=224, num_classes=args.classes)
            
            if pretrain:
                net.load_from(args.init_model)

            return net



    elif args.model.dimension == '3d':
        if args.model.model_name == 'segformer3d':
            from .three_d.segformer.segformer import build_segformer3d_model

            return build_segformer3d_model({
                'in_channels': args.model.in_channels,
                'sr_ratios': args.model.sr_ratios,
                'embed_dims': args.model.embed_dims,
                'patch_kernel_size': args.model.patch_kernel_size,
                'patch_stride': args.model.patch_stride,
                'patch_padding': args.model.patch_padding,
                'mlp_ratios': args.model.mlp_ratios,
                'num_heads': args.model.num_heads,
                'depths': args.model.depths,
                'num_classes': args.model.num_classes,
                'decoder_head_embedding_dim': args.model.decoder_head_embedding_dim*(int(128/args.dataset.patch_size[0])**3),
                'decoder_dropout': args.model.decoder_dropout
            })



        elif args.model.model_name == 'segmamba':
            from .three_d.segmamba.segmamba import SegMamba
            return SegMamba(in_chans=args.model.in_channels,
                        out_chans=args.model.num_classes,
                        depths=args.model.depths,
                        feat_size=args.model.feat_size,)

        elif args.model.model_name == 'slim_unetr':
            from .three_d.slim_unetr.SlimUNETR import SlimUNETR
            return SlimUNETR(in_channels=args.model.in_channels,
                        out_channels=args.model.out_channels,
                        embed_dim=args.model.embed_dim,
                        embedding_dim=args.model.embedding_dim,
                        channels=args.model.channels,
                        blocks=args.model.blocks,
                        heads=args.model.heads,
                        r=args.model.r,
                        dropout=args.model.dropout,
                        )
        elif args.model.model_name == 'EfficientMedNeXt_T':
            from .three_d.MedNeXt.mednextv1.create_efficient_mednext import create_efficient_mednext
            return create_efficient_mednext(
                    args.model.in_channels, 
                    args.model.out_channels, 
                    'T', 
                    n_channels=args.model.feature_size,
                    kernel_sizes=args.model.kernel_sizes,
                    strides=args.model.strides,
                    uniform_dec_channels=args.model.n_decoder_channels,
                    deep_supervision=args.ds
                )
        elif args.model.model_name == 'EfficientMedNeXt_S':
            from .three_d.MedNeXt.mednextv1.create_efficient_mednext import create_efficient_mednext
            return create_efficient_mednext(
                    args.model.in_channels, 
                    args.model.out_channels, 
                    'S', 
                    n_channels=args.model.feature_size,
                    kernel_sizes=args.model.kernel_sizes,
                    strides=args.model.strides,
                    uniform_dec_channels=args.model.n_decoder_channels,
                    deep_supervision=args.ds
                )
        elif args.model.model_name == 'EfficientMedNeXt_M':
            from .three_d.MedNeXt.mednextv1.create_efficient_mednext import create_efficient_mednext
            return create_efficient_mednext(
                    args.model.in_channels, 
                    args.model.out_channels, 
                    'M', 
                    n_channels=args.model.feature_size,
                    kernel_sizes=args.model.kernel_sizes,
                    strides=args.model.strides,
                    uniform_dec_channels=args.model.n_decoder_channels,
                    deep_supervision=args.ds
                )
        elif args.model.model_name == 'EfficientMedNeXt_L':
            from .three_d.MedNeXt.mednextv1.create_efficient_mednext import create_efficient_mednext
            return create_efficient_mednext(
                    args.model.in_channels, 
                    args.model.out_channels, 
                    'L', 
                    n_channels=args.model.feature_size,
                    kernel_sizes=args.model.kernel_sizes,
                    strides=args.model.strides,
                    uniform_dec_channels=args.model.n_decoder_channels,
                    deep_supervision=args.model.deep_supervision
                )

        elif args.model.model_name == '3DUXNET':
            from .three_d.UXNet_3D.network_backbone import UXNET
            return UXNET(
                    in_chans=args.model.in_channels,
                    out_chans=args.model.out_channels,
                    depths=args.model.depths,
                    feat_size=args.model.feat_size,
                    drop_path_rate=args.model.drop_path_rate,
                    layer_scale_init_value=args.model.layer_scale_init_value,
                    spatial_dims=args.model.spatial_dims,
                )

        elif args.model.model_name == '3DUXNET_pretrain':
            from .three_d.UXNet_3D.network_backbone import UXNET
            model = UXNET(
                    in_chans=args.model.in_channels,
                    out_chans=args.model.pretrain_classes,
                    depths=args.model.depths,
                    feat_size=args.model.feat_size,
                    drop_path_rate=args.model.drop_path_rate,
                    layer_scale_init_value=args.model.layer_scale_init_value,
                    spatial_dims=args.model.spatial_dims,
                )
            model.load_state_dict(torch.load(args.model.pretrained_weights))
            model.out = UnetOutBlock(spatial_dims=args.model.spatial_dims, in_channels=args.model.in_channels, out_channels=args.model.out_channels)
            return model


        elif args.model == 'unet':
            from .three_d import UNet
            return UNet(args.in_chan, args.base_chan, num_classes=args.classes, scale=args.down_scale, norm=args.norm, kernel_size=args.kernel_size, block=args.block)
        elif args.model == 'unet++':
            from .dim3 import UNetPlusPlus
            return UNetPlusPlus(args.in_chan, args.base_chan, num_classes=args.classes, scale=args.down_scale, norm=args.norm, kernel_size=args.kernel_size, block=args.block)
        elif args.model == 'attention_unet':
            from .dim3 import AttentionUNet
            return AttentionUNet(args.in_chan, args.base_chan, num_classes=args.classes, scale=args.down_scale, norm=args.norm, kernel_size=args.kernel_size, block=args.block)

        elif args.model == 'medformer':
            from .dim3 import MedFormer

            return MedFormer(args.in_chan, args.classes, args.base_chan, map_size=args.map_size, conv_block=args.conv_block, conv_num=args.conv_num, trans_num=args.trans_num, num_heads=args.num_heads, fusion_depth=args.fusion_depth, fusion_dim=args.fusion_dim, fusion_heads=args.fusion_heads, expansion=args.expansion, attn_drop=args.attn_drop, proj_drop=args.proj_drop, proj_type=args.proj_type, norm=args.norm, act=args.act, kernel_size=args.kernel_size, scale=args.down_scale, aux_loss=args.aux_loss)
    
        elif args.model == 'unetr':
            from .dim3 import UNETR
            model = UNETR(args.in_chan, args.classes, args.training_size, feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12, pos_embed='perceptron', norm_name='instance', res_block=True)
            
            return model
        elif args.model == 'vtunet':
            from .dim3 import VTUNet
            model = VTUNet(args, args.classes)

            if pretrain:
                model.load_from(args)
            return model
        elif args.model == 'swin_unetr':
            from .dim3 import SwinUNETR
            model = SwinUNETR(args.window_size, args.in_chan, args.classes, feature_size=args.base_chan)

            if args.pretrain:
                weights = torch.load('/research/cbim/vast/yg397/ConvFormer/ConvFormer/initmodel/model_swinvit.pt')
                model.load_from(weights=weights)

            return model
        elif args.model == 'nnformer':
            from .dim3 import nnFormer
            model = nnFormer(args.window_size, input_channels=args.in_chan, num_classes=args.classes, deep_supervision=args.aux_loss)

            return model
    else:
        raise ValueError('Invalid dimension, should be \'2d\' or \'3d\'')