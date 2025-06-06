import torch
import torch.nn as nn
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID
from .IED import IED
from .Myloss import L_color, L_TV, L_spa, L_exp_v2, L_denoise

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.reduce_feat_dim = cfg.MODEL.REDUCE_FEAT_DIM
        self.feat_dim = cfg.MODEL.FEAT_DIM
        self.dropout_rate = cfg.MODEL.DROPOUT_RATE
        self.mean=torch.tensor(cfg.INPUT.PIXEL_MEAN).to("cuda").view(1, 3, 1, 1)
        self.std=torch.tensor(cfg.INPUT.PIXEL_STD).to("cuda").view(1, 3, 1, 1)
        self.enhance_mean=torch.tensor(cfg.INPUT.ENHANCE_PIXEL_MEAN).to("cuda").view(1, 3, 1, 1)
        self.enhance_std=torch.tensor(cfg.INPUT.ENHANCE_PIXEL_STD).to("cuda").view(1, 3, 1, 1)


        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0
        
        self.L_color = L_color()
        self.L_exp_v2 = L_exp_v2(16,0.5)
        self.L_TV = L_TV()
        self.L_spa = L_spa()
        self.L_denoise = L_denoise()
        self.ied=IED()
        self.gray=torch.tensor([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1).cuda()

        self.base_i = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH, drop_rate= cfg.MODEL.DROP_OUT,attn_drop_rate=cfg.MODEL.ATT_DROP_RATE, gem_pool=cfg.MODEL.GEM_POOLING, stem_conv=cfg.MODEL.STEM_CONV)
        self.base_d = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH, drop_rate= cfg.MODEL.DROP_OUT,attn_drop_rate=cfg.MODEL.ATT_DROP_RATE, gem_pool=cfg.MODEL.GEM_POOLING, stem_conv=cfg.MODEL.STEM_CONV)
        self.in_planes = self.base_i.in_planes
        if pretrain_choice == 'imagenet':
            self.base_i.load_param(model_path,hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
            self.base_d.load_param(model_path,hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        self.classifier_i = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_i.apply(weights_init_classifier)
        self.classifier_d = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_d.apply(weights_init_classifier)
        self.classifier_c = nn.Linear(self.in_planes*2, self.num_classes, bias=False)
        self.classifier_c.apply(weights_init_classifier)

        self.bottleneck_i = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_i.bias.requires_grad_(False)
        self.bottleneck_i.apply(weights_init_kaiming)
        self.bottleneck_d = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_d.bias.requires_grad_(False)
        self.bottleneck_d.apply(weights_init_kaiming)
        self.bottleneck_c = nn.BatchNorm1d(self.in_planes*2)
        self.bottleneck_c.bias.requires_grad_(False)
        self.bottleneck_c.apply(weights_init_kaiming)
        
        self.dropout = nn.Dropout(self.dropout_rate)

        if pretrain_choice == 'self':
            self.load_param(model_path)

    def forward(self, x, label=None, cam_label= None, view_label=None):
        x=torch.clamp(x, min=0, max=1)

        en_img,en_img_d,A=self.ied(x)
                  
        x_i = (en_img_d - self.enhance_mean) / self.enhance_std
        x_d = (x - self.mean) / self.std
        
        global_feat_i = self.base_i(x_i, cam_label=cam_label, view_label=view_label)
        feat_i = self.bottleneck_i(global_feat_i)
        feat_cls_i = self.dropout(feat_i)

        global_feat_d = self.base_d(x_d, cam_label=cam_label, view_label=view_label)
        feat_d = self.bottleneck_d(global_feat_d)
        feat_cls_d = self.dropout(feat_d)

        global_feat_c = torch.cat((global_feat_d,global_feat_i),1)
        feat_c = self.bottleneck_c(global_feat_c)
        feat_cls_c = self.dropout(feat_c)

        if self.training:
            
            cls_score_i = self.classifier_i(feat_cls_i)
            cls_score_d = self.classifier_d(feat_cls_d)  
            cls_score_c = self.classifier_c(feat_cls_c)

            Loss_TV = 200*self.L_TV(A)
            loss_spa = torch.mean(self.L_spa(en_img_d, x))
            loss_col = 5*torch.mean(self.L_color(en_img_d))
            loss_exp = 5*torch.mean(self.L_exp_v2(en_img_d,x))          

            en_img_c = torch.sum(en_img_d * self.gray, dim=1, keepdim=True)
            grad_x=self.L_denoise(en_img_c, 'x')
            grad_y=self.L_denoise(en_img_c, 'y')
            loss_den = 10 * (torch.mean(grad_x * torch.exp(-10 * grad_x))+torch.mean(grad_y * torch.exp(-10 * grad_y)))

            loss_dce = Loss_TV +loss_spa + loss_col + loss_exp + loss_den

            return cls_score_i, global_feat_i, cls_score_d, global_feat_d, cls_score_c, global_feat_c, loss_dce
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat_i, feat_d, feat_c
            else:
                # print("Test with feature before BN")
                return global_feat_i, global_feat_d, global_feat_c

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location = 'cpu')
        for i in param_dict:
            try:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            except:
                continue
        print('Loading pretrained model from {}'.format(trained_path))


__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
}

def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
    print('===========building transformer===========')
    return model
