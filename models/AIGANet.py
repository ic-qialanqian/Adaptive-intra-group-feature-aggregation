import torch
from torch import nn
import torch.nn.functional as F
from models.vgg import VGG_Backbone
import numpy as np
import torch.optim as optim
from torchvision.models import vgg16
import fvcore.nn.weight_init as weight_init
from torch.nn import Conv2d, Parameter, Softmax

class group_comp(nn.Module):
    def __init__(self):
        super(group_comp,self).__init__()
        self.correlation = div()
        
    def forward(self,x):
        [S, _, H, W] = x.size()
        
        
        f_end = int(S / 2)
        s_end = int(S)
        
        x_group1 = x[0:f_end,:,:,:]
        x_group2 = x[f_end:s_end,:,:,:]
                
        x_mean1 = torch.mean(x_group1,dim=0)
        x_mean2 = torch.mean(x_group2,dim=0)
        
        group_mean1 = 0
        group_mean2 = 0
        
        x0 = x[0]
        cor = self.correlation(x0,x_group1)
        x_0 = x0 + x_mean1 * cor
        x_0 = x_0.unsqueeze(0)
        
        for i in range(1,f_end):
            x_single = x[i]
            cor = self.correlation(x_single,x_group1)
            x_enhance = x_single + x_mean1 * cor
            x_enhance = x_enhance.unsqueeze(0)
            x_0 = torch.cat([x_0,x_enhance],0)

             
        
        for i in range(f_end,s_end):
            x_single = x[i]
            cor = self.correlation(x_single,x_group2)
            x_enhance = x_single + x_mean2 * cor
            x_enhance = x_enhance.unsqueeze(0)
            x_0 = torch.cat([x_0,x_enhance],0)
        
        return x_0
    
            
        

    
    
    
class div(nn.Module):
    def __init__(self):
        super(div, self).__init__()
        
        
        
    def forward(self,x_single,x_group):
    
        
        x_group = torch.mean(x_group,dim=0)
        
        
        
        x_single = x_single.unsqueeze(0)
        x_group = x_group.unsqueeze(0)
        
        
        
        x_single = x_single.view(1,-1)
        x_group = x_group.view(1,-1)
        
        
        
        
        x_single = F.normalize(x_single,p=2,dim=1)
        x_group = F.normalize(x_group,p=2,dim=1)
        
        x_single2 = x_single.view(-1)
        x_group2 = x_group.view(-1)
        
        
        feat_vec = torch.cat([x_single,x_group],0)
        feat_vec_T = torch.transpose(feat_vec,0,1)
        
        
        
        kernel_mat = torch.matmul(feat_vec,feat_vec_T)
        
        div = torch.det(kernel_mat)
        
                
        return div
        
        
    

class resattetnion(nn.Module):
    def __init__(self, in_channel=64):
        super(resattetnion, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mask = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.globalinfo = nn.AdaptiveAvgPool2d(12)
        self.attentioninfo = nn.AdaptiveMaxPool2d(12)
        
        #self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        x = self.enlayer(x)
        
        [B, _, H, W] = x.size()
   
        mean_info = self.globalinfo(x)
        max_info = self.attentioninfo(x)
        
        lambda_ = 1
        
        y = mean_info + lambda_ * max_info
        
        y = F.interpolate(
            y, size=(H, W), mode='bilinear', align_corners=True)
        
        return x
        
        
class EnLayer(nn.Module):
    def __init__(self, in_channel=64):
        super(EnLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.enlayer(x)
        return x


class LatLayer(nn.Module):
    def __init__(self, in_channel):
        super(LatLayer, self).__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.convlayer(x)
        return x


class DSLayer(nn.Module):
    def __init__(self, in_channel=64):
        super(DSLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.predlayer = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())

    def forward(self, x):
        x = self.enlayer(x)
        x = self.predlayer(x)
        return x

class half_DSLayer(nn.Module):
    def __init__(self, in_channel=512):
        super(half_DSLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, int(in_channel/4), kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.Conv2d(int(in_channel/2), int(in_channel/4), kernel_size=3, stride=1, padding=1),
            #nn.ReLU(inplace=True),
        )
        self.predlayer = nn.Sequential(
            nn.Conv2d(int(in_channel/4), 1, kernel_size=1, stride=1, padding=0)) #, nn.Sigmoid())

    def forward(self, x):
        x = self.enlayer(x)
        x = self.predlayer(x)
        return x


class AllAttLayer(nn.Module):
    def __init__(self, input_channels=512):

        super(AllAttLayer, self).__init__()
        self.query_transform = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0) 
        self.key_transform = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0) 

        self.scale = 1.0 / (input_channels ** 0.5)

        self.conv6 = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0) 

        for layer in [self.query_transform, self.key_transform, self.conv6]:
            weight_init.c2_msra_fill(layer)

    def forward(self, x5):
        # x: B,C,H,W
        # x_query: B,C,HW
        B, C, H5, W5 = x5.size()

        x_query = self.query_transform(x5).view(B, C, -1)

        # x_query: B,HW,C
        x_query = torch.transpose(x_query, 1, 2).contiguous().view(-1, C) # BHW, C
        # x_key: B,C,HW
        x_key = self.key_transform(x5).view(B, C, -1)

        x_key = torch.transpose(x_key, 0, 1).contiguous().view(C, -1) # C, BHW

        # W = Q^T K: B,HW,HW
        x_w = torch.matmul(x_query, x_key) #* self.scale # BHW, BHW
        x_w = x_w.view(B*H5*W5, B, H5*W5)
        x_w = torch.max(x_w, -1).values # BHW, B
        x_w = x_w.mean(-1)
        #x_w = torch.mean(x_w, -1).values # BHW
        x_w = x_w.view(B, -1) * self.scale # B, HW
        x_w = F.softmax(x_w, dim=-1) # B, HW
        x_w = x_w.view(B, H5, W5).unsqueeze(1) # B, 1, H, W
 
        x5 = x5 * x_w
        x5 = self.conv6(x5)

        return x5

class CoAttLayer(nn.Module):
    def __init__(self, input_channels=512):

        super(CoAttLayer, self).__init__()

        self.all_attention = AllAttLayer(input_channels)
        self.conv_output = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0) 
        self.conv_transform = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0) 
        self.fc_transform = nn.Linear(input_channels, input_channels)

        for layer in [self.conv_output, self.conv_transform, self.fc_transform]:
            weight_init.c2_msra_fill(layer)
    
    def forward(self, x5):
        if self.training:
            f_begin = 0
            f_end = int(x5.shape[0] / 2)
            s_begin = f_end
            s_end = int(x5.shape[0])

            x5_1 = x5[f_begin: f_end]
            x5_2 = x5[s_begin: s_end]

            x5_new_1 = self.all_attention(x5_1)
            x5_new_2 = self.all_attention(x5_2)

            x5_1_proto = torch.mean(x5_new_1, (0, 2, 3), True).view(1, -1)
            x5_1_proto = x5_1_proto.unsqueeze(-1).unsqueeze(-1) # 1, C, 1, 1

            x5_2_proto = torch.mean(x5_new_2, (0, 2, 3), True).view(1, -1)
            x5_2_proto = x5_2_proto.unsqueeze(-1).unsqueeze(-1) # 1, C, 1, 1

            x5_11 = x5_1 * x5_1_proto
            x5_22 = x5_2 * x5_2_proto
            weighted_x5 = torch.cat([x5_11, x5_22], dim=0)

            x5_12 = x5_1 * x5_2_proto
            x5_21 = x5_2 * x5_1_proto
            neg_x5 = torch.cat([x5_12, x5_21], dim=0)
        else:

            x5_new = self.all_attention(x5)
            x5_proto = torch.mean(x5_new, (0, 2, 3), True).view(1, -1)
            x5_proto = x5_proto.unsqueeze(-1).unsqueeze(-1) # 1, C, 1, 1

            weighted_x5 = x5 * x5_proto #* cweight
            neg_x5 = None
        return weighted_x5, neg_x5


class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:
        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if x.numel() == 0 and self.training:
            # https://github.com/pytorch/pytorch/issues/12013
            assert not isinstance(
                self.norm, torch.nn.SyncBatchNorm
            ), "SyncBatchNorm does not support empty inputs!"

        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class GINet(nn.Module):
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, mode='train'):
        super(GINet, self).__init__()
        self.gradients = None
        self.backbone = VGG_Backbone()
        self.mode = mode

        self.toplayer = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))

        self.latlayer4 = LatLayer(in_channel=512)
        self.latlayer3 = LatLayer(in_channel=256)
        self.latlayer2 = LatLayer(in_channel=128)
        self.latlayer1 = LatLayer(in_channel=64)

        self.enlayer4 = EnLayer()
        self.enlayer3 = EnLayer()
        self.enlayer2 = EnLayer()
        self.enlayer1 = EnLayer()

        self.dslayer4 = DSLayer()
        self.dslayer3 = DSLayer()
        self.dslayer2 = DSLayer()
        self.dslayer1 = DSLayer()
        
        
        self.group_comp = group_comp()
        

    def set_mode(self, mode):
        self.mode = mode

    def _upsample_add(self, x, y):
        [_, _, H, W] = y.size()
        return F.interpolate(
            x, size=(H, W), mode='bilinear', align_corners=True) + y

    def _fg_att(self, feat, pred):
        [_, _, H, W] = feat.size()
        pred = F.interpolate(pred,
                             size=(H, W),
                             mode='bilinear',
                             align_corners=True)
        return feat * pred

    def forward(self, x):
        if self.mode == 'train':
            preds = self._forward(x)
        else:
            with torch.no_grad():
                preds = self._forward(x)

        return preds

    def _forward(self, x):
        [S, _, H, W] = x.size()
        x1 = self.backbone.conv1(x)
        x2 = self.backbone.conv2(x1)
        x3 = self.backbone.conv3(x2)
        x4 = self.backbone.conv4(x3)
        x5 = self.backbone.conv5(x4)
        
        x_0= self.group_comp(x5)
        
        
         
        cam = torch.mean(x_0, dim=1).unsqueeze(1)
        cam = cam.sigmoid()
            
        ########## Up-Sample ##########
        preds = []
        
       
        p5 = self.toplayer(x_0)
        _pred = cam
        
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))
        
        
        p4 = self._upsample_add(p5, self.latlayer4(x4)) 
        p4 = self.enlayer4(p4)
        p4 = self.group_comp(p4)
        
        
        
        _pred = self.dslayer4(p4)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))
        
        p3 = self._upsample_add(p4, self.latlayer3(x3)) 
        p3 = self.enlayer3(p3)
        p3 = self.group_comp(p3)
        
        
        _pred = self.dslayer3(p3)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))
        
        p2 = self._upsample_add(p3, self.latlayer2(x2)) 
        p2 = self.enlayer2(p2)
        
        _pred = self.dslayer2(p2)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))
        
        p1 = self._upsample_add(p2, self.latlayer1(x1)) 
        p1 = self.enlayer1(p1)

        
        _pred = self.dslayer1(p1)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))
        

       

        if self.training:
            return preds
        else:
            return preds


class AIGANet(nn.Module):
    def __init__(self, mode='train'):
        super(AIGANet, self).__init__()
        self.co_classifier = vgg16(pretrained=True).eval()
        self.ginet = GINet()
        self.mode = mode

    def set_mode(self, mode):
        self.mode = mode
        self.ginet.set_mode(self.mode)

    def forward(self, x):
        ########## Co-SOD ############
        preds = self.ginet(x)

        return preds
