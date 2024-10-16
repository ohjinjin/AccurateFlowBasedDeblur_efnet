'''
EFNet
@inproceedings{sun2022event,
      author = {Sun, Lei and Sakaridis, Christos and Liang, Jingyun and Jiang, Qi and Yang, Kailun and Sun, Peng and Ye, Yaozu and Wang, Kaiwei and Van Gool, Luc},
      title = {Event-Based Fusion for Motion Deblurring with Cross-modal Attention},
      booktitle = {European Conference on Computer Vision (ECCV)},
      year = 2022
      }
'''

import torch
import torch.nn as nn
import math
from basicsr.models.archs.arch_util import EventImage_ChannelAttentionTransformerBlock, TransformerBlock
from torch.nn import functional as F

def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

## Supervised Attention Module
## https://github.com/swz30/MPRNet
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img

##########################################################################
##---------- Prompt Gen Module -----------------------
## https://github.com/va1shn9v/PromptIR/blob/main/net/model.py
class PromptGenBlock(nn.Module):
    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192):
#         super(PromptGenBlock,self).__init__()
#         self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
#         self.linear_layer = nn.Linear(lin_dim,prompt_len)
#         self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
        super(PromptGenBlock,self).__init__()
        self.N = prompt_len
#         self.prompt_param = nn.Parameter(torch.rand(1,self.N,prompt_dim,prompt_size,prompt_size))
        # N개의 서로 다른 커널 크기를 가지는 Convolution Layer 정의
        self.convs = nn.ModuleList([
            nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, padding=1, bias=False),  # 3x3 커널
#             nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.Conv2d(prompt_dim, prompt_dim, kernel_size=5, padding=2, bias=False),  # 5x5 커널
#             nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.Conv2d(prompt_dim, prompt_dim, kernel_size=7, padding=3, bias=False),  # 7x7 커널
#             nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.Conv2d(prompt_dim, prompt_dim, kernel_size=9, padding=4, bias=False),  # 9x9 커널
#             nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, padding=5, dilation=5, bias=False)
            nn.Conv2d(prompt_dim, prompt_dim, kernel_size=11, padding=5, bias=False)  # 11x11 커널
        ])
        self.linear_layer = nn.Linear(lin_dim,self.N*lin_dim)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)


    def forward(self,x, motion):
#         B,C,H,W = x.shape
#         emb = x.mean(dim=(-2,-1))
#         prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
#         prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
#         prompt = torch.sum(prompt,dim=1)
#         prompt = F.interpolate(prompt,(H,W),mode="bilinear")
#         prompt = self.conv3x3(prompt)
#         cHECK JNJIN::: torch.Size([1, 256, 64, 64]) torch.Size([1, 256, 64, 64])
# CHECK JINJIN ::: C: 256 N: 5 prompt_weights: torch.Size([1, 256, 5])
# CHECK JINJIN::: prompt param: torch.Size([1, 5, 256, 64, 64])
# CHECK JINJIN::: prompt: torch.Size([1, 256, 64, 64])

#         print("cHECK JNJIN:::", x.shape, motion.shape)
        B,C,H,W = x.shape
        emb = motion.mean(dim=(-2,-1))  # GAP(G_l)
        prompt_weights = F.softmax(self.linear_layer(emb).view(B, C, self.N),dim=-1) # (B, C, N)
        # 각 Convolution Layer를 적용하고, 그 결과를 리스트로 저장
        conv_outputs = []
        for conv in self.convs:
            out = conv(x)  # Convolution 적용
            conv_outputs.append(out.unsqueeze(1))  # N축을 추가하여 (B, 1, C, H, W) 형태로

        # 각 Convolution 레이어의 출력을 N축으로 concat
        prompt_param = torch.cat(conv_outputs, dim=1)  # (B, N, C, H, W)
#         print("CHECK JINJIN ::: C:", C, "N:", self.N, "prompt_weights:", prompt_weights.shape)
#         print("CHECK JINJIN::: prompt param:", self.prompt_param.shape)
        # Original prompt_param: (1, N, C, H_p, W_p)
        # Permute to (1, C, N, H, W) to align with (B, C, N)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1) * prompt_param.permute(0, 2, 1, 3, 4)
        prompt = torch.sum(prompt,dim=2)
#         print("CHECK JINJIN::: prompt:", prompt.shape)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt

class EFNet(nn.Module):
    def __init__(self, in_chn=3, ev_chn=6, fl_chn=3, wf=64, depth=3, fuse_before_downsample=True, relu_slope=0.2, num_heads=[1,2,4]):
        super(EFNet, self).__init__()

        self.depth = depth
        self.fuse_before_downsample = fuse_before_downsample
        self.num_heads = num_heads
        self.down_path_1 = nn.ModuleList()
        self.down_path_2 = nn.ModuleList()
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.conv_02 = nn.Conv2d(in_chn, wf, 3, 1, 1)

        # event
        self.down_path_ev = nn.ModuleList()
        self.conv_ev1 = nn.Conv2d(ev_chn, wf, 3, 1, 1)
        # flow
        self.down_path_fl = nn.ModuleList()
        self.conv_fl1 = nn.Conv2d(fl_chn, wf, 3, 1, 1)

        prev_channels = self.get_input_chn(wf)
        for i in range(depth):
            downsample = True if (i+1) < depth else False 

            self.down_path_1.append(UNetConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, num_heads=self.num_heads[i]))
            self.down_path_2.append(UNetConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, use_emgc=downsample))
            # ev encoder, f1 encoder
            if i < self.depth:
                self.down_path_ev.append(UNetEVConvBlock(prev_channels, (2**i) * wf, downsample , relu_slope))
                self.down_path_fl.append(UNetEVConvBlock(prev_channels, (2**i) * wf, downsample , relu_slope))

            prev_channels = (2**i) * wf

        self.up_path_1 = nn.ModuleList()
        self.up_path_2 = nn.ModuleList()
        self.skip_conv_1 = nn.ModuleList()
        self.skip_conv_1_motion = nn.ModuleList()
        self.skip_conv_2 = nn.ModuleList()
        self.skip_conv_2_motion = nn.ModuleList()
        for i in reversed(range(depth - 1)):
#             self.up_path_1.append(PromptGenBlock(prompt_dim=64,prompt_len=5,prompt_size = 64,lin_dim = 96)) ### prompt3-1
#             self.up_path_1.append(TransformerBlock(dim=int(dim*2**2) + 512, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)) ### noise_level3-1
#             self.up_path_1.append(nn.Conv2d(int(dim*2**2)+512,int(dim*2**2),kernel_size=1,bias=bias)) ### reduce_noise_level3-1
#             self.up_path_1.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope)) # up4_3, concat, reduce_chan_level3-1
#             self.up_path_1.append(nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])) ### decoder_level3-1
            self.up_path_1.append(CustomUpBlock(prev_channels, (2**i)*wf, relu_slope, num_heads=self.num_heads[i], prompt_size=int(prev_channels/4)))

            self.up_path_2.append(CustomUpBlock(prev_channels, (2**i)*wf, relu_slope, num_heads=self.num_heads[i], prompt_size=int(prev_channels/4)))
            self.skip_conv_1.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            self.skip_conv_1_motion.append(nn.Conv2d(prev_channels, prev_channels, 3, 1, 1))
            self.skip_conv_2.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            self.skip_conv_2_motion.append(nn.Conv2d(prev_channels, prev_channels, 3, 1, 1))
            prev_channels = (2**i)*wf
        self.sam12 = SAM(prev_channels)

        self.cat12 = nn.Conv2d(prev_channels*2, prev_channels, 1, 1, 0)
        self.last = conv3x3(prev_channels, in_chn, bias=True)

    def forward(self, x, event, flow, mask=None):
        image = x
#         print("CHECK JINJIN1==== x: ", x.shape, "event: ", event.shape, "flow: ", flow.shape, "mask: ", mask.shape)
# CHECK JINJIN1==== x:  torch.Size([8, 3, 256, 256]) event:  torch.Size([8, 6, 256, 256]) flow:  torch.Size([8, 3, 256, 256]) mask:  torch.Size([8, 1, 256, 256])
        ev = []
        #EVencoder
        e1 = self.conv_ev1(event)
#         print("JINJIN2==== e1.shape", e1.shape)
# JINJIN2==== e1.shape torch.Size([8, 64, 256, 256]) c h w
        for i, down in enumerate(self.down_path_ev):
            if i < self.depth-1:
                e1, e1_up = down(e1, self.fuse_before_downsample)
#                 print("JINJIN3==== e1.shape", e1.shape, "e1_up.shape", e1_up.shape)
# JINJIN3==== e1.shape torch.Size([8, 64, 128, 128]) c h/2 w/2
# e1_up.shape torch.Size([8, 64, 256, 256]) c h w
# JINJIN3==== e1.shape torch.Size([8, 128, 64, 64]) 2c h/4 w/4
# e1_up.shape torch.Size([8, 128, 128, 128]) 2c h/2 w/2

                if self.fuse_before_downsample:
                    ev.append(e1_up)
                else:
                    ev.append(e1)
            else:
                e1 = down(e1, self.fuse_before_downsample)
#                 print("JINJIN4==== e1.shape", e1.shape)
# JINJIN4==== e1.shape torch.Size([8, 256, 64, 64]) 4c h/4 w/4
                ev.append(e1)

        fl = []
        #FLencoder
        f1 = self.conv_fl1(flow)
#         print("JINJIN2==== f1.shape", f1.shape)
# JINJIN2==== f1.shape torch.Size([8, 64, 256, 256]) c h w
        for i , down in enumerate(self.down_path_fl):
            if i < self.depth-1:
                f1, f1_up = down(f1, self.fuse_before_downsample)
#                 print("JINJIN3==== f1.shape", f1.shape, "f1_up.shape", f1_up.shape)
# JINJIN3==== f1.shape torch.Size([8, 64, 128, 128]) c h/2 w/2
# f1_up.shape torch.Size([8, 64, 256, 256]) c h w
# JINJIN3==== f1.shape torch.Size([8, 128, 64, 64]) 2c h/4 w/4
# f1_up.shape torch.Size([8, 128, 128, 128]) 2c h/2 w/2
                if self.fuse_before_downsample:
                    fl.append(f1_up)
                else:
                    fl.append(f1)
            else:
                f1 = down(f1, self.fuse_before_downsample)
#                 print("JINJIN4==== f1.shape", f1.shape)
# JINJIN4==== f1.shape torch.Size([8, 256, 64, 64]) 4c h/4 w/4
                fl.append(f1)

        #stage 1
        x1 = self.conv_01(image)
#         print("JINJIN2==== x1.shape", x1.shape)
# JINJIN2==== x1.shape torch.Size([8, 64, 256, 256]) c h w
        encs = []
        decs = []
        masks = []
        for i, down in enumerate(self.down_path_1):
            if (i+1) < self.depth:
                x1, x1_up = down(x1, event_filter=ev[i], merge_before_downsample=self.fuse_before_downsample)
#                 print("JINJIN3==== x1.shape", x1.shape, "x1_up.shape", x1_up.shape)
# JINJIN3==== x1.shape torch.Size([8, 64, 128, 128]) c h/2 w/2
# x1_up.shape torch.Size([8, 64, 256, 256]) c h w
# JINJIN3==== x1.shape torch.Size([8, 128, 64, 64]) 2c h/4 w/4
# x1_up.shape torch.Size([8, 128, 128, 128]) 2c h/2 w/2
                encs.append(x1_up)

                if mask is not None:
                    masks.append(F.interpolate(mask, scale_factor = 0.5**i))

            else:
                x1 = down(x1, event_filter=ev[i], merge_before_downsample=self.fuse_before_downsample)
#                 print("JINJIN4==== x1.shape", x1.shape)
# JINJIN4==== x1.shape torch.Size([8, 256, 64, 64]) 4c h/4 w/4

        for i, up in enumerate(self.up_path_1):
            x1 = up(x1, self.skip_conv_1[i](encs[-i-1]), self.skip_conv_1_motion[i](fl[-i-1]))
#             print("JINJIN7==== x1.shape", x1.shape)
# JINJIN7==== x1.shape torch.Size([8, 128, 128, 128]) 2c h/2 w/2
# JINJIN7==== x1.shape torch.Size([8, 64, 256, 256]) c h w
            decs.append(x1)
#             if i == 0:
#                 p2 = self.prompt2(x1)
#                 x1 = torch.cat([x1, p2], 1)
# #                 x1 = self.noise_level2(x1)
# #                 x1 = self.reduce_noise_level2(x1)
#             if i == 1:
#                 p1 = self.prompt1(x1)
#                 x1 = torch.cat([x1, p1], 1)
#                 x1 = self.noise_level1(x1)
#                 x1 = self.reduce_noise_level1(x1)
        sam_feature, out_1 = self.sam12(x1, image)
#         print("JINJIN5==== sam_feature.shape", sam_feature.shape, "out_1.shape", out_1.shape)
# JINJIN5==== sam_feature.shape torch.Size([8, 64, 256, 256]) c h w
# out_1.shape torch.Size([8, 3, 256, 256])
        #stage 2
        x2 = self.conv_02(image)
#         print("JINJIN2==== x2.shape", x2.shape)
# JINJIN2==== x2.shape torch.Size([8, 64, 256, 256]) c h w
        x2 = self.cat12(torch.cat([x2, sam_feature], dim=1))
#         print("JINJIN6==== x2.shape", x2.shape)
# JINJIN6==== x2.shape torch.Size([8, 64, 256, 256]) c h w
        blocks = []
        for i, down in enumerate(self.down_path_2):
            if (i+1) < self.depth:
                if mask is not None:
                    x2, x2_up = down(x2, encs[i], decs[-i-1], mask=masks[i])
#                     print("JINJIN3==== x2.shape", x2.shape, "x2_up.shape", x2_up.shape)
# JINJIN3==== x2.shape torch.Size([8, 64, 128, 128]) c h/2 w/2
# x2_up.shape torch.Size([8, 64, 256, 256]) c h w
# JINJIN3==== x2.shape torch.Size([8, 128, 64, 64]) 2c h/4 w/4
# x2_up.shape torch.Size([8, 128, 128, 128]) 2c h/2 w/2
                else:
                    x2, x2_up = down(x2, encs[i], decs[-i-1])
#                     print("JINJIN4==== x2.shape", x2.shape, "x2_up.shape", x2_up.shape)
                blocks.append(x2_up)
            else:
                x2 = down(x2)
#                 print("JINJIN3==== x2.shape", x2.shape)
# JINJIN3==== x2.shape torch.Size([8, 256, 64, 64]) 4c h/4 w/4

        for i, up in enumerate(self.up_path_2):
            x2 = up(x2, self.skip_conv_2[i](blocks[-i-1]), self.skip_conv_2_motion[i](fl[-i-1]))
#             print("JINJIN7==== x2.shape", x2.shape)
# JINJIN7==== x2.shape torch.Size([8, 128, 128, 128]) 2c h/2 w/2
# JINJIN7==== x2.shape torch.Size([8, 64, 256, 256]) c h w

        out_2 = self.last(x2)
#         print("JINJIN8==== out_2.shape", out_2.shape)
# JINJIN8==== out_2.shape torch.Size([8, 3, 256, 256])
        out_2 = out_2 + image

        return [out_1, out_2]

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_emgc=False, num_heads=None): # cat
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_emgc = use_emgc
        self.num_heads = num_heads

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)        

        if downsample and use_emgc:
            self.emgc_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_enc_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

        if self.num_heads is not None:
            self.image_event_transformer = EventImage_ChannelAttentionTransformerBlock(out_size, num_heads=self.num_heads, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')


    def forward(self, x, enc=None, dec=None, mask=None, event_filter=None, merge_before_downsample=True):
        out = self.conv_1(x)

        out_conv1 = self.relu_1(out)
        out_conv2 = self.relu_2(self.conv_2(out_conv1))

        out = out_conv2 + self.identity(x)

        if enc is not None and dec is not None and mask is not None:
            assert self.use_emgc
            out_enc = self.emgc_enc(enc) + self.emgc_enc_mask((1-mask)*enc)
            out_dec = self.emgc_dec(dec) + self.emgc_dec_mask(mask*dec)
            out = out + out_enc + out_dec        

        if event_filter is not None and merge_before_downsample:
            # b, c, h, w = out.shape
            out = self.image_event_transformer(out, event_filter)

        if self.downsample:
            out_down = self.downsample(out)
            if not merge_before_downsample: 
                out_down = self.image_event_transformer(out_down, event_filter)

            return out_down, out

        else:
            if merge_before_downsample:
                return out
            else:
                out = self.image_event_transformer(out, event_filter)


class UNetEVConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_emgc=False):
        super(UNetEVConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_emgc = use_emgc

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        self.conv_before_merge = nn.Conv2d(out_size, out_size , 1, 1, 0) 
        if downsample and use_emgc:
            self.emgc_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_enc_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, merge_before_downsample=True):
        out = self.conv_1(x)

        out_conv1 = self.relu_1(out)
        out_conv2 = self.relu_2(self.conv_2(out_conv1))

        out = out_conv2 + self.identity(x)

        if self.downsample:

            out_down = self.downsample(out)

            if not merge_before_downsample: 

                out_down = self.conv_before_merge(out_down)
            else : 
                out = self.conv_before_merge(out)
            return out_down, out

        else:

            out = self.conv_before_merge(out)
            return out


class UNetUpBlock(nn.Module):

    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out


class CustomUpBlock(nn.Module):  # in_size = 2*out_size
    def __init__(self, in_size, out_size, relu_slope, prompt_len=5, prompt_size=16, num_heads=None, ffn_expansion_factor=2.66, bias=False, num_blocks=[1, 4, 4], LayerNorm_type='WithBias'):
        super(CustomUpBlock, self).__init__()

        # PromptGenBlock: 프롬프트 생성 블록
        self.prompt = PromptGenBlock(prompt_dim=in_size, prompt_len=prompt_len, prompt_size=prompt_size, lin_dim=in_size)

        # TransformerBlock: 노이즈 레벨을 처리하는 Transformer 블록
        self.noise = TransformerBlock(dim=in_size*2, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

        # Conv2D: 노이즈를 줄이기 위한 Conv2D 레이어
        self.reduce_noise = nn.Conv2d(in_size*2, in_size, kernel_size=1, bias=bias)

        # UNetUpBlock: 업샘플링과 skip connection 결합
        self.up_cat_reducechan = UNetUpBlock(in_size, out_size, relu_slope)

        # Transformer 블록 시퀀스: 디코더 블록을 위한 Transformer 블록들
        self.decoder = nn.Sequential(
            *[TransformerBlock(dim=out_size, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[0])]
        )

    def forward(self, out_dec_prev_level, out_enc_curr_level, motion_dec_prev_level):
#         print("CHECK JINJIN out_dec_prev_level::::", out_dec_prev_level.shape, "out_enc_curr_level:::: ", out_enc_curr_level.shape)
        # CHECK JINJIN out_dec_prev_level:::: torch.Size([1, 256, 64, 64]) out_enc_curr_level::::  torch.Size([1, 128, 128, 128])

        # 1. PromptGenBlock 처리
        dec_curr_param = self.prompt(out_dec_prev_level, motion_dec_prev_level)  # equation (2)에서 F_l대신 motion feature인 G_l로 입력 바꿔줘야하고 함수정의자체에서도 5개의 앙상블링하게끔 마저 수정필요함.
#         print("CHEKC JININ 11:", dec_curr_param.shape)  # torch.Size([1, 256, 64, 64])
        # 2. TransformerBlock으로 노이즈 처리
        out_dec_prev_level = torch.cat([out_dec_prev_level, dec_curr_param], 1)
#         print("CHEKC JININ 2:", out_dec_prev_level.shape)  # torch.Size([1, 512, 64, 64])
        out_dec_prev_level = self.noise(out_dec_prev_level)
#         print("CHEKC JININ 3:", out_dec_prev_level.shape)# torch.Size([1, 512, 64, 64])

        # 3. Conv2D로 노이즈 줄이기
        out_dec_prev_level = self.reduce_noise(out_dec_prev_level)
#         print("CHEKC JININ 4:", out_dec_prev_level.shape)  # torch.Size([1, 256, 64, 64])
        # 4. 업샘플링 및 skip connection 결합
        inp_dec_curr_level = self.up_cat_reducechan(out_dec_prev_level, out_enc_curr_level)
#         print("CHEKC JININ 5:", inp_dec_curr_level.shape)  # torch.Size([1, 128, 128, 128])

        # 5. Transformer 블록들 적용 (디코더 처리)
        out_dec_curr_level = self.decoder(inp_dec_curr_level)
#         print("CHEKC JININ 6:", out_dec_curr_level.shape)  # torch.Size([1, 128, 128, 128])

        return out_dec_curr_level