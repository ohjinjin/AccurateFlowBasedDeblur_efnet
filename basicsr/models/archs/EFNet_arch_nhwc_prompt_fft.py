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
from basicsr.models.archs.arch_util import EventImage_ChannelAttentionTransformerBlock, FlowImage_ChannelAttentionTransformerBlock, FlowEvent_ChannelAttentionTransformerBlock, TransformerBlock
# from basicsr.models.archs.arch_util import FlowImage_ChannelAttentionTransformerBlock
# from basicsr.models.archs.arch_util import FlowEvent_ChannelAttentionTransformerBlock
from torch.nn import functional as F
import torch.fft

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
    def __init__(self,prompt_dim=128,prompt_len=3,prompt_size = 96,lin_dim = 192, relu_slope=0.2):
        super(PromptGenBlock,self).__init__()
        self.N = prompt_len
        self.prompt_dim = prompt_dim
        self.prompt_size = prompt_size
        
        
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        # N개의 서로 다른 커널 크기를 가지는 Convolution Layer 정의
#         self.convs = nn.ModuleList([
#             nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, padding=1, bias=False),  # 3x3 커널
#             nn.Conv2d(prompt_dim, prompt_dim, kernel_size=5, padding=2, bias=False),  # 5x5 커널
#             nn.Conv2d(prompt_dim, prompt_dim, kernel_size=7, padding=3, bias=False),  # 7x7 커널
#             nn.Conv2d(prompt_dim, prompt_dim, kernel_size=9, padding=4, bias=False),  # 9x9 커널
#             nn.Conv2d(prompt_dim, prompt_dim, kernel_size=11, padding=5, bias=False)  # 11x11 커널
#         ])
        self.identity_1_low = nn.Conv2d(prompt_dim, prompt_dim, 1, 1, 0)
        self.conv_1_low = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, padding=1, bias=True)
        self.relu_1_low = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2_low = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, padding=1, bias=True)
        self.relu_2_low = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_before_merge_1_low = nn.Conv2d(prompt_dim, prompt_dim , 1, 1, 0)
        
#         self.identity_2_low = nn.Conv2d(prompt_dim, prompt_dim, 1, 1, 0)
#         self.conv_3_low = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, padding=1, bias=True)
#         self.relu_3_low = nn.LeakyReLU(relu_slope, inplace=False)
#         self.conv_4_low = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, padding=1, bias=True)
#         self.relu_4_low = nn.LeakyReLU(relu_slope, inplace=False)
#         self.conv_before_merge_2_low = nn.Conv2d(prompt_dim, prompt_dim , 1, 1, 0)
        self.conv1x1_low = nn.Conv2d(in_channels=prompt_dim, out_channels=prompt_dim, kernel_size=1, stride=1, padding=0, bias=False)
        
        
        self.identity_1_middle = nn.Conv2d(prompt_dim, prompt_dim, 1, 1, 0)
        self.conv_1_middle = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, padding=1, bias=True)
        self.relu_1_middle = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2_middle = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, padding=1, bias=True)
        self.relu_2_middle = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_before_merge_1_middle = nn.Conv2d(prompt_dim, prompt_dim , 1, 1, 0)
        
#         self.identity_2_middle = nn.Conv2d(prompt_dim, prompt_dim, 1, 1, 0)
#         self.conv_3_middle = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, padding=1, bias=True)
#         self.relu_3_middle = nn.LeakyReLU(relu_slope, inplace=False)
#         self.conv_4_middle = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, padding=1, bias=True)
#         self.relu_4_middle = nn.LeakyReLU(relu_slope, inplace=False)
#         self.conv_before_merge_2_middle = nn.Conv2d(prompt_dim, prompt_dim , 1, 1, 0)
        self.conv1x1_middle = nn.Conv2d(in_channels=prompt_dim, out_channels=prompt_dim, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.identity_1_high = nn.Conv2d(prompt_dim, prompt_dim, 1, 1, 0)
        self.conv_1_high = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, padding=1, bias=True)
        self.relu_1_high = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2_high = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, padding=1, bias=True)
        self.relu_2_high = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_before_merge_1_high = nn.Conv2d(prompt_dim, prompt_dim , 1, 1, 0)
        
#         self.identity_2_high = nn.Conv2d(prompt_dim, prompt_dim, 1, 1, 0)
#         self.conv_3_high = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, padding=1, bias=True)
#         self.relu_3_high = nn.LeakyReLU(relu_slope, inplace=False)
#         self.conv_4_high = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, padding=1, bias=True)
#         self.relu_4_high = nn.LeakyReLU(relu_slope, inplace=False)
#         self.conv_before_merge_2_high = nn.Conv2d(prompt_dim, prompt_dim , 1, 1, 0)
        
        self.conv1x1_high = nn.Conv2d(in_channels=prompt_dim, out_channels=prompt_dim, kernel_size=1, stride=1, padding=0, bias=False)
#         self.linear_layer = nn.Linear(lin_dim,self.N*lin_dim)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)


    def forward(self, motion):
        B,C,H,W = motion.shape
#         emb = motion.mean(dim=(-2,-1))
#         prompt_weights = F.softmax(self.linear_layer(emb).view(B, C, self.N), dim=-1)
#         merged_in = torch.cat([x, motion], 1)
        radius_low = int(0.1 * (H + W) / 2)   # 10% of average dimension
        radius_high = int(0.4 * (H + W) / 2)  # 40% of average dimension
        # 1. Apply Fourier transform to get momentum domain representation
        tensor_fft = torch.fft.fft2(motion, dim=(-2, -1))
        tensor_fft_shifted = torch.fft.fftshift(tensor_fft, dim=(-2, -1))

        # 2. Define frequency masks
        center_h, center_w = H // 2, W // 2
        distance = torch.sqrt((torch.arange(H).unsqueeze(1) - center_h) ** 2 + 
                              (torch.arange(W).unsqueeze(0) - center_w) ** 2).to(motion.device)

        # Low, middle, and high frequency masks in frequency domain
        mask_low = (distance <= radius_low).float()
        mask_middle = ((distance > radius_low) & (distance <= radius_high)).float()
        mask_high = (distance > radius_high).float()
        # Expanding masks to match B, C, H, W dimensions
        mask_low = mask_low.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)
        mask_middle = mask_middle.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)
        mask_high = mask_high.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)
        # 3. Apply masks to frequency domain tensor
        tensor_low_freq = tensor_fft_shifted * mask_low
        tensor_middle_freq = tensor_fft_shifted * mask_middle
        tensor_high_freq = tensor_fft_shifted * mask_high
        # 4. Inverse Fourier transform to bring back to spatial domain
        tensor_low_spatial = torch.fft.ifftshift(tensor_low_freq, dim=(-2, -1))
        tensor_low_spatial = torch.fft.ifft2(tensor_low_spatial, dim=(-2, -1)).real

        tensor_middle_spatial = torch.fft.ifftshift(tensor_middle_freq, dim=(-2, -1))
        tensor_middle_spatial = torch.fft.ifft2(tensor_middle_spatial, dim=(-2, -1)).real

        tensor_high_spatial = torch.fft.ifftshift(tensor_high_freq, dim=(-2, -1))
        tensor_high_spatial = torch.fft.ifft2(tensor_high_spatial, dim=(-2, -1)).real
#         filtered_motion_list = [tensor_low_spatial, tensor_middle_spatial, tensor_high_spatial]
        
        merged_in = tensor_low_spatial#motion
        merged_out = self.conv_1_low(merged_in)        
        merged_out_conv1 = self.relu_1_low(merged_out)
        merged_out_conv2 = self.relu_2_low(self.conv_2_low(merged_out_conv1))
        merged_out = merged_out_conv2 + self.identity_1_low(merged_in)
        merged_out = self.conv_before_merge_1_low(merged_out)
#         merged_in = merged_out
#         merged_out = self.conv_3_low(merged_in)
#         merged_out_conv3 = self.relu_3_low(merged_out)
#         merged_out_conv4 = self.relu_4_low(self.conv_4_low(merged_out_conv3))
#         merged_out = merged_out_conv3 + self.identity_2_low(merged_in)
#         merged_out = self.conv_before_merge_2_low(merged_out)
        merged_out_low = self.conv1x1_low(merged_out).unsqueeze(1)
        
        merged_in = tensor_middle_spatial#motion
        merged_out = self.conv_1_middle(merged_in)        
        merged_out_conv1 = self.relu_1_middle(merged_out)
        merged_out_conv2 = self.relu_2_middle(self.conv_2_middle(merged_out_conv1))
        merged_out = merged_out_conv2 + self.identity_1_middle(merged_in)
        merged_out = self.conv_before_merge_1_middle(merged_out)
#         merged_in = merged_out
#         merged_out = self.conv_3_middle(merged_in)
#         merged_out_conv3 = self.relu_3_middle(merged_out)
#         merged_out_conv4 = self.relu_4_middle(self.conv_4_middle(merged_out_conv3))
#         merged_out = merged_out_conv3 + self.identity_2_middle(merged_in)
#         merged_out = self.conv_before_merge_2_middle(merged_out)
        merged_out_middle = self.conv1x1_middle(merged_out).unsqueeze(1)
        
        merged_in = tensor_high_spatial#motion
        merged_out = self.conv_1_high(merged_in)        
        merged_out_conv1 = self.relu_1_high(merged_out)
        merged_out_conv2 = self.relu_2_high(self.conv_2_high(merged_out_conv1))
        merged_out = merged_out_conv2 + self.identity_1_high(merged_in)
        merged_out = self.conv_before_merge_1_high(merged_out)
#         merged_in = merged_out
#         merged_out = self.conv_3_high(merged_in)
#         merged_out_conv3 = self.relu_3_high(merged_out)
#         merged_out_conv4 = self.relu_4_high(self.conv_4_high(merged_out_conv3))
#         merged_out = merged_out_conv3 + self.identity_2_high(merged_in)
#         merged_out = self.conv_before_merge_2_high(merged_out)
        merged_out_high = self.conv1x1_high(merged_out).unsqueeze(1)
#         print("CHCHCHCH:::::", merged_out_high.shape) # torch.Size([8, 256, 64, 64])
        merged_out = torch.cat([merged_out_low, merged_out_middle, merged_out_high], dim=1)  # (B, 3, C, H, W)
        
        prompt_weights = F.softmax(merged_out, dim=1)
        
#         conv_outputs = []
#         for conv in self.convs:
#             out = conv(x)
#             conv_outputs.append(out.unsqueeze(1))
#         prompt_param = torch.cat(conv_outputs, dim=1)
#         print("JINCjin checkkkk 11111111:", self.prompt_size, self.prompt_dim)
#         print("jinjin CHECKKKK::::", torch.cat(conv_outputs, dim=1).shape, self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1).shape, prompt_weights.shape)
#         prompt_param = F.interpolate(self.prompt_param,(H,W),mode="bilinear")
        
        B_param, N_param, C_param, H_param, W_param = self.prompt_param.shape
        prompt_param = self.prompt_param.view(B_param * N_param, C_param, H_param, W_param)
        prompt_param = F.interpolate(prompt_param, size=(H, W), mode="bilinear")
        # 원래 크기인 (B, N, C, H, W)로 복원
        prompt_param = prompt_param.view(B_param, N_param, C_param, H, W)
        prompt_param = prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
#         prompt_param = F.interpolate(prompt_param,(H,W),mode="bilinear")
#         prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1) * prompt_param.permute(0, 2, 1, 3, 4)
#         print("CHECKKK,,,,,,,,,,,,,,,,,,LL::", B,C,H,W,prompt_param.shape, prompt_weights.shape)
        prompt = prompt_param * prompt_weights
#         prompt = torch.sum(prompt,dim=2)
        prompt = torch.sum(prompt,dim=1)
#         print("JINJINJIN:::::", prompt.shape)
        #prompt = F.interpolate(prompt,(H,W),mode="bilinear")
#         print("JINJINJIN:2222::::", prompt.shape)
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
            # ev encoder, fl encoder
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
#             self.up_path_1.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            self.up_path_1.append(CustomUpBlock(prev_channels, (2**i)*wf, relu_slope, num_heads=self.num_heads[i], prompt_size=int((2**(depth-1-i))*wf/2)))
            self.up_path_2.append(CustomUpBlock(prev_channels, (2**i)*wf, relu_slope, num_heads=self.num_heads[i], prompt_size=int((2**(depth-1-i))*wf/2)))
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

        ev = []
        #EVencoder
        e1 = self.conv_ev1(event)
        for i, down in enumerate(self.down_path_ev):
            if i < self.depth-1:
                e1, e1_up = down(e1, self.fuse_before_downsample)
                if self.fuse_before_downsample:
                    ev.append(e1_up)
                else:
                    ev.append(e1)
            else:
                e1 = down(e1, self.fuse_before_downsample)
                ev.append(e1)
        
        fl = []
        #EVencoder
        f1 = self.conv_fl1(flow)
        for i, down in enumerate(self.down_path_fl):
            if i < self.depth-1:
                f1, f1_up = down(f1, self.fuse_before_downsample)
                if self.fuse_before_downsample:
                    fl.append(f1_up)
                else:
                    fl.append(f1)
            else:
                f1 = down(f1, self.fuse_before_downsample)
                fl.append(f1)

        #stage 1
        x1 = self.conv_01(image)
        encs = []
        decs = []
        masks = []
        for i, down in enumerate(self.down_path_1):
            if (i+1) < self.depth:

                x1, x1_up = down(x1, event_filter=ev[i], flow_filter=fl[i], merge_before_downsample=self.fuse_before_downsample)
                encs.append(x1_up)

                if mask is not None:
                    masks.append(F.interpolate(mask, scale_factor = 0.5**i))
            
            else:
                x1 = down(x1, event_filter=ev[i], flow_filter=fl[i], merge_before_downsample=self.fuse_before_downsample)


        for i, up in enumerate(self.up_path_1):
#             print("CHECKKKK:::::", x1.shape, self.skip_conv_1[i](encs[-i-1]).shape, self.skip_conv_1_motion[i](fl[-i-1]).shape)
            x1 = up(x1, self.skip_conv_1[i](encs[-i-1]), self.skip_conv_1_motion[i](fl[-i-1]))
            decs.append(x1)
        sam_feature, out_1 = self.sam12(x1, image)

        #stage 2
        x2 = self.conv_02(image)
        x2 = self.cat12(torch.cat([x2, sam_feature], dim=1))
        blocks = []
        for i, down in enumerate(self.down_path_2):
            if (i+1) < self.depth:
                if mask is not None:
                    x2, x2_up = down(x2, encs[i], decs[-i-1], mask=masks[i])
                else:
                    x2, x2_up = down(x2, encs[i], decs[-i-1])
                blocks.append(x2_up)
            else:
                x2 = down(x2)

        for i, up in enumerate(self.up_path_2):
            x2 = up(x2, self.skip_conv_2[i](blocks[-i-1]), self.skip_conv_2_motion[i](fl[-i-1]))

        out_2 = self.last(x2)
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
            self.flow_event_transformer = FlowEvent_ChannelAttentionTransformerBlock(out_size, num_heads=self.num_heads, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')
            self.image_flow_transformer = FlowImage_ChannelAttentionTransformerBlock(out_size, num_heads=self.num_heads, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')
        

    def forward(self, x, enc=None, dec=None, mask=None, event_filter=None, flow_filter=None, merge_before_downsample=True):
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
        if flow_filter is not None and merge_before_downsample:
            out = self.image_flow_transformer(out, flow_filter)
            out = self.flow_event_transformer(out, event_filter, flow_filter)
             
        if self.downsample:
            out_down = self.downsample(out)
            if not merge_before_downsample: 
                out_down = self.image_event_transformer(out_down, event_filter)
                out_down = self.image_flow_transformer(out_down, flow_filter)
                out_down = self.flow_event_transformer(out_down, event_filter, flow_filter)

            return out_down, out

        else:
            if merge_before_downsample:
                return out
            else:
                out = self.image_event_transformer(out, event_filter)
                out = self.image_flow_transformer(out, flow_filter)
                out = self.flow_event_transformer(out, event_filter, flow_filter)


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
    def __init__(self, in_size, out_size, relu_slope, prompt_len=3, prompt_size=16, num_heads=None, ffn_expansion_factor=2.66, bias=False, num_blocks=[1, 4, 4], LayerNorm_type='WithBias'):
        super(CustomUpBlock, self).__init__()

        # PromptGenBlock: 프롬프트 생성 블록
        self.prompt = PromptGenBlock(prompt_dim=in_size, prompt_len=prompt_len, prompt_size=prompt_size, lin_dim=in_size, relu_slope=relu_slope)

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
        dec_curr_param = self.prompt(motion_dec_prev_level)  # equation (2)에서 F_l대신 motion feature인 G_l로 입력 바꿔줘야하고 함수정의자체에서도 5개의 앙상블링하게끔 마저 수정필요함.
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
    
    
if __name__ == "__main__":
    pass
