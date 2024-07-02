import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from torchvision.models.optical_flow import raft_large
import torchvision.transforms as T
# from torchvision.utils import save_image, flow_to_image
# import numpy as np
# import matplotlib.pyplot as plt

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')


class ImageEventRestorationModel(BaseModel):
    """Base Event-based deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageEventRestorationModel, self).__init__(opt)
        
        self.raft_model = raft_large(pretrained=True, progress=False).to(self.device)
        self.raft_model = self.raft_model.eval()
        
        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            self.pixel_type = train_opt['pixel_opt'].pop('type')
            # print('LOSS: pixel_type:{}'.format(self.pixel_type))
            cri_pix_cls = getattr(loss_module, self.pixel_type)

            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        optim_params_lowlr = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                if k.startswith('module.offsets') or k.startswith('module.dcns'):
                    optim_params_lowlr.append(v)
                else:
                    optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        ratio = 0.1

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}, {'params': optim_params_lowlr, 'lr': train_opt['optim_g']['lr'] * ratio}],
                                                **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}, {'params': optim_params_lowlr, 'lr': train_opt['optim_g']['lr'] * ratio}],
                                                **train_opt['optim_g'])

        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)
    
    def preprocess(self, batch):
        transforms = T.Compose(
            [
                T.ConvertImageDtype(torch.float32),
                T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
    #             T.Resize(size=(520, 960)),
            ]
        )
        batch = transforms(batch)
        return batch
    
    def feed_data(self, data):
#         image_1 = Image.open(image_path_1)
#         image_2 = Image.open(image_path_2)
#         tensor_image_1 = F.to_tensor(image_1)
#         tensor_image_2 = F.to_tensor(image_2)
#         print("JJJJ:::::", data['frame_prev'])
        img0 = self.preprocess(data['frame_prev']).to(self.device)
        img1 = self.preprocess(data['frame_gt']).to(self.device)
        img2 = self.preprocess(data['frame_next']).to(self.device)
#         print("CHECKKKK:::::::::::", img0.shape)
#         self.flow10 = self.raft_model(img1, img0)[-1]
#         self.flow12 = self.raft_model(img1, img2)[-1]
        self.flow02 = self.raft_model(img0, img2)[-1]
#         print("CEHCK JINJIN::::", self.flow10.shape, self.flow12.shape)
        
        self.lq = data['frame'].to(self.device)
#         lq_bgr = self.lq[:, [2, 1, 0], :, :]
#         img0_bgr = (img0[:, [2, 1, 0], :, :] + 1) / 2.0
#         img1_bgr = (img1[:, [2, 1, 0], :, :] + 1) / 2.0
#         img2_bgr = (img2[:, [2, 1, 0], :, :] + 1) / 2.0
        
#         # 각 이미지를 개별적으로 저장
#         import os

#         for i in range(lq_bgr.size(0)):
#             if not os.path.exists(f'/home/ohjinjin/{self.lq.device}_blur_image_{i}.png'):
#                 save_image(lq_bgr[i], f'/home/ohjinjin/{self.lq.device}_blur_image_{i}.png')
#         for i in range(img0_bgr.size(0)):
#             if not os.path.exists(f'/home/ohjinjin/{img0.device}_prev_image_{i}.png'):
#                 save_image(img0_bgr[i], f'/home/ohjinjin/{img0.device}_prev_image_{i}.png')
#         for i in range(img1_bgr.size(0)):
#             if not os.path.exists(f'/home/ohjinjin/{img1.device}_sharp_image_{i}.png'):
#                 save_image(img1_bgr[i], f'/home/ohjinjin/{img1.device}_sharp_image_{i}.png')
#         for i in range(img2_bgr.size(0)):
#             if not os.path.exists(f'/home/ohjinjin/{img2.device}_next_image_{i}.png'):
#                 save_image(img2_bgr[i], f'/home/ohjinjin/{img2.device}_next_image_{i}.png')
                
#         flow_image10 = flow_to_image(self.flow10)/255.0
#         flow_image12 = flow_to_image(self.flow12)/255.0
# #         flow_image10 = flow_image10_.to(torch.uint8)
# #         flow_image12 = flow_image12_.to(torch.uint8)
# #         print("CHCHCHCH::", img0.min())
# #         print("CHCHCHCH::", img0.max())
# #         print("CHCHCHCH::", self.lq.min())
# #         print("CHCHCHCH::", self.lq.max())
#         for i in range(flow_image10.size(0)):
#             if not os.path.exists(f'/home/ohjinjin/{self.flow10.device}_flow10_{i}.png'):
#                 save_image(flow_image10[i], f'/home/ohjinjin/{self.flow10.device}_flow10_{i}.png')
#         for i in range(flow_image12.size(0)):
#             if not os.path.exists(f'/home/ohjinjin/{self.flow12.device}_flow12_{i}.png'):
#                 save_image(flow_image12[i], f'/home/ohjinjin/{self.flow12.device}_flow12_{i}.png')
        

#         print("CHECK JINJIN1:::", type(data['frame_prev']), data['frame_prev'].shape, data['frame'].shape)
        # CHECK JINJIN1::: <class 'torch.Tensor'> torch.Size([4, 3, 256, 256]) torch.Size([4, 3, 256, 256])

        
#         self.flow_01 = data['frame_prev'],data['frame_gt']
#         self.flow_10 = data['frame_gt'],data['frame_next']
        self.voxel=data['voxel'].to(self.device)  # torch.Size([B, 6, 256, 256])
#         print("CHECK SIZE::::", self.voxel.size(0))
# #         print("CHECK::::::", torch.unique(self.voxel[:,0,:,:]))
# #         print("CHECK::::::", torch.unique(self.voxel[:,1,:,:]))
# #         print("CHECK::::::", torch.unique(self.voxel))
#         for i in range(self.voxel.size(0)):
#             if not os.path.exists(f'/home/ohjinjin/{self.voxel.device}_overlay10_v0_{i}.png'):
#                 voxel_image = self.voxel[i, 0, :, :].cpu().numpy()  # 각 배치의 첫 번째 채널 (256, 256)
#                 flow_image = flow_image10[i].cpu().numpy().transpose(1, 2, 0)  # (256, 256, 3)

#                 # 이미지 오버레이
#                 fig, ax = plt.subplots()
#                 ax.imshow(voxel_image, cmap='bwr', vmin=-1, vmax=1)  # voxel 이미지
#                 ax.imshow(flow_image, alpha=0.5)  # flow 이미지 오버레이

#                 # 이미지 저장
# #                 overlay_filename = os.path.join(output_dir, f'{self.flow10.device}_overlay_{i}.png')
#                 plt.axis('off')  # 축 숨기기
#                 plt.savefig(f'/home/ohjinjin/{self.voxel.device}_overlay10_v0_{i}.png', bbox_inches='tight', pad_inches=0)
#                 plt.close()
# #                 # 이미지 저장
# # #                 filename = os.path.join(output_dir, f'image_batch_{i}.png')
# #                 plt.imshow(batch_image, cmap='bwr', vmin=-1, vmax=1)
# # #                 plt.colorbar()  # 컬러바 추가
# #                 plt.axis('off')  # 축 숨기기
# #                 plt.savefig(f'/home/ohjinjin/{self.voxel.device}_overlay_v0_{i}.png', bbox_inches='tight', pad_inches=0)
# #                 plt.close()
#                 print(f'Saved')
#             if not os.path.exists(f'/home/ohjinjin/{self.voxel.device}_overlay10_v1_{i}.png'):
#                 voxel_image = self.voxel[i, 1, :, :].cpu().numpy()  # 각 배치의 첫 번째 채널 (256, 256)
#                 flow_image = flow_image10[i].cpu().numpy().transpose(1, 2, 0)  # (256, 256, 3)

#                 # 이미지 오버레이
#                 fig, ax = plt.subplots()
#                 ax.imshow(voxel_image, cmap='bwr', vmin=-1, vmax=1)  # voxel 이미지
#                 ax.imshow(flow_image, alpha=0.5)  # flow 이미지 오버레이

#                 # 이미지 저장
# #                 overlay_filename = os.path.join(output_dir, f'{self.flow10.device}_overlay_{i}.png')
#                 plt.axis('off')  # 축 숨기기
#                 plt.savefig(f'/home/ohjinjin/{self.voxel.device}_overlay10_v1_{i}.png', bbox_inches='tight', pad_inches=0)
#                 plt.close()

# #                 # 이미지 저장
# # #                 filename = os.path.join(output_dir, f'image_batch_{i}.png')
# #                 plt.imshow(batch_image, cmap='bwr', vmin=-1, vmax=1)
# # #                 plt.colorbar()  # 컬러바 추가
# #                 plt.axis('off')  # 축 숨기기
# #                 plt.savefig(f'/home/ohjinjin/{self.voxel.device}_overlay_v1_{i}.png', bbox_inches='tight', pad_inches=0)
# #                 plt.close()
#                 print(f'Saved')
#             if not os.path.exists(f'/home/ohjinjin/{self.voxel.device}_overlay10_v2_{i}.png'):
#                 voxel_image = self.voxel[i, 2, :, :].cpu().numpy()  # 각 배치의 첫 번째 채널 (256, 256)
#                 flow_image = flow_image10[i].cpu().numpy().transpose(1, 2, 0)  # (256, 256, 3)

#                 # 이미지 오버레이
#                 fig, ax = plt.subplots()
#                 ax.imshow(voxel_image, cmap='bwr', vmin=-1, vmax=1)  # voxel 이미지
#                 ax.imshow(flow_image, alpha=0.5)  # flow 이미지 오버레이

#                 # 이미지 저장
# #                 overlay_filename = os.path.join(output_dir, f'{self.flow10.device}_overlay_{i}.png')
#                 plt.axis('off')  # 축 숨기기
#                 plt.savefig(f'/home/ohjinjin/{self.voxel.device}_overlay10_v2_{i}.png', bbox_inches='tight', pad_inches=0)
#                 plt.close()

# #                 # 이미지 저장
# # #                 filename = os.path.join(output_dir, f'image_batch_{i}.png')
# #                 plt.imshow(batch_image, cmap='bwr', vmin=-1, vmax=1)
# # #                 plt.colorbar()  # 컬러바 추가
# #                 plt.axis('off')  # 축 숨기기
# #                 plt.savefig(f'/home/ohjinjin/{self.voxel.device}_overlay_v2_{i}.png', bbox_inches='tight', pad_inches=0)
# #                 plt.close()
#                 print(f'Saved')
#             if not os.path.exists(f'/home/ohjinjin/{self.voxel.device}_overlay10_v3_{i}.png'):
#                 voxel_image = self.voxel[i, 3, :, :].cpu().numpy()  # 각 배치의 첫 번째 채널 (256, 256)
#                 flow_image = flow_image10[i].cpu().numpy().transpose(1, 2, 0)  # (256, 256, 3)

#                 # 이미지 오버레이
#                 fig, ax = plt.subplots()
#                 ax.imshow(voxel_image, cmap='bwr', vmin=-1, vmax=1)  # voxel 이미지
#                 ax.imshow(flow_image, alpha=0.5)  # flow 이미지 오버레이

#                 # 이미지 저장
# #                 overlay_filename = os.path.join(output_dir, f'{self.flow10.device}_overlay_{i}.png')
#                 plt.axis('off')  # 축 숨기기
#                 plt.savefig(f'/home/ohjinjin/{self.voxel.device}_overlay10_v3_{i}.png', bbox_inches='tight', pad_inches=0)

# #                 # 이미지 저장
# # #                 filename = os.path.join(output_dir, f'image_batch_{i}.png')
# #                 plt.imshow(batch_image, cmap='bwr', vmin=-1, vmax=1)
# # #                 plt.colorbar()  # 컬러바 추가
# #                 plt.axis('off')  # 축 숨기기
# #                 plt.savefig(f'/home/ohjinjin/{self.voxel.device}_overlay_v3_{i}.png', bbox_inches='tight', pad_inches=0)
#                 plt.close()
#                 print(f'Saved')
#             if not os.path.exists(f'/home/ohjinjin/{self.voxel.device}_overlay10_v4_{i}.png'):
#                 voxel_image = self.voxel[i, 4, :, :].cpu().numpy()  # 각 배치의 첫 번째 채널 (256, 256)
#                 flow_image = flow_image10[i].cpu().numpy().transpose(1, 2, 0)  # (256, 256, 3)

#                 # 이미지 오버레이
#                 fig, ax = plt.subplots()
#                 ax.imshow(voxel_image, cmap='bwr', vmin=-1, vmax=1)  # voxel 이미지
#                 ax.imshow(flow_image, alpha=0.5)  # flow 이미지 오버레이

#                 # 이미지 저장
# #                 overlay_filename = os.path.join(output_dir, f'{self.flow10.device}_overlay_{i}.png')
#                 plt.axis('off')  # 축 숨기기
#                 plt.savefig(f'/home/ohjinjin/{self.voxel.device}_overlay10_v4_{i}.png', bbox_inches='tight', pad_inches=0)
#                 plt.close()

# #                 # 이미지 저장
# # #                 filename = os.path.join(output_dir, f'image_batch_{i}.png')
# #                 plt.imshow(batch_image, cmap='bwr', vmin=-1, vmax=1)
# # #                 plt.colorbar()  # 컬러바 추가
# #                 plt.axis('off')  # 축 숨기기
# #                 plt.savefig(f'/home/ohjinjin/{self.voxel.device}_overlay_v4_{i}.png', bbox_inches='tight', pad_inches=0)
# #                 plt.close()
#                 print(f'Saved')
#             if not os.path.exists(f'/home/ohjinjin/{self.voxel.device}_overlay10_v5_{i}.png'):
#                 voxel_image = self.voxel[i, 5, :, :].cpu().numpy()  # 각 배치의 첫 번째 채널 (256, 256)
#                 flow_image = flow_image10[i].cpu().numpy().transpose(1, 2, 0)  # (256, 256, 3)

#                 # 이미지 오버레이
#                 fig, ax = plt.subplots()
#                 ax.imshow(voxel_image, cmap='bwr', vmin=-1, vmax=1)  # voxel 이미지
#                 ax.imshow(flow_image, alpha=0.5)  # flow 이미지 오버레이

#                 # 이미지 저장
# #                 overlay_filename = os.path.join(output_dir, f'{self.flow10.device}_overlay_{i}.png')
#                 plt.axis('off')  # 축 숨기기
#                 plt.savefig(f'/home/ohjinjin/{self.voxel.device}_overlay10_v5_{i}.png', bbox_inches='tight', pad_inches=0)
#                 plt.close()

# #                 # 이미지 저장
# # #                 filename = os.path.join(output_dir, f'image_batch_{i}.png')
# #                 plt.imshow(batch_image, cmap='bwr', vmin=-1, vmax=1)
# # #                 plt.colorbar()  # 컬러바 추가
# #                 plt.axis('off')  # 축 숨기기
# #                 plt.savefig(f'/home/ohjinjin/{self.voxel.device}_overlay_v5_{i}.png', bbox_inches='tight', pad_inches=0)
# #                 plt.close()
#                 print(f'Saved')
#             if not os.path.exists(f'/home/ohjinjin/{self.voxel.device}_overlay12_v0_{i}.png'):
#                 voxel_image = self.voxel[i, 0, :, :].cpu().numpy()  # 각 배치의 첫 번째 채널 (256, 256)
#                 flow_image = flow_image12[i].cpu().numpy().transpose(1, 2, 0)  # (256, 256, 3)

#                 # 이미지 오버레이
#                 fig, ax = plt.subplots()
#                 ax.imshow(voxel_image, cmap='bwr', vmin=-1, vmax=1)  # voxel 이미지
#                 ax.imshow(flow_image, alpha=0.5)  # flow 이미지 오버레이

#                 # 이미지 저장
# #                 overlay_filename = os.path.join(output_dir, f'{self.flow10.device}_overlay_{i}.png')
#                 plt.axis('off')  # 축 숨기기
#                 plt.savefig(f'/home/ohjinjin/{self.voxel.device}_overlay12_v0_{i}.png', bbox_inches='tight', pad_inches=0)
#                 plt.close()
# #                 # 이미지 저장
# # #                 filename = os.path.join(output_dir, f'image_batch_{i}.png')
# #                 plt.imshow(batch_image, cmap='bwr', vmin=-1, vmax=1)
# # #                 plt.colorbar()  # 컬러바 추가
# #                 plt.axis('off')  # 축 숨기기
# #                 plt.savefig(f'/home/ohjinjin/{self.voxel.device}_overlay_v0_{i}.png', bbox_inches='tight', pad_inches=0)
# #                 plt.close()
#                 print(f'Saved')
#             if not os.path.exists(f'/home/ohjinjin/{self.voxel.device}_overlay12_v1_{i}.png'):
#                 voxel_image = self.voxel[i, 1, :, :].cpu().numpy()  # 각 배치의 첫 번째 채널 (256, 256)
#                 flow_image = flow_image12[i].cpu().numpy().transpose(1, 2, 0)  # (256, 256, 3)

#                 # 이미지 오버레이
#                 fig, ax = plt.subplots()
#                 ax.imshow(voxel_image, cmap='bwr', vmin=-1, vmax=1)  # voxel 이미지
#                 ax.imshow(flow_image, alpha=0.5)  # flow 이미지 오버레이

#                 # 이미지 저장
# #                 overlay_filename = os.path.join(output_dir, f'{self.flow10.device}_overlay_{i}.png')
#                 plt.axis('off')  # 축 숨기기
#                 plt.savefig(f'/home/ohjinjin/{self.voxel.device}_overlay12_v1_{i}.png', bbox_inches='tight', pad_inches=0)
#                 plt.close()

# #                 # 이미지 저장
# # #                 filename = os.path.join(output_dir, f'image_batch_{i}.png')
# #                 plt.imshow(batch_image, cmap='bwr', vmin=-1, vmax=1)
# # #                 plt.colorbar()  # 컬러바 추가
# #                 plt.axis('off')  # 축 숨기기
# #                 plt.savefig(f'/home/ohjinjin/{self.voxel.device}_overlay_v1_{i}.png', bbox_inches='tight', pad_inches=0)
# #                 plt.close()
#                 print(f'Saved')
#             if not os.path.exists(f'/home/ohjinjin/{self.voxel.device}_overlay12_v2_{i}.png'):
#                 voxel_image = self.voxel[i, 2, :, :].cpu().numpy()  # 각 배치의 첫 번째 채널 (256, 256)
#                 flow_image = flow_image12[i].cpu().numpy().transpose(1, 2, 0)  # (256, 256, 3)

#                 # 이미지 오버레이
#                 fig, ax = plt.subplots()
#                 ax.imshow(voxel_image, cmap='bwr', vmin=-1, vmax=1)  # voxel 이미지
#                 ax.imshow(flow_image, alpha=0.5)  # flow 이미지 오버레이

#                 # 이미지 저장
# #                 overlay_filename = os.path.join(output_dir, f'{self.flow10.device}_overlay_{i}.png')
#                 plt.axis('off')  # 축 숨기기
#                 plt.savefig(f'/home/ohjinjin/{self.voxel.device}_overlay12_v2_{i}.png', bbox_inches='tight', pad_inches=0)
#                 plt.close()

# #                 # 이미지 저장
# # #                 filename = os.path.join(output_dir, f'image_batch_{i}.png')
# #                 plt.imshow(batch_image, cmap='bwr', vmin=-1, vmax=1)
# # #                 plt.colorbar()  # 컬러바 추가
# #                 plt.axis('off')  # 축 숨기기
# #                 plt.savefig(f'/home/ohjinjin/{self.voxel.device}_overlay_v2_{i}.png', bbox_inches='tight', pad_inches=0)
# #                 plt.close()
#                 print(f'Saved')
#             if not os.path.exists(f'/home/ohjinjin/{self.voxel.device}_overlay12_v3_{i}.png'):
#                 voxel_image = self.voxel[i, 3, :, :].cpu().numpy()  # 각 배치의 첫 번째 채널 (256, 256)
#                 flow_image = flow_image12[i].cpu().numpy().transpose(1, 2, 0)  # (256, 256, 3)

#                 # 이미지 오버레이
#                 fig, ax = plt.subplots()
#                 ax.imshow(voxel_image, cmap='bwr', vmin=-1, vmax=1)  # voxel 이미지
#                 ax.imshow(flow_image, alpha=0.5)  # flow 이미지 오버레이

#                 # 이미지 저장
# #                 overlay_filename = os.path.join(output_dir, f'{self.flow10.device}_overlay_{i}.png')
#                 plt.axis('off')  # 축 숨기기
#                 plt.savefig(f'/home/ohjinjin/{self.voxel.device}_overlay12_v3_{i}.png', bbox_inches='tight', pad_inches=0)

# #                 # 이미지 저장
# # #                 filename = os.path.join(output_dir, f'image_batch_{i}.png')
# #                 plt.imshow(batch_image, cmap='bwr', vmin=-1, vmax=1)
# # #                 plt.colorbar()  # 컬러바 추가
# #                 plt.axis('off')  # 축 숨기기
# #                 plt.savefig(f'/home/ohjinjin/{self.voxel.device}_overlay_v3_{i}.png', bbox_inches='tight', pad_inches=0)
#                 plt.close()
#                 print(f'Saved')
#             if not os.path.exists(f'/home/ohjinjin/{self.voxel.device}_overlay12_v4_{i}.png'):
#                 voxel_image = self.voxel[i, 4, :, :].cpu().numpy()  # 각 배치의 첫 번째 채널 (256, 256)
#                 flow_image = flow_image12[i].cpu().numpy().transpose(1, 2, 0)  # (256, 256, 3)

#                 # 이미지 오버레이
#                 fig, ax = plt.subplots()
#                 ax.imshow(voxel_image, cmap='bwr', vmin=-1, vmax=1)  # voxel 이미지
#                 ax.imshow(flow_image, alpha=0.5)  # flow 이미지 오버레이

#                 # 이미지 저장
# #                 overlay_filename = os.path.join(output_dir, f'{self.flow10.device}_overlay_{i}.png')
#                 plt.axis('off')  # 축 숨기기
#                 plt.savefig(f'/home/ohjinjin/{self.voxel.device}_overlay12_v4_{i}.png', bbox_inches='tight', pad_inches=0)
#                 plt.close()

# #                 # 이미지 저장
# # #                 filename = os.path.join(output_dir, f'image_batch_{i}.png')
# #                 plt.imshow(batch_image, cmap='bwr', vmin=-1, vmax=1)
# # #                 plt.colorbar()  # 컬러바 추가
# #                 plt.axis('off')  # 축 숨기기
# #                 plt.savefig(f'/home/ohjinjin/{self.voxel.device}_overlay_v4_{i}.png', bbox_inches='tight', pad_inches=0)
# #                 plt.close()
#                 print(f'Saved')
#             if not os.path.exists(f'/home/ohjinjin/{self.voxel.device}_overlay12_v5_{i}.png'):
#                 voxel_image = self.voxel[i, 5, :, :].cpu().numpy()  # 각 배치의 첫 번째 채널 (256, 256)
#                 flow_image = flow_image12[i].cpu().numpy().transpose(1, 2, 0)  # (256, 256, 3)

#                 # 이미지 오버레이
#                 fig, ax = plt.subplots()
#                 ax.imshow(voxel_image, cmap='bwr', vmin=-1, vmax=1)  # voxel 이미지
#                 ax.imshow(flow_image, alpha=0.5)  # flow 이미지 오버레이

#                 # 이미지 저장
# #                 overlay_filename = os.path.join(output_dir, f'{self.flow10.device}_overlay_{i}.png')
#                 plt.axis('off')  # 축 숨기기
#                 plt.savefig(f'/home/ohjinjin/{self.voxel.device}_overlay12_v5_{i}.png', bbox_inches='tight', pad_inches=0)
#                 plt.close()

# #                 # 이미지 저장
# # #                 filename = os.path.join(output_dir, f'image_batch_{i}.png')
# #                 plt.imshow(batch_image, cmap='bwr', vmin=-1, vmax=1)
# # #                 plt.colorbar()  # 컬러바 추가
# #                 plt.axis('off')  # 축 숨기기
# #                 plt.savefig(f'/home/ohjinjin/{self.voxel.device}_overlay_v5_{i}.png', bbox_inches='tight', pad_inches=0)
# #                 plt.close()
#                 print(f'Saved')
        
        if 'mask' in data:
            self.mask = data['mask'].to(self.device)
        if 'frame_gt' in data:
            self.gt = data['frame_gt'].to(self.device)

    def transpose(self, t, trans_idx):
        # print('transpose jt .. ', t.size())
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return torch.rot90(t, trans_idx % 4, [2, 3])

    def transpose_inverse(self, t, trans_idx):
        # print( 'inverse transpose .. t', t.size())
        t = torch.rot90(t, 4 - trans_idx % 4, [2, 3])
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return t

    def grids_voxel(self):
        b, c, h, w = self.voxel.size()
        self.original_size_voxel = self.voxel.size()
        assert b == 1
        crop_size = self.opt['val'].get('crop_size')
        # step_j = self.opt['val'].get('step_j', crop_size)
        # step_i = self.opt['val'].get('step_i', crop_size)
        ##adaptive step_i, step_j
        num_row = (h - 1) // crop_size + 1
        num_col = (w - 1) // crop_size + 1

        import math
        step_j = crop_size if num_col == 1 else math.ceil((w - crop_size) / (num_col - 1) - 1e-8)
        step_i = crop_size if num_row == 1 else math.ceil((h - crop_size) / (num_row - 1) - 1e-8)

        # print('step_i, stepj', step_i, step_j)
        # exit(0)


        parts = []
        idxes = []

        # cnt_idx = 0

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size >= h:
                i = h - crop_size
                last_i = True


            last_j = False
            while j < w and not last_j:
                if j + crop_size >= w:
                    j = w - crop_size
                    last_j = True
                # from i, j to i+crop_szie, j + crop_size
                # print(' trans 8')
                for trans_idx in range(self.opt['val'].get('trans_num', 1)):
                    parts.append(self.transpose(self.voxel[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                    idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})
                    # cnt_idx += 1
                j = j + step_j
            i = i + step_i
        if self.opt['val'].get('random_crop_num', 0) > 0:
            for _ in range(self.opt['val'].get('random_crop_num')):
                import random
                i = random.randint(0, h-crop_size)
                j = random.randint(0, w-crop_size)
                trans_idx = random.randint(0, self.opt['val'].get('trans_num', 1) - 1)
                parts.append(self.transpose(self.voxel[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})


        self.origin_voxel = self.voxel
        self.voxel = torch.cat(parts, dim=0)
        print('----------parts voxel .. ', len(parts), self.voxel.size())
        self.idxes = idxes


    def grids(self):
        b, c, h, w = self.lq.size()  # lq is after data augment (for example, crop, if have)
        self.original_size = self.lq.size()
        assert b == 1
        crop_size = self.opt['val'].get('crop_size')
        # step_j = self.opt['val'].get('step_j', crop_size)
        # step_i = self.opt['val'].get('step_i', crop_size)
        ##adaptive step_i, step_j
        num_row = (h - 1) // crop_size + 1
        num_col = (w - 1) // crop_size + 1

        import math
        step_j = crop_size if num_col == 1 else math.ceil((w - crop_size) / (num_col - 1) - 1e-8)
        step_i = crop_size if num_row == 1 else math.ceil((h - crop_size) / (num_row - 1) - 1e-8)


        # print('step_i, stepj', step_i, step_j)
        # exit(0)


        parts = []
        idxes = []

        # cnt_idx = 0

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size >= h:
                i = h - crop_size
                last_i = True


            last_j = False
            while j < w and not last_j:
                if j + crop_size >= w:
                    j = w - crop_size
                    last_j = True
                # from i, j to i+crop_szie, j + crop_size
                # print(' trans 8')
                for trans_idx in range(self.opt['val'].get('trans_num', 1)):
                    parts.append(self.transpose(self.lq[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                    idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})
                    # cnt_idx += 1
                j = j + step_j
            i = i + step_i
        if self.opt['val'].get('random_crop_num', 0) > 0:
            for _ in range(self.opt['val'].get('random_crop_num')):
                import random
                i = random.randint(0, h-crop_size)
                j = random.randint(0, w-crop_size)
                trans_idx = random.randint(0, self.opt['val'].get('trans_num', 1) - 1)
                parts.append(self.transpose(self.lq[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})


        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        # print('parts .. ', len(parts), self.lq.size())
        self.idxes = idxes
    
    def grids_flow02(self):
        b, c, h, w = self.flow02.size()  # lq is after data augment (for example, crop, if have)
        self.original_size_flow02 = self.flow02.size()
        assert b == 1
        crop_size = self.opt['val'].get('crop_size')
        # step_j = self.opt['val'].get('step_j', crop_size)
        # step_i = self.opt['val'].get('step_i', crop_size)
        ##adaptive step_i, step_j
        num_row = (h - 1) // crop_size + 1
        num_col = (w - 1) // crop_size + 1

        import math
        step_j = crop_size if num_col == 1 else math.ceil((w - crop_size) / (num_col - 1) - 1e-8)
        step_i = crop_size if num_row == 1 else math.ceil((h - crop_size) / (num_row - 1) - 1e-8)


        # print('step_i, stepj', step_i, step_j)
        # exit(0)


        parts = []
        idxes = []

        # cnt_idx = 0

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size >= h:
                i = h - crop_size
                last_i = True


            last_j = False
            while j < w and not last_j:
                if j + crop_size >= w:
                    j = w - crop_size
                    last_j = True
                # from i, j to i+crop_szie, j + crop_size
                # print(' trans 8')
                for trans_idx in range(self.opt['val'].get('trans_num', 1)):
                    parts.append(self.transpose(self.flow02[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                    idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})
                    # cnt_idx += 1
                j = j + step_j
            i = i + step_i
        if self.opt['val'].get('random_crop_num', 0) > 0:
            for _ in range(self.opt['val'].get('random_crop_num')):
                import random
                i = random.randint(0, h-crop_size)
                j = random.randint(0, w-crop_size)
                trans_idx = random.randint(0, self.opt['val'].get('trans_num', 1) - 1)
                parts.append(self.transpose(self.flow02[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})


        self.origin_flow02 = self.flow02
        self.flow02 = torch.cat(parts, dim=0)
        # print('parts .. ', len(parts), self.lq.size())
        self.idxes = idxes
        
    def grids_flow10(self):
        b, c, h, w = self.flow10.size()  # lq is after data augment (for example, crop, if have)
        self.original_size_flow10 = self.flow10.size()
        assert b == 1
        crop_size = self.opt['val'].get('crop_size')
        # step_j = self.opt['val'].get('step_j', crop_size)
        # step_i = self.opt['val'].get('step_i', crop_size)
        ##adaptive step_i, step_j
        num_row = (h - 1) // crop_size + 1
        num_col = (w - 1) // crop_size + 1

        import math
        step_j = crop_size if num_col == 1 else math.ceil((w - crop_size) / (num_col - 1) - 1e-8)
        step_i = crop_size if num_row == 1 else math.ceil((h - crop_size) / (num_row - 1) - 1e-8)


        # print('step_i, stepj', step_i, step_j)
        # exit(0)


        parts = []
        idxes = []

        # cnt_idx = 0

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size >= h:
                i = h - crop_size
                last_i = True


            last_j = False
            while j < w and not last_j:
                if j + crop_size >= w:
                    j = w - crop_size
                    last_j = True
                # from i, j to i+crop_szie, j + crop_size
                # print(' trans 8')
                for trans_idx in range(self.opt['val'].get('trans_num', 1)):
                    parts.append(self.transpose(self.flow10[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                    idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})
                    # cnt_idx += 1
                j = j + step_j
            i = i + step_i
        if self.opt['val'].get('random_crop_num', 0) > 0:
            for _ in range(self.opt['val'].get('random_crop_num')):
                import random
                i = random.randint(0, h-crop_size)
                j = random.randint(0, w-crop_size)
                trans_idx = random.randint(0, self.opt['val'].get('trans_num', 1) - 1)
                parts.append(self.transpose(self.flow10[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})


        self.origin_flow10 = self.flow10
        self.flow10 = torch.cat(parts, dim=0)
        # print('parts .. ', len(parts), self.lq.size())
        self.idxes = idxes
        
    def grids_flow12(self):
        b, c, h, w = self.flow12.size()  # lq is after data augment (for example, crop, if have)
        self.original_size_flow12 = self.flow12.size()
        assert b == 1
        crop_size = self.opt['val'].get('crop_size')
        # step_j = self.opt['val'].get('step_j', crop_size)
        # step_i = self.opt['val'].get('step_i', crop_size)
        ##adaptive step_i, step_j
        num_row = (h - 1) // crop_size + 1
        num_col = (w - 1) // crop_size + 1

        import math
        step_j = crop_size if num_col == 1 else math.ceil((w - crop_size) / (num_col - 1) - 1e-8)
        step_i = crop_size if num_row == 1 else math.ceil((h - crop_size) / (num_row - 1) - 1e-8)


        # print('step_i, stepj', step_i, step_j)
        # exit(0)


        parts = []
        idxes = []

        # cnt_idx = 0

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size >= h:
                i = h - crop_size
                last_i = True


            last_j = False
            while j < w and not last_j:
                if j + crop_size >= w:
                    j = w - crop_size
                    last_j = True
                # from i, j to i+crop_szie, j + crop_size
                # print(' trans 8')
                for trans_idx in range(self.opt['val'].get('trans_num', 1)):
                    parts.append(self.transpose(self.flow12[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                    idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})
                    # cnt_idx += 1
                j = j + step_j
            i = i + step_i
        if self.opt['val'].get('random_crop_num', 0) > 0:
            for _ in range(self.opt['val'].get('random_crop_num')):
                import random
                i = random.randint(0, h-crop_size)
                j = random.randint(0, w-crop_size)
                trans_idx = random.randint(0, self.opt['val'].get('trans_num', 1) - 1)
                parts.append(self.transpose(self.flow12[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})


        self.origin_flow12 = self.flow10
        self.flow12 = torch.cat(parts, dim=0)
        # print('parts .. ', len(parts), self.lq.size())
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size).to(self.device)
        b, c, h, w = self.original_size

        print('...', self.device)

        count_mt = torch.zeros((b, 1, h, w)).to(self.device)
        crop_size = self.opt['val'].get('crop_size')

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            trans_idx = each_idx['trans_idx']
            preds[0, :, i:i + crop_size, j:j + crop_size] += self.transpose_inverse(self.output[cnt, :, :, :].unsqueeze(0), trans_idx).squeeze(0)
            count_mt[0, 0, i:i + crop_size, j:j + crop_size] += 1.

        self.output = preds / count_mt
        self.lq = self.origin_lq
        self.voxel = self.origin_voxel
#         self.flow10 = self.origin_flow10
#         self.flow12 = self.origin_flow12
        self.flow02 = self.origin_flow02


    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        
        self.input_event_flow = torch.cat((self.voxel, self.flow02), dim=1)

        if self.opt['datasets']['train'].get('use_mask'):
            preds = self.net_g(x = self.lq, event = self.input_event_flow, mask = self.mask)

        elif self.opt['datasets']['train'].get('return_ren'):
            preds = self.net_g(x = self.lq, event = self.input_event_flow, ren = self.ren)

        else:
            preds = self.net_g(x = self.lq, event = self.input_event_flow)

        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = 0.

            if self.pixel_type == 'PSNRATLoss':
                l_pix += self.cri_pix(*preds, self.gt)

            elif self.pixel_type == 'PSNRGateLoss':
                for pred in preds:
                    l_pix += self.cri_pix(pred, self.gt, self.mask)

            elif self.pixel_type == 'PSNRLoss':
                for pred in preds:
                    l_pix += self.cri_pix(pred, self.gt)
            
            else:
                for pred in preds:
                    l_pix += self.cri_pix(pred, self.gt)             

            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        # if self.cri_perceptual:
        #
        #
        #     l_percep, l_style = self.cri_perceptual(self.output, self.gt)
        #
        #     if l_percep is not None:
        #         l_total += l_percep
        #         loss_dict['l_percep'] = l_percep
        #     if l_style is not None:
        #         l_total += l_style
        #         loss_dict['l_style'] = l_style

        l_total = l_total + 0 * sum(p.sum() for p in self.net_g.parameters())

        l_total.backward()
        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()


        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.input_event_flow = torch.cat((self.voxel, self.flow02), dim=1)
            n = self.lq.size(0)  # n: batch size
            outs = []
            m = self.opt['val'].get('max_minibatch', n)  # m is the minibatch, equals to batch size or mini batch size
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n

                if self.opt['datasets']['val'].get('use_mask'):
                    pred = self.net_g(x = self.lq[i:j, :, :, :], event = self.input_event_flow[i:j, :, :, :], mask = self.mask[i:j, :, :, :])  # mini batch all in 

                elif self.opt['datasets']['val'].get('return_ren'):
                    pred = self.net_g(x = self.lq[i:j, :, :, :], event = self.input_event_flow[i:j, :, :, :], ren = self.ren[i:j,:])

                else:
                    pred = self.net_g(x = self.lq[i:j, :, :, :], event = self.input_event_flow[i:j, :, :, :])  # mini batch all in 
            
                if isinstance(pred, list):
                    pred = pred[-1]
                outs.append(pred)
                i = j

            self.output = torch.cat(outs, dim=0)  # all mini batch cat in dim0
        self.net_g.train()

#     def single_image_inference(self, img, voxel, save_path):
#         self.feed_data(data={'frame': img.unsqueeze(dim=0), 'voxel': voxel.unsqueeze(dim=0)})
#         if self.opt['val'].get('grids') is not None:
#             self.grids()
#             self.grids_voxel()
#             self.grids_flow10()
#             self.grids_flow12()

#         self.test()

#         if self.opt['val'].get('grids') is not None:
#             self.grids_inverse()
#             # self.grids_inverse_voxel()

#         visuals = self.get_current_visuals()
#         sr_img = tensor2img([visuals['result']])
#         imwrite(sr_img, save_path)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        logger = get_root_logger()
        # logger.info('Only support single GPU validation.')
        import os
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = self.opt.get('name') # !
        
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = '{:08d}'.format(cnt)

            self.feed_data(val_data)
            if self.opt['val'].get('grids') is not None:
                self.grids()
                self.grids_voxel()
#                 self.grids_flow10()
#                 self.grids_flow12()
                self.grids_flow02()

            self.test()

            if self.opt['val'].get('grids') is not None:
                self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                
                if self.opt['is_train']:
                    if cnt == 1: # visualize cnt=1 image every time
                        save_img_path = osp.join(self.opt['path']['visualization'],
                                                img_name,
                                                f'{img_name}_{current_iter}.png')
                        
                        save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                img_name,
                                                f'{img_name}_{current_iter}_gt.png')
                else:
#                     print('Save path:{}'.format(self.opt['path']['visualization']))
#                     print('Dataset name:{}'.format(dataset_name))
#                     print('Img_name:{}'.format(img_name))
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')
                    
                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
            cnt += 1
        pbar.close()

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
