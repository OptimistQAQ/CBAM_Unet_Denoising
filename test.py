import torch
import sys
import os
## add current path to the enviorment
# __file__ = "/raid/qinjiahao/projects/PMN"
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from archs import *

import os
from glob import glob
from mipi_starting_kit.code_example.dataset import calculate_ratio, depack_meta
from mipi_starting_kit.code_example.lite_isp import process
from torchvision.utils import save_image


def load_weights(model, pretrained_dict, multi_gpu=False, by_name=False):
    model_dict = model.module.state_dict() if multi_gpu else model.state_dict()
    # 1. filter out unnecessary keys
    tsm_replace = []
    for k in pretrained_dict:
        if 'tsm_shift' in k:
            k_new = k.replace('tsm_shift', 'tsm_buffer')
            tsm_replace.append((k, k_new))
    for k, k_new in tsm_replace:
        pretrained_dict[k_new] = pretrained_dict[k]
    if by_name:
        del_list = []
        for k, v in pretrained_dict.items():
            if k in model_dict:
                if model_dict[k].shape != pretrained_dict[k].shape:
                    del_list.append(k)

                pretrained_dict[k] = v
            else:
                del_list.append(k)
        for k in del_list:
            del pretrained_dict[k]
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    if multi_gpu:
        model.module.load_state_dict(model_dict)
    else:
        model.load_state_dict(model_dict)
    return model

device = "cpu"
model_path = "checkpoints\\SonyA7S2_My_Unet_best_model.pth"

n_channel_in = {}
n_channel_in['in_nc'] = 4
net = Unet(n_channel_in)

model = torch.load(model_path, map_location=device)

net = load_weights(net, model, by_name=True)

cameras = ['Camera1', 'Camera2']
data_path = 'data\\valid'
save_path = 'data\\image'

my_algo = lambda x, camera: x       # a function for process the image
                                    # you could use different weights for different cameras

for curr_cam in cameras:
    os.makedirs(f'{save_path}/{curr_cam}', exist_ok=True)
    input_npzs = list(sorted(glob(f'{data_path}/{curr_cam}/short/*.npz')))
    print(len(input_npzs))
    for input_npz in input_npzs:
        ## load and prepare data
        ratio = calculate_ratio(input_npz)
        im, wb, cam2rgb = depack_meta(input_npz, to_tensor=True)
        
        im = (im * ratio).clamp(None, 1.0).unsqueeze(0)
        wb = wb.unsqueeze(0)
        cam2rgb = cam2rgb.unsqueeze(0)

        ## denoising
        out = net(im)

        print(out.shape)

        ## perform lite isp and save image
        rgb = process(out, wb, cam2rgb, gamma=2.2)

        ## save image
        curr_im_name = os.path.basename(input_npz).replace('.npz', '.png')
        save_image(rgb, f'{save_path}/{curr_cam}/{curr_im_name}')
