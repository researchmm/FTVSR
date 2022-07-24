import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def build_filter(pos, freq, POS):
    result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
    if freq == 0:
        return result
    else:
        return result * math.sqrt(2)
    
def generate_dct_matrix(h=8, w=8):
    matrix = np.zeros((h, w, h, w))
    
    us = list(range(h))
    vs = list(range(w))
    

    
    for u in range(h):
        for v in range(w):
            for i in range(h):
                for j in range(w):
                    matrix[u, v, i, j] = build_filter(i, u, h) * build_filter(j, v, w)

    matrix = matrix.reshape(-1, h, w)
    
    return matrix


class dct_layer(nn.Module):
    def __init__(self, in_c=3, h=8, w=8):
        super(dct_layer, self).__init__()
        assert h == w

        self.dct_conv = nn.Conv2d(in_c, in_c*h*w, h, h, bias=False, groups=in_c) 
        matrix = generate_dct_matrix(h=h, w=w)
        self.weight = torch.from_numpy(matrix).float().unsqueeze(1) #64,1,8,8
        
        self.dct_conv.weight.data = torch.cat([self.weight] * in_c, dim=0) #192,1,8,8
        self.dct_conv.weight.requires_grad = False

    def forward(self, x):
        dct = self.dct_conv(x)

        return dct


class reverse_dct_layer(nn.Module):
    def __init__(self, out_c=3, h=8, w=8):
        super(reverse_dct_layer, self).__init__()

        assert h == w

        self.reverse_dct_conv = nn.ConvTranspose2d(out_c * h * w, out_c, h, h, bias=False, groups=out_c)
        matrix = generate_dct_matrix(h=h, w=w)
        self.weight = torch.from_numpy(matrix).float().unsqueeze(1) #64,1,8,8

        self.reverse_dct_conv.weight.data = torch.cat([self.weight] * out_c, dim=0) #192,1,8,8
        self.reverse_dct_conv.weight.requires_grad = False

    def forward(self, x):
        rdct = self.reverse_dct_conv(x)

        return rdct

def check_and_padding_imgs(imgs, dct_kernel=(8,8)):
    n,t,c,h,w = imgs.size()

    if h % dct_kernel[0] != 0:
        k_t = h // dct_kernel[0]
        new_h = (k_t + 1) * dct_kernel[0]
    else:
        new_h = h
    
    if w % dct_kernel[1] != 0:
        k_t = w // dct_kernel[1]
        new_w = (k_t + 1) * dct_kernel[1]
    else:
        new_w = w 
    
    new_imgs = imgs.new_zeros(n, t, c, new_h, new_w)
    padding_h = new_h - h
    padding_w = new_w - w 

    new_imgs[:, :, :, :h, :w] = imgs
    new_imgs[:, :, :, -padding_h:, -padding_w:] = imgs[:, :, :, -padding_h:, -padding_w:]

    return new_imgs, padding_h, padding_w 

def remove_image_padding(imgs, padding_h, padding_w):
    n, c, h, w = imgs.shape

    new_imgs = imgs[:, :, :h-padding_h, :w-padding_w]

    return new_imgs


def resize_flow(flow, size_type, sizes, interp_mode='bilinear', align_corners=False):
    """Resize a flow according to ratio or shape.
    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.
    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow, size=(output_h, output_w), mode=interp_mode, align_corners=align_corners)
    return resized_flow



