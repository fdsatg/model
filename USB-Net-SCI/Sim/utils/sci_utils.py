import scipy.io as sio
import os
import numpy as np
import torch
from utils.ssim_torch import ssim


def generate_masks(mask_path, batch_size):
    mask = sio.loadmat(mask_path + "/mask.mat")
    mask = mask["mask"]
    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)
    [nC, H, W] = mask3d.shape
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()
    return mask3d_batch


def generate_shift_masks(mask_path, batch_size):
    mask = sio.loadmat(mask_path + "/mask_3d_shift.mat")
    mask_3d_shift = mask["mask_3d_shift"]
    mask_3d_shift = np.transpose(mask_3d_shift, [2, 0, 1])
    mask_3d_shift = torch.from_numpy(mask_3d_shift)
    [nC, H, W] = mask_3d_shift.shape
    Phi_batch = mask_3d_shift.expand([batch_size, nC, H, W]).cuda().float()
    Phi_s_batch = torch.sum(Phi_batch**2, 1)
    Phi_s_batch[Phi_s_batch == 0] = 1
    return Phi_batch, Phi_s_batch


def LoadTest(path_test):
    scene_list = os.listdir(path_test)
    scene_list.sort()
    test_data = np.zeros((len(scene_list), 256, 256, 28))
    for i in range(len(scene_list)):
        scene_path = path_test + scene_list[i]
        img = sio.loadmat(scene_path)["img"]
        test_data[i, :, :, :] = img
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    return test_data


# We find that this calculation method is more close to DGSMP's.
def torch_psnr(img, ref):  # input [28,256,256]
    img = (img * 256).round()
    ref = (ref * 256).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((255 * 255) / mse)
    return psnr / nC


def torch_ssim(img, ref):  # input [28,256,256]
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))


def gen_meas_torch(data_batch, mask3d_batch, Y2H=True, mul_mask=False):
    nC = data_batch.shape[1]
    temp = shift(mask3d_batch * data_batch, 2)
    meas = torch.sum(temp, 1)
    if Y2H:
        meas = meas / nC * 2
        H = shift_back(meas)
        if mul_mask:
            HM = torch.mul(H, mask3d_batch)
            return HM
        return H
    return meas


def shift(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col + (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, step * i : step * i + col] = inputs[:, i, :, :]
    return output


def shift_back(inputs, step=2):  # input [bs,256,310]  output [bs, 28, 256, 256]
    [bs, row, col] = inputs.shape
    nC = 28
    output = torch.zeros(bs, nC, row, col - (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, :] = inputs[:, :, step * i : step * i + col - (nC - 1) * step]
    return output


def init_mask(mask_path, mask_type, batch_size):
    mask3d_batch = generate_masks(mask_path, batch_size)
    if mask_type == "Phi":
        shift_mask3d_batch = shift(mask3d_batch)
        input_mask = shift_mask3d_batch
    elif mask_type == "Phi_PhiPhiT":
        Phi_batch, Phi_s_batch = generate_shift_masks(mask_path, batch_size)
        input_mask = (Phi_batch, Phi_s_batch)
    elif mask_type == "Mask":
        input_mask = mask3d_batch
    elif mask_type == None:
        input_mask = None
    return mask3d_batch, input_mask


def init_meas(gt, mask, input_setting):
    if input_setting == "H":
        input_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=False)
    elif input_setting == "HM":
        input_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=True)
    elif input_setting == "Y":
        input_meas = gen_meas_torch(gt, mask, Y2H=False, mul_mask=True)
    return input_meas

