import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
from pytorch_msssim import ms_ssim
import argparse
import numpy as np
from collections import defaultdict
from PIL import Image
import math
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
    return window


def MSSSIM_Calculation(img1, img2):
    """
    img1: (1, 3, W, H)
    img2: (1, 3, W, H)
    """
    # convert inputs to tensors
    img1 = torch.from_numpy(img1).float().unsqueeze(0).permute(0, 3, 1, 2)
    img2 = torch.from_numpy(img2).float().unsqueeze(0).permute(0, 3, 1, 2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return ms_ssim(img1, img2, data_range=255, size_average=False).numpy()[0]


def SSIM_Calculation(img1, img2):
    """
    img1: Tensor
    img2: Tensor
    """
    # convert inputs to tensors
    img1 = torch.from_numpy(img1).float().unsqueeze(0).permute(0, 3, 1, 2)
    img2 = torch.from_numpy(img2).float().unsqueeze(0).permute(0, 3, 1, 2)

    (_, channel, _, _) = img1.size()

    window_size = 11
    window = create_window(window_size, channel)

    mu1 = F.conv2d(img1, window, padding = int(window_size/2), groups = channel)
    mu2 = F.conv2d(img2, window, padding = int(window_size/2), groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = int(window_size/2), groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = int(window_size/2), groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = int(window_size/2), groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def PSNR_Calculation(img1, img2):
    """
    img1: Numpy
    img2: Numpy
    """
    mse = np.mean( (img1/255. - img2/255.) ** 2 )

    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def pre_processing(base_path):
    # A: blur B: sharp
    record = defaultdict(list)
    for img_file in os.listdir(base_path):
        name_list = img_file.split('_')
        if (name_list[1] == 'fake' and name_list[2].split('.')[0] == 'B') or (name_list[1] == 'real' and name_list[2].split('.')[0] == 'A'):
            record[name_list[0]] += [img_file]
    return record


def main():
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--resultFolder', required=True, help='path to synthetic images.')
    arg = parser.parse_args()
    folder_path = arg.resultFolder
    
    output_record = pre_processing(folder_path)
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_msssim = 0.0

    for _, val in output_record.items():
        img1_path = os.path.join(folder_path, val[0])
        img2_path = os.path.join(folder_path, val[1])
        img1 = np.array(Image.open(img1_path), dtype=np.float64)
        img2 = np.array(Image.open(img2_path), dtype=np.float64)

        # PSNR
        avg_psnr += PSNR_Calculation(img1, img2)
        # SSIM
        avg_ssim += SSIM_Calculation(img1, img2)
        # MSSSIM
        avg_msssim += MSSSIM_Calculation(img1, img2)
    
    print('Average PSNR: {:.3f}'.format(avg_psnr / len(output_record)))
    print('Average SSIM: {:.3f}'.format(avg_ssim / len(output_record)))
    print('Average MS-SSIM: {:.3f}'.format(avg_msssim / len(output_record)))


if __name__ == '__main__':
    main()

