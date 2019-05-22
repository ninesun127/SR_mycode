import argparse
import os
from math import log10
import numpy as np
import pandas as pd
import torch
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import ToTensor,ToPILImage
import pytorch_ssim
from data_utils import TestDatasetFromFolder, display_transform
from model import Generator

parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--model_name', default='netG_epoch_4_1000.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
MODEL_NAME = opt.model_name

results = {'Set5': {'psnr': [], 'ssim': []}, 'Set14': {'psnr': [], 'ssim': []}, 'BSD100': {'psnr': [], 'ssim': []},
           'Urban100': {'psnr': [], 'ssim': []}, 'SunHays80': {'psnr': [], 'ssim': []}}


device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
model = Generator(UPSCALE_FACTOR).eval().to(device)
model.load_state_dict(torch.load('epochs/' + MODEL_NAME))

test_set = TestDatasetFromFolder('data/test', upscale_factor=UPSCALE_FACTOR)
test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

out_path = 'benchmark_results/SRF_' + str(UPSCALE_FACTOR) + '/'
if not os.path.exists(out_path):
    os.makedirs(out_path)

for image_name, lr_image, hr_restore_img, hr_image in test_bar:
    image_name = image_name[0]
    #lr_image = Variable(lr_image, volatile=True)
    #hr_image = Variable(hr_image, volatile=Tru
    lr_image=lr_image.to(device)
    hr_image=hr_image.to(device)

    sr_image = model(lr_image)

    y_hr,_,_r=ToPILImage()(hr_image.cpu().squeeze(0)).convert('YCbCr').split()
    y_sr,_,_r=ToPILImage()(sr_image.cpu().squeeze(0)).convert('YCbCr').split()
    y_hr=ToTensor()(y_hr).unsqueeze(0)
    y_sr=ToTensor()(y_sr).unsqueeze(0)
    mse = ((y_hr - y_sr) ** 2).data.mean()
    psnr = 10 * log10(1 / mse)
    ssim = pytorch_ssim.ssim(y_sr, y_hr).item()

    test_images = torch.stack(
        [display_transform()(hr_restore_img.squeeze(0)), display_transform()(hr_image.data.cpu().squeeze(0)),
         display_transform()(sr_image.data.cpu().squeeze(0))])
    image = utils.make_grid(test_images, nrow=3, padding=5)
    utils.save_image(image, out_path + image_name.split('.')[0] + '_Y_channel_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) +
                     image_name.split('.')[-1], padding=5)

    # save psnr\ssim
    results[image_name.split('_')[0]]['psnr'].append(psnr)
    results[image_name.split('_')[0]]['ssim'].append(ssim)

out_path = 'statistics/'
saved_results = {'psnr': [], 'ssim': []}
for item in results.values():
    psnr = np.array(item['psnr'])
    ssim = np.array(item['ssim'])
    if (len(psnr) == 0) or (len(ssim) == 0):
        psnr = 'No data'
        ssim = 'No data'
    else:
        psnr = psnr.mean()
        ssim = ssim.mean()
    saved_results['psnr'].append(psnr)
    saved_results['ssim'].append(ssim)

data_frame = pd.DataFrame(saved_results, results.keys())
data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_test_Y_channel_results.csv', index_label='DataSet')
