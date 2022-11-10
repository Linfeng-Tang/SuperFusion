import torch
import argparse
import os
from model import SuperFusion
from dataset import TestData, imsave
from time import time
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES']='2'
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='Reg&Fusion', help='Reg for only image registration, Fusion for only image fusion, Reg&Fusion for image registration and fusion')
parser.add_argument('--dataset_name', type=str, default='MSRS', help='MSRS or Roadscene')

if __name__ == '__main__':
    opts = parser.parse_args()
    img_path = os.path.join('./dataset/test', opts.dataset_name)
    if opts.mode == 'Fusion':
        ir_path = os.path.join(img_path, 'ir')
    else:
        ir_path = os.path.join(img_path, 'ir_warp')
    vi_path = os.path.join(img_path, 'vi')
    model_path = os.path.join('./checkpoint', opts.dataset_name + '.pth')
    save_dir = os.path.join('./results', opts.mode,  opts.dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    model = SuperFusion()
    model.resume(model_path)
    model = model.cuda()
    model.eval()
    test_dataloader = TestData(ir_path, vi_path)
    p_bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))

    for idx, [ir, vi, name] in p_bar:
        vi_tensor = vi.cuda()
        ir_tenor = ir.cuda()
        start = time()
        with torch.no_grad():
            if opts.mode == 'Reg':
                results = model.registration_forward(ir_tenor, vi_tensor)
            elif opts.mode == 'Fusion':
                results = model.fusion_forward(ir_tenor, vi_tensor)
            else:
                results = model.forward(ir_tenor, vi_tensor)
        end = time()
        imsave(results, os.path.join(save_dir, name))

        if opts.mode == 'Reg':
            p_bar.set_description(f'registering {name} | time : {str(round(end - start, 4))}')
        elif opts.mode == 'Fusion':
            p_bar.set_description(f'fusing {name} | time : {str(round(end - start, 4))}')
        else:
            p_bar.set_description(f'registering and fusing {name} | time : {str(round(end - start, 4))}')
