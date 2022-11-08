import torch
import time
import os
import numpy as np
from models.networks import NGP
from models.rendering import render
from metrics import psnr
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from datasets import dataset_dict
from datasets.ray_utils import get_rays
from utils import load_ckpt
from train import depth2img
import imageio

dataset_name = 'ngp'
scene = 'shuttle4_scaled'
dataset = dataset_dict[dataset_name](
    f'/home/ubuntu/data/shuttle4',
    downsample=1/1
)

model = NGP().cuda()
ckpt_path = f'/home/ubuntu/ngp_pl/ckpts/{dataset_name}/{scene}/epoch=29_slim.ckpt'
print(ckpt_path)
load_ckpt(model, ckpt_path)

from datasets.ray_utils import get_ray_directions

psnrs = []; ts = []; imgs = []; depths = []
# os.makedirs(f'results/{dataset_name}/{scene}_traj', exist_ok=True)
nadir_pose = [[1., 0., 0.,  0.0],
              [0., 1., 0.,  0.0],
              [0., 0., 1.,  16]]

w, h = int(1024), int(1024)
cx, cy = w//2, h//2
fx = fy = 15000 # Trial and error adjustment of fx and z height 

K = np.float32([[fx, 0, cx],
                [0, fy, cy],
                [0,  0,  1]])

nadir_directions = get_ray_directions(h, w, torch.FloatTensor(K)) # ngp style
nadir_pose = torch.tensor(nadir_pose)
nadir_pose[:, 1:3] *= -1
nadir = True

if nadir:
    t = time.time()
    rays_o, rays_d = get_rays(nadir_directions.cuda(), nadir_pose.cuda())
    results = render(model, rays_o, rays_d,
                     **{'test_time': True,
                        'T_threshold': 1e-1})
    torch.cuda.synchronize()
    ts += [time.time()-t]
    print(f'Rendered image in {ts}s')

    pred = results['rgb'].reshape(w, h, 3).cpu().numpy()
    pred = (pred*255).astype(np.uint8)
    depth = results['depth'].reshape(w, h).cpu().numpy()
    depth_ = depth2img(depth)
    opacity = results['opacity'].reshape(w, h).cpu().numpy()
    
else:
    for img_idx in [7]:#tqdm(range(len(dataset))):
        t = time.time()
        #rays_o, rays_d = get_rays(dataset.directions[img_idx].cuda(), dataset.poses[img_idx].cuda())
        rays_o, rays_d = get_rays(nadir_directions.cuda(), nadir_pose.cuda())
        results = render(model, rays_o, rays_d,
                        **{'test_time': True,
                            'T_threshold': 1e-2,
                            'exp_step_factor': 1/256})
        torch.cuda.synchronize()
        ts += [time.time()-t]

        pred = results['rgb'].reshape(int(dataset.img_wh[img_idx][1]), int(dataset.img_wh[img_idx][0]), 3).cpu().numpy()
        pred = (pred*255).astype(np.uint8)
        depth = results['depth'].reshape(int(dataset.img_wh[img_idx][1]), int(dataset.img_wh[img_idx][0])).cpu().numpy()
        depth_ = depth2img(depth)
        imgs += [pred]
        depths += [depth_]
        opacity = results['opacity'].reshape(int(dataset.img_wh[img_idx][1]), int(dataset.img_wh[img_idx][0])).cpu().numpy()
    #    imageio.imwrite(f'results/{dataset_name}/{scene}_traj/{img_idx:03d}.png', pred)
    #    imageio.imwrite(f'results/{dataset_name}/{scene}_traj/{img_idx:03d}_d.png', depth_)

        #if dataset.split != 'test_traj':
        #    rgb_gt = dataset[img_idx]['rgb'].cuda()
        #    psnrs += [psnr(results['rgb'], rgb_gt).item()]
            
if psnrs: print(f'mean PSNR: {np.mean(psnrs):.2f}, min: {np.min(psnrs):.2f}, max: {np.max(psnrs):.2f}')
print(f'mean time: {np.mean(ts):.4f} s, FPS: {1/np.mean(ts):.2f}')
print(f'mean samples per ray: {results["total_samples"]/len(rays_d):.2f}')

if len(imgs)>30:
    imageio.mimsave(f'results/{dataset_name}/{scene}_traj/rgb.mp4', imgs, fps=30)
    imageio.mimsave(f'results/{dataset_name}/{scene}_traj/depth.mp4', depths, fps=30)

plt.subplots(figsize=(15, 12))
plt.tight_layout()
plt.subplot(131)
plt.title('pred')
#pred = results['rgb'].reshape(int(dataset.img_wh[img_idx][1]), int(dataset.img_wh[img_idx][0]), 3).cpu().numpy()
plt.imshow(pred)
plt.subplot(132)
plt.title('depth')
#depth = results['depth'].reshape(int(dataset.img_wh[img_idx][1]), int(dataset.img_wh[img_idx][0])).cpu().numpy()
#depth_ = depth2img(depth)
plt.imshow(depth_)
plt.subplot(133)
plt.title('opacity')
plt.imshow(opacity, 'bone')
plt.show()