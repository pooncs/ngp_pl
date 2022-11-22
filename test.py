import torch
import time
import os
import glob
import numpy as np
from models.networks import NGP
from models.rendering import render
from metrics import psnr
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import dataset_dict
from datasets.ray_utils import get_rays, pose_spherical, get_ray_directions
from utils import load_ckpt
from train import depth2img
import imageio
t0 = time.time()

dataset_name = 'ngp'
scene = 's4_s2'
data_dir = '/home/ubuntu/data/shuttle4'

render_nadir = True
render_spherical = True
render_testset = False
save_spherical_raw = False

################################################## Load images ##################################################
ckpt_path = glob.glob(f'/home/ubuntu/ngp_pl/ckpts/{dataset_name}/{scene}/epoch=*_slim.ckpt')[0]
save_path = f'results/{dataset_name}/{scene}'

dataset = dataset_dict[dataset_name](data_dir, split='test', downsample=1)
model = NGP(scale=0.5).cuda()
print(ckpt_path)
load_ckpt(model, ckpt_path)
print(f'(Load images) Elapsed time: {(time.time()-t0):.2f}s')

################################################## Render images ##################################################
psnrs = []; ts = []; imgs = []; depths = []
os.makedirs(save_path, exist_ok=True)
nadir_pose = [[1., 0., 0.,  0.0],
              [0., 1., 0.,  0.0],
              [0., 0., 1.,  15]]

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

if render_nadir:
    rays_o, rays_d = get_rays(nadir_directions.cuda(), nadir_pose.cuda())
    results = render(model, rays_o, rays_d,
                     **{'test_time': True,
                        'T_threshold': 1e-2,
                        'exp_step_factor': 1/256})

    pred = results['rgb'].reshape(w, h, 3).cpu().numpy()
    pred = (pred*255).astype(np.uint8)
    depth = results['depth'].reshape(w, h).cpu().numpy()
    depth_ = depth2img(depth)
    opacity = results['opacity'].reshape(w, h).cpu().numpy()
    imageio.imwrite(f'{save_path}/nadir_rgb.jpg', pred)
    imageio.imwrite(f'{save_path}/nadir_depth.jpg', depth_)
    imageio.imwrite(f'{save_path}/nadir_opacity.jpg', opacity)
    print(f'(Render nadir images) Elapsed time: {(time.time()-t0):.2f}s')

if render_spherical:
    simgs, sdepths, sopacs = [],[],[]
    ht = nadir_pose[-1][-1]
    poses = pose_spherical(ht)

    for pose in tqdm(poses):
        rays_o, rays_d = get_rays(nadir_directions.cuda(), pose.cuda())
        results_sphere = render(model, rays_o, rays_d,
                            **{'test_time': True,
                            'T_threshold': 1e-2,
                            'exp_step_factor': 1/256})
        #print(f'Rendered spherical pose image {i+1}/{len(poses)} in {time.time()-t}s')
        
        simg = (results_sphere['rgb'].reshape(w, h, 3).cpu().numpy()*255).astype(np.uint8)
        simgs.append(simg)
        sdepth = depth2img(results_sphere['depth'].reshape(w, h).cpu().numpy())
        sdepths.append(sdepth)
        sopac = depth2img(results_sphere['opacity'].reshape(w, h).cpu().numpy())
        sopacs.append(sopac)

        if save_spherical_raw:
            os.makedirs(f'{save_path}/sph_rgb/', exist_ok=True)
            os.makedirs(f'{save_path}/sph_dep/', exist_ok=True)
            os.makedirs(f'{save_path}/sph_opa/', exist_ok=True)
            imageio.imwrite(f'{save_path}/sph_rgb/{i}.jpg', simg)
            imageio.imwrite(f'{save_path}/sph_dep/{i}_d.jpg', sdepth)
            imageio.imwrite(f'{save_path}/sph_opa/{i}_d.jpg', sopac)
    
    imageio.mimsave(f'{save_path}/rgb.mp4', simgs, fps=10)
    imageio.mimsave(f'{save_path}/depth.mp4', sdepths, fps=10)
    imageio.mimsave(f'{save_path}/opacity.mp4', sopacs, fps=10)
    print(f'(Render Spherical images) Elapsed time: {(time.time()-t0):.2f}s')

if render_testset:
    os.makedirs(f'{save_path}/val_rgb/', exist_ok=True)
    os.makedirs(f'{save_path}/val_dep/', exist_ok=True)
    for img_idx in tqdm(range(len(dataset))):
        t = time.time()
        rays_o, rays_d = get_rays(dataset.directions[img_idx].cuda(), dataset.poses[img_idx].cuda())
        results = render(model, rays_o, rays_d,
                        **{'test_time': True,
                            'T_threshold': 1e-2,
                            'exp_step_factor': 1/256})

        pred = results['rgb'].reshape(int(dataset.img_wh[img_idx][1]), int(dataset.img_wh[img_idx][0]), 3).cpu().numpy()
        pred = (pred*255).astype(np.uint8)
        depth = results['depth'].reshape(int(dataset.img_wh[img_idx][1]), int(dataset.img_wh[img_idx][0])).cpu().numpy()
        depth_ = depth2img(depth)
        imgs += [pred]
        depths += [depth_]
        opacity = results['opacity'].reshape(int(dataset.img_wh[img_idx][1]), int(dataset.img_wh[img_idx][0])).cpu().numpy()
        imageio.imwrite(f'{save_path}/val_rgb/{img_idx:03d}.png', pred)
        imageio.imwrite(f'{save_path}/val_dep/{img_idx:03d}_d.png', depth_)

        gb_gt = dataset[img_idx]['rgb'].cuda()
        psnrs += [psnr(results['rgb'], gb_gt).cpu().numpy()]
            
    if psnrs: print(f'mean PSNR: {np.mean(psnrs):.2f}, min: {np.min(psnrs):.2f}, max: {np.max(psnrs):.2f}')
    print(f'mean samples per ray: {results["total_samples"]/len(rays_d):.2f}')
    print(f'(Render test images) Elapsed time: {(time.time()-t0):.2f}s')

'''


Mesh generation



from kornia.utils.grid import create_meshgrid3d
import vren

xyz = create_meshgrid3d(model.grid_size, model.grid_size, model.grid_size, False, dtype=torch.int32).reshape(-1, 3)
# _density_bitfield = model.density_bitfield
# density_bitfield = torch.zeros(model.cascades*model.grid_size**3//8, 8, dtype=torch.bool)
# for i in range(8):
#     density_bitfield[:, i] = _density_bitfield & torch.tensor([2**i], device='cuda')
# density_bitfield = density_bitfield.reshape(model.cascades, model.grid_size**3).cpu()
indices = vren.morton3D(xyz.cuda()).long()
import mcubes
import trimesh

### Tune these parameters until the whole object lies tightly in range with little noise ###
N = 128 # controls the resolution, set this number small here because we're only finding
        # good ranges here, not yet for mesh reconstruction; we can set this number high
        # when it comes to final reconstruction.
xmin, xmax = -0.5, 0.5 # left/right range
ymin, ymax = -0.5, 0.5 # forward/backward range
zmin, zmax = -0.5, 0.5 # up/down range
## Attention! the ranges MUST have the same length!
sigma_threshold = 20. # controls the noise (lower=maybe more noise; higher=some mesh might be missing)
############################################################################################

x = np.linspace(xmin, xmax, N)
y = np.linspace(ymin, ymax, N)
z = np.linspace(zmin, zmax, N)

xyz = torch.FloatTensor(np.stack(np.meshgrid(x, y, z), -1).reshape(-1, 3)).cuda()

with torch.no_grad():
    sigma = model.density(xyz).cpu().numpy().astype(np.float32)

sigma = sigma.reshape(N, N, N)
# The below lines are for visualization, COMMENT OUT once you find the best range and increase N!
vertices, triangles = mcubes.marching_cubes(sigma, sigma_threshold)
mesh = trimesh.Trimesh(vertices/N, triangles)
#mesh.show()

'''


#Plot rays


'''
from datasets.ray_utils import get_rays
import plotly.graph_objects as go
from collections import defaultdict

fig = go.Figure()

xlines = []; ylines = []; zlines = []
for ip in range(len(dataset.poses)):
    # cameras
    # TODO: add axes
    fx, fy, cx, cy = dataset.K[ip, 0, 0], dataset.K[ip, 1, 1], dataset.K[ip, 0, 2], dataset.K[ip, 1, 2]
    u = torch.FloatTensor([0, dataset.img_wh[ip][0]-1, dataset.img_wh[ip][0]-1, 0])
    v = torch.FloatTensor([0, 0, dataset.img_wh[ip][1]-1, dataset.img_wh[ip][1]-1])
    ds = torch.stack([(u-cx+0.5)/fx, (v-cy+0.5)/fy, torch.ones_like(u)], -1)

    o, d = get_rays(ds, torch.FloatTensor(dataset.poses[ip]))
    o = o.numpy(); d = d.numpy()/10
    od = o+d
    xlines += [od[0, 0], od[1, 0], od[2, 0], od[3, 0], od[0, 0], None]
    ylines += [od[0, 1], od[1, 1], od[2, 1], od[3, 1], od[0, 1], None]
    zlines += [od[0, 2], od[1, 2], od[2, 2], od[3, 2], od[0, 2], None]
    for i in range(4):
        xlines += [o[i, 0], od[i, 0], None]
        ylines += [o[i, 1], od[i, 1], None]
        zlines += [o[i, 2], od[i, 2], None]
fig.add_trace(
    go.Scatter3d(
        x=xlines,
        y=ylines,
        z=zlines,
        mode='lines',
        name='cameras',
        marker=dict(size=1, color='black')
    )
)

# RAYS
try:
    xlines = []; ylines = []; zlines = []
    rays_ = dataset.rays.cpu().numpy()
    for i in range(len(rays)):
        xlines += [rays_[i, 0], rays_[i, 0]+3*rays_[i, 3], None]
        ylines += [rays_[i, 1], rays_[i, 1]+3*rays_[i, 4], None]
        zlines += [rays_[i, 2], rays_[i, 2]+3*rays_[i, 5], None]

    fig.add_trace(
        go.Scatter3d(
            x=xlines,
            y=ylines,
            z=zlines,
            mode='lines',
            name='rays',
            marker=dict(size=1, color='green')
        )
    )
    xs = []; ys = []; zs = []
    for i in range(len(rays)):
        ray_idx, start_idx, N_samples = results['rays_a'][i].cpu().numpy()
        for s in range(start_idx, start_idx+N_samples):
            t = results['ts'][s].cpu().numpy()
            xs += [rays_[ray_idx, 0]+t*rays_[ray_idx, 3]]
            ys += [rays_[ray_idx, 1]+t*rays_[ray_idx, 4]]
            zs += [rays_[ray_idx, 2]+t*rays_[ray_idx, 5]]

    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode='markers',
            name='sample points',
            marker=dict(size=2.5, color='blue'),
        )
    )
except: 
    print('Plot rays failed')
    pass

try:
    fig.add_trace(
        go.Scatter3d(
            x=dataset.pts3d[:, 0],
            y=dataset.pts3d[:, 1],
            z=dataset.pts3d[:, 2],
            mode='markers',
            name='scene pts',
            marker=dict(size=0.4, color='red'),
        )
    )
except: 
    print('Plot 3D failed')
    pass

for ca in range(model.cascades):
    s = min(2**(ca-1), model.scale)
    xlines = [s, s, s, s, s, None, -s, -s, -s, -s, -s, None]
    ylines = [-s, -s, s, s, -s, None, -s, -s, s, s, -s, None]
    zlines = [s, -s, -s, s, s, None, s, -s, -s, s, s, None]
    xlines += [s, -s, None, s, -s, None, s, -s, None, s, -s, None]
    ylines += [-s, -s, None, -s, -s, None, s, s, None, s, s, None]
    zlines += [s, s, None, -s, -s, None, -s, -s, None, s, s, None]
    fig.add_trace(
        go.Scatter3d(
            x=xlines,
            y=ylines,
            z=zlines,
            mode='lines',
            name=f'bbox {ca}',
            marker=dict(size=1, color='orange')
        )
    )

try:
    m = defaultdict(list)
    cube_colors = ['lightgray', 'lightcyan', 'magenta']
    for ca in range(model.cascades):
        s = min(2**(ca-1), model.scale)
        a = density_bitfield[ca, indices]
        xyz_ = xyz[a]
        if len(xyz_)==0: continue
        for i in tqdm(range(len(xyz_))):
            hs = s/model.grid_size
            c = (xyz_[i].numpy()/(model.grid_size-1)*2-1)*(s-hs)
            m['x'] += [(np.array([0, 0, 1, 1, 0, 0, 1, 1])-0.5)*2*hs+c[0]]
            m['y'] += [(np.array([0, 1, 1, 0, 0, 1, 1, 0])-0.5)*2*hs+c[1]]
            m['z'] += [(np.array([0, 0, 0, 0, 1, 1, 1, 1])-0.5)*2*hs+c[2]]
            m['i'] += [np.array([7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2])+len(m['i'])*8]
            m['j'] += [np.array([3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3])+len(m['j'])*8]
            m['k'] += [np.array([0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6])+len(m['k'])*8]
        fig.add_trace(
            go.Mesh3d(
                x=np.concatenate(m['x']),
                y=np.concatenate(m['y']),
                z=np.concatenate(m['z']),
                i=np.concatenate(m['i']),
                j=np.concatenate(m['j']),
                k=np.concatenate(m['k']),
                color=cube_colors[ca],
                name=f'occupied cells {ca}',
                showlegend=True,
                opacity=0.4**(ca+1)
            )
        )
except: 
    print('Plot Cube failed')
    pass

layout = go.Layout(scene=dict(aspectmode='data'), dragmode='orbit')
fig.update_layout(layout)

fig.show()

'''