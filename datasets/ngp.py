import torch
import glob
import json
import numpy as np
import os
from tqdm import tqdm

from .ray_utils import get_ray_directions
from .color_utils import read_image

from .base import BaseDataset


class ngpDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        if kwargs.get('read_meta', True):
            self.read_meta(split)
        else:
            # For GUI
            with open(os.path.join(self.root_dir, 'transforms.json'), 'r') as f:
                meta = json.load(f)
                frame = [meta['frames'][0]]
                
                if 'aabb' in meta:
                    #self.shift = np.array(aabb['scene_center_3d_box'])
                    aabb = meta['aabb']
                    self.scale = max(abs(np.array(aabb[1]))+
                                abs(np.array(aabb[0])))
                    print(f'Scale from aabb: {self.scale}')
                elif 'scale' in meta:
                    self.scale = meta['scale']
                    print(f'Scale from scale: {self.scale}')
                else:
                    self.scale = 1
                    print(f'Default scale: {self.scale}')
            
            #self.shift = np.array(aabb['scene_center_3d_box'])
            self.K, self.directions, self.img_wh = self.get_intrinsics(frame[0])
    
    def get_intrinsics(self, frame):
        
        fx = frame['fl_x'] * self.downsample
        fy = frame['fl_y'] * self.downsample
        cx = frame['cx'] * self.downsample
        cy = frame['cy'] * self.downsample
        w = int(frame['w']*self.downsample)
        h = int(frame['h']*self.downsample)
        K = np.float32([[fx, 0, cx],
                        [0, fy, cy],
                        [0,  0,  1]])
        directions = [get_ray_directions(h, w, torch.FloatTensor(K))]
        img_wh = (w, h)
        return K, directions, img_wh

    def read_meta(self, split):
        self.rays = []
        self.poses = []
        self.K = []
        self.img_wh = []
        self.directions = []

        with open(os.path.join(self.root_dir, 'transforms.json'), 'r') as f:
            meta = json.load(f)
        
            if 'aabb' in meta:
                #self.shift = np.array(aabb['scene_center_3d_box'])
                aabb = meta['aabb']
                self.scale = max(abs(np.array(aabb[1]))+
                            abs(np.array(aabb[0])))
                print(f'Scale from aabb: {self.scale}')
            elif 'scale' in meta:
                self.scale = meta['scale']
                print(f'Scale from scale: {self.scale}')
            else:
                self.scale = 1
                print(f'Default scale: {self.scale}')

            for frame in tqdm(meta['frames']):
                K, directions, img_wh = self.get_intrinsics(frame)
                self.K += [K]
                self.directions += directions
                self.img_wh += [img_wh]

                c2w = np.array(frame['transform_matrix'])[:3, :4]
                c2w[:, 1:3] *= -1 # [right up back] to [right down front]
                #pose_radius_scale = 1
                c2w[:, 3] /= np.linalg.norm(c2w[:, 3]) #/pose_radius_scale
                self.poses += [c2w]

                image_path = os.path.join(self.root_dir, frame['file_path'][2:])
                #self.image_paths += [image_path]
                img = read_image(image_path, img_wh)
                self.rays += [img]
            
            if len(self.rays)>0:
                self.rays = torch.FloatTensor(np.stack(self.rays)) # (N_images, hw, rgb)
            
            self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
            #self.poses[..., 3] /= np.linalg.norm(self.poses[..., 3])
            '''aabb_range = self.poses[..., 3].max(0)[0].abs() + \
                        self.poses[..., 3].min(0)[0].abs()
            self.poses[..., 3] /= (aabb_range[:2].max()/2)'''
            self.poses[..., 3] *= self.scale

        self.K = torch.FloatTensor(self.K) # (N_images, 3, 3)
        self.directions = torch.stack(self.directions) # (N_images, hw, rgb)
        self.img_wh = torch.FloatTensor(self.img_wh) # (N_images, 2)


