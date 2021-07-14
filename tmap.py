# -*- coding: utf-8 -*-
"""
tmap.py

Module for performing tortuosity estimation calculations on 3D volumetric
datasets. Estimates tortuosity by simulating turbulence-free flow of a virtual
fluid through the input sampled space. Algorithm inspired by and based on
the one described by Chen-Wiegart et al in:
    http://dx.doi.org/10.1016/j.jpowsour.2013.10.026

Created on Wed Jul  8 17:15:34 2020

@author: tyler
Do it for the Vine!
"""

import numpy as np
import matplotlib.pyplot as plt
import time

def euclidean_distance(a,b):
    x1, y1, z1 = a
    x2, y2, z2 = b
    d = np.sqrt(((x1-x2)**2) + ((y1-y2)**2) + ((z1-z2)**2))
    return d

class TMap:

    def __init__(self, arr, seed=None, step=(1,1,1)):
        self.prop_front = []
        self.front_index = 1
        self.volume = arr
        self.shape = arr.shape
        z, y, x = self.shape
        self.fluid = np.zeros_like(self.volume, dtype='float32')
        self.tortuosity = None
        self.step_size = step
        self.seeds = []

        # self.realspace_map = np.zeros(shape=(z,y,x,3),dtype='float16')  #Map of coordinate space to real space for datasets with uneven step sizes
        # for i in range(z):
        #     for j in range(y):
        #         for k in range(x):
        #             self.realspace_map[i,j,k] = [i*self.step_size[0], j*self.step_size[1], k*self.step_size[2]]

        if seed is None:
            self.seeds = [self._autoseed_()]
        else:
            if type(seed) is list:
                self.seeds = seed
            elif type(seed) is str:
                acceptable_strings = ['random','center','x','-x','y','-y','z','-z']
                if seed == 'random':
                    #If seed is 'random', randomly generate coordinate tuples
                    #until an empty voxel is found, use as intiial seed
                    is_empty = False
                    while is_empty == False:
                        x, y, z = self.shape
                        a, b, c = [np.random.randint(0,i) for i in [x, y, z]]
                        if self.volume[a,b,c] == 0:
                            is_empty = True
                            self.seeds = [(a,b,c)]
                elif seed == 'center':
                    self.seeds = [self._autoseed_()]
                elif seed == 'x':
                    for i in range(y):
                        for j in range(z):
                            self.seeds.append((j,i,0))
                elif seed == '-x':
                    for i in range(y):
                        for j in range(z):
                            self.seeds.append((j,i,x-1))
                elif seed == 'y':
                    for i in range(x):
                        for j in range(z):
                            self.seeds.append((j,0,i))
                elif seed == '-y':
                    for i in range(x):
                        for j in range(z):
                            self.seeds.append((j,y-1,i))
                elif seed == 'z':
                    for i in range(x):
                        for j in range(y):
                            self.seeds.append((0,i,j))
                elif seed == '-z':
                    for i in range(x):
                        for j in range(y):
                            self.seeds.append((z-1,i,j))

        for seed in self.seeds:
            self.set_seed(seed)
        return

    def _autoseed_(self):

        def proxy_sort(template, data):
            order = np.argsort(template)
            sorted_data = [data[i] for i in order]
            return sorted_data

        x, y, z = self.fluid.shape
        center_of_volume = (x/2,y/2,z/2)
        points = []
        for a in range(x):
            for b in range(y):
                for c in range(z):
                    points.append((a, b, c))
        distances = [euclidean_distance(p, center_of_volume) for p in points]
        points = proxy_sort(distances, points)
        for a,b,c in points:
            if self.volume[a,b,c] == 0:
                return (a,b,c)
        print('Warning! No suitable autoseed location could be found!')
        print('Please manually specify a fluid seed location!')
        return

    def set_seed(self, location):
        a, b, c = location
        if self.volume[a,b,c] == 1:
            # print('Voxel at (%i, %i, %i) is full and cannot be used as a seed.'%(a,b,c))
            return
        self.fluid[a, b, c] = self.front_index
        self.prop_front.append((a,b,c))
        return

    def step(self, mode='chessboard'):

        approved_modes = ('chessboard', 'block', 'euclidean')
        try:
            assert mode in approved_modes
        except AssertionError:
            raise ValueError('%s not recognized as an approved mode!')

        front_copy = list(self.prop_front)
        self.front_index += 1

        test_coords = set()
        vshape = self.volume.shape
        for x, y, z in front_copy:
            if mode == 'block':
                test_coords |= set([

                # Coordinate axes
                (x+1, y, z),
                (x-1, y, z),
                (x, y+1, z),
                (x, y-1, z),
                (x, y, z+1),
                (x, y, z-1)

                ])
            else:
                test_coords |= set([

                # Coordinate axes
                (x+1, y, z),
                (x-1, y, z),
                (x, y+1, z),
                (x, y-1, z),
                (x, y, z+1),
                (x, y, z-1),

                # In-plane diagonals
                (x+1,y+1,z),
                (x+1,y-1,z),
                (x-1,y+1,z),
                (x+1,y-1,z),

                # z+1 diagonals
                (x + 1, y + 1, z + 1),
                (x - 1, y + 1, z + 1),
                (x + 1, y - 1, z + 1),
                (x - 1, y - 1, z + 1),
                (x + 1, y, z + 1),
                (x - 1, y, z + 1),
                (x, y - 1, z + 1),
                (x, y + 1, z + 1),

                # z-1 diagonals
                (x + 1, y + 1, z - 1),
                (x - 1, y + 1, z - 1),
                (x + 1, y - 1, z - 1),
                (x - 1, y - 1, z - 1),
                (x + 1, y, z - 1),
                (x - 1, y, z - 1),
                (x, y - 1, z - 1),
                (x, y + 1, z - 1)

                ])

            self.prop_front.remove((x,y,z))

        for a, b, c in list(test_coords):

            if any([
                a<0, b<0, c<0,
                a>=vshape[0], b>=vshape[1], c>=vshape[2]
                    ]):
                test_coords.remove((a,b,c))

        for a, b, c in test_coords:

            volume_value = self.volume[a,b,c]
            tort_value = self.fluid[a,b,c]
            distances = []

            if volume_value==0 and tort_value==0:

                self.prop_front.append((a,b,c))

                for x, y, z in front_copy:
                    
                    if any([a < x-1, a > x + 1, b < y-1, b > y+1, c<z-1, c>z+1]): continue

                    if mode=='euclidean' or mode=='block':
                        realspace_coordinates1 = [x*self.step_size[0], y*self.step_size[1], z*self.step_size[2]]
                        realspace_coordinates2 = [a*self.step_size[0], b*self.step_size[1], c*self.step_size[2]]
                        distances.append(self.fluid[x,y,z] + euclidean_distance(realspace_coordinates1, realspace_coordinates2))
                    else:
                        distances.append(self.front_index)

                self.fluid[a,b,c] = min(distances)

        return

    def multistep(self, n_steps):
        for i in range(n_steps):
            self.step()
        return

    def fill(self, mode='euclidean'):
        old_sum = 0
        new_sum = 1
        while new_sum > old_sum:
            old_sum = float(new_sum)
            self.step(mode=mode)
            new_sum = np.sum(self.fluid)
        return

    def calc_tortuosity(self, mode='euclidean', faces_only=False):
        zz,xx,yy = self.shape
        comp_map = TMap(np.zeros_like(self.fluid), seed=self.seeds, step=self.step_size)
        comp_map.fill(mode=mode)
        tort_map = np.zeros_like(self.fluid, dtype=np.float32)
        for x, plane in enumerate(tort_map):
            for y, row in enumerate(plane):
                for z, cell in enumerate(row):
                    tort_map[x, y, z] = self.fluid[x, y, z] / comp_map.fluid[x, y, z]
        # self.tortuosity = tort_map

        accessible_voxels = tort_map[np.where(tort_map > 0)].flatten()

        if faces_only:
            for i, plane in enumerate(tort_map):
                if i == 0 or i == zz-1:
                    face_voxels = plane.flatten()
                else:
                    face_voxels = plane[0].flatten()
                    face_voxels = np.hstack((face_voxels, plane[-1].flatten()))
                    face_voxels = np.hstack((face_voxels, plane[0,1:-1].flatten()))
                    face_voxels = np.hstack((face_voxels, plane[-1,1:-1].flatten()))
            accessible_face_voxels = face_voxels[np.where(face_voxels > 0)]
            tortuosity = np.mean(accessible_face_voxels)
        else:
            tortuosity = np.mean(accessible_voxels)

        return tortuosity, tort_map

    def save(self, fname_prefix):
        seeds = np.zeros((len(self.seeds),3))
        for i, (z, y, x) in enumerate(self.seeds):
            seeds[i] = [z,y,x]
        np.save(fname_prefix+'_fluid.npy', self.fluid)
        np.save(fname_prefix+'_solid.npy', self.volume)
        np.save(fname_prefix+'_seeds.npy', seeds)
        # if self.tortuosity is not None: np.save(fname_prefix+'_tortuosity.npy', self.tortuosity)
        return

    def show(self, mode='tortuosity', title=None, showseed=True):
        zz,yy,xx = self.shape

        #build seed map
        seedmap = np.zeros_like(self.volume)
        for z, y, x in self.seeds:
            seedmap[z,y,x] = 1

        # if mode == 'tortuosity':
        #     mappable = self.tortuosity
        if mode == 'fluid':
            mappable = self.fluid
        for i in range(zz):
            plt.figure()
            plt.imshow(mappable[i], cmap='Blues', vmin=1, vmax=np.max(mappable))
            plt.colorbar()
            plt.imshow(self.volume[i], cmap='Reds', alpha=0.4)
            # plt.imshow(seedmap[i],cmap='Greens',alpha=0.5)
            if showseed:
                for z, y, x  in self.seeds:
                    if z == i:
                        plt.scatter(x,y,c='g')

            if title is not None:
                title_string = title + ': Layer %i'%(i+1)
                plt.title(title_string)

if __name__ == "__main__":
    base = np.load('base.npy')

    tmap = TMap(base, step=(5,10,10))
    tmap.fill('euclidean')
    tortuosity, tortuosity_map = tmap.calc_tortuosity('euclidean')
    tmap.show(title='Isotropic')
    print('Isotropic tortuosity: %f'%(tortuosity))

    tortuosities = []
    for i in range(25):
        tmap = TMap(base, step=(5,10,10), seed='random')
        tmap.fill('euclidean')
        tortuosity, tortuosity_map = tmap.calc_tortuosity('euclidean')
        # tmap.show(title='Isotropic')
        tortuosities.append(tortuosity)

    print('Randomized Isotropic tortuosity: %f'%(np.median(tortuosities)))

    seeds = []
    for i in range(25):
        for j in range(25):
            seeds.append((0,i,j))
    tmap = TMap(base, seed=seeds, step=(5,10,10))
    tmap.fill('euclidean')
    tortuosity, tortuosity_map = tmap.calc_tortuosity('euclidean')
    tmap.show(title='Ansotropic Z')
    print('Ansotropic tortuosity Z: %f'%(tortuosity))

    seeds = []
    for i in range(25):
        for j in range(4):
            seeds.append((j,0,i))
    tmap = TMap(base, seed=seeds, step=(5,10,10))
    tmap.fill('euclidean')
    tortuosity, tortuosity_map = tmap.calc_tortuosity('euclidean')
    tmap.show(title='Ansotropic Y')
    print('Ansotropic tortuosity Y: %f'%(tortuosity))

    seeds = []
    for i in range(25):
        for j in range(4):
            seeds.append((j,i,0))
    tmap = TMap(base, seed=seeds, step=(5,10,10))
    tmap.fill('euclidean')
    tortuosity, tortuosity_map = tmap.calc_tortuosity('euclidean')
    tmap.show(title='Ansotropic X')
    print('Ansotropic tortuosity X: %f'%(tortuosity))
