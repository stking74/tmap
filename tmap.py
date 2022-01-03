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

import os
import time

import numpy as np
import matplotlib.pyplot as plt
import h5py

def prop_voxel(coords, volume_value, tort_value, step_size, front_index, front, mode):

    front_copy, front_values = front
    new_front_coords = []
    a,b,c = coords
    distances = []

    if volume_value==0 and tort_value==0:

        new_front_coords.append((a,b,c))

        for i, (x, y, z) in enumerate(front_copy):

            if mode=='euclidean' or mode=='block':
                realspace_coordinates1 = [x*step_size[0], y*step_size[1], z*step_size[2]]
                realspace_coordinates2 = [a*step_size[0], b*step_size[1], c*step_size[2]]
                distances.append(front_values[i] + euclidean_distance(realspace_coordinates1, realspace_coordinates2))
            else:
                distances.append(front_index)

        return min(distances), new_front_coords

def euclidean_distance(a,b):
    x1, y1, z1 = a
    x2, y2, z2 = b
    d = np.sqrt(((x1-x2)**2) + ((y1-y2)**2) + ((z1-z2)**2))
    return d

class TMap:

    def __init__(self, arr, seed=None, step=(1,1,1)):
        '''
        Input array serves as a mask to discriminate pore volume from solid material.
        Expects boolean array, with all solid voxels flagged as True or 1 and all pore voxels flagged as False or 0.
        '''
        self.prop_front = []
        self.front_index = 1
        self.volume = arr
        self.shape = arr.shape
        z, y, x = self.shape
        self.fluid = np.zeros_like(self.volume, dtype=np.float32)
        self.tortuosity = None
        self.step_size = step
        self.seeds = []
        self.seedmode = None

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
                assert seed in acceptable_strings
                self.seedmode = seed
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
                            self.seeds.append((0,j,i))
                elif seed == '-z':
                    for i in range(x):
                        for j in range(y):
                            self.seeds.append((z-1,j,i))

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

        vvals = []
        tvals = []
        for a, b, c in list(test_coords):

            if any([
                a<0, b<0, c<0,
                a>=self.volume.shape[0], b>=self.volume.shape[1], c>=self.volume.shape[2]
                    ]):
                test_coords.remove((a,b,c))
                continue

            volume_value = self.volume[a,b,c]
            tort_value = self.fluid[a,b,c]
            if volume_value!=0 or tort_value!=0:
                test_coords.remove((a,b,c))
                continue
            vvals.append(volume_value)
            tvals.append(tort_value)

        for a, b, c in test_coords:

            distances = []

            self.prop_front.append((a,b,c))

            for x, y, z in front_copy:

                test_conditions = [
                    x < a - 1,
                    x > a + 1,
                    y < b - 1,
                    y > b + 1,
                    z < c - 1,
                    z > c + 1
                ]

                if any(test_conditions):
                    continue

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

    def fill(self, mode='euclidean', save_q=0, fname=None):

        if save_q > 0:
            save_count = 0
            assert fname is not None, 'A filename must be specified to save an in-progress calculation'

        old_sum = np.sum(self.fluid)
        new_sum = old_sum + 1
        step_count = 0
        while new_sum > old_sum:
            if save_q > 0:
                if step_count % save_q == 0:
                    qfname = fname[:-3]+'_%i.h5'%(save_count)
                    self.save(qfname)
                    save_count += 1
            old_sum = float(new_sum)
            ti = time.time()
            self.step(mode=mode)
            new_sum = np.sum(self.fluid)
            print('Step %i:\nNewSum:%i\nCalcTime:%i'%(step_count, new_sum, time.time()-ti))
            step_count += 1


        return

    # def fill_mp(self, mode='euclidean', n_cores=2):

    #     def step_mp(mode='chessboard'):

    #         pool = Pool(processes=n_cores)

    #         approved_modes = ('chessboard', 'block', 'euclidean')
    #         try:
    #             assert mode in approved_modes
    #         except AssertionError:
    #             raise ValueError('%s not recognized as an approved mode!')

    #         front_copy = list(self.prop_front)
    #         front_values = [self.fluid[x,y,z] for x,y,z in self.prop_front]
    #         front = (front_copy, front_values)
    #         self.front_index += 1

    #         test_coords = set()
    #         for x, y, z in front_copy:
    #             if mode == 'block':
    #                 test_coords |= set([

    #                 # Coordinate axes
    #                 (x+1, y, z),
    #                 (x-1, y, z),
    #                 (x, y+1, z),
    #                 (x, y-1, z),
    #                 (x, y, z+1),
    #                 (x, y, z-1)

    #                 ])
    #             else:
    #                 test_coords |= set([

    #                 # Coordinate axes
    #                 (x+1, y, z),
    #                 (x-1, y, z),
    #                 (x, y+1, z),
    #                 (x, y-1, z),
    #                 (x, y, z+1),
    #                 (x, y, z-1),

    #                 # In-plane diagonals
    #                 (x+1,y+1,z),
    #                 (x+1,y-1,z),
    #                 (x-1,y+1,z),
    #                 (x+1,y-1,z),

    #                 # z+1 diagonals
    #                 (x + 1, y + 1, z + 1),
    #                 (x - 1, y + 1, z + 1),
    #                 (x + 1, y - 1, z + 1),
    #                 (x - 1, y - 1, z + 1),
    #                 (x + 1, y, z + 1),
    #                 (x - 1, y, z + 1),
    #                 (x, y - 1, z + 1),
    #                 (x, y + 1, z + 1),

    #                 # z-1 diagonals
    #                 (x + 1, y + 1, z - 1),
    #                 (x - 1, y + 1, z - 1),
    #                 (x + 1, y - 1, z - 1),
    #                 (x - 1, y - 1, z - 1),
    #                 (x + 1, y, z - 1),
    #                 (x - 1, y, z - 1),
    #                 (x, y - 1, z - 1),
    #                 (x, y + 1, z - 1)

    #                 ])

    #             self.prop_front.remove((x,y,z))

    #         for a, b, c in list(test_coords):

    #             if any([
    #                 a<0, b<0, c<0,
    #                 a>=self.volume.shape[0], b>=self.volume.shape[1], c>=self.volume.shape[2]
    #                     ]):
    #                 test_coords.remove((a,b,c))

    #         param_list = [((a,b,c), self.volume[a,b,c], self.fluid[a,b,c], self.step_size, self.front_index, front, mode) for (a,b,c) in test_coords]

    #         test_params = param_list[7]
    #         test_result = prop_voxel(test_params[0],test_params[1],test_params[2],test_params[3],test_params[4],test_params[5],test_params[6])

    #         results = pool.starmap(prop_voxel, param_list)

    #         pool.close()
    #         pool.join()

    #         new_front_coords = set()

    #         for i, (a,b,c) in enumerate(test_coords):
    #             result = results[i]
    #             if result is not None:
    #                 distance, new_front = results[i]
    #                 self.fluid[a,b,c] = distance
    #                 for c in new_front:
    #                     new_front_coords.add(c)

    #         self.prop_front += list(new_front_coords)

    #         return

    #     from multiprocessing import Pool



    #     old_sum = 0
    #     new_sum = 1
    #     while new_sum > old_sum:
    #         old_sum = float(new_sum)
    #         step_mp(mode=mode)
    #         new_sum = np.sum(self.fluid)
    #     return

    def calc_tortuosity(self, mode='euclidean', faces_only=False, reference=None):
        zz,yy,xx = self.shape

        if reference is None:
            #Build empty reference

            if self.seedmode=='z':
                comp_map = np.zeros_like(self.fluid)
                for i in range(comp_map.shape[0]):
                    comp_map[i,:,:] = i+1
            elif self.seedmode=='-z':
                comp_map = np.zeros_like(self.fluid)
                for i in range(comp_map.shape[0]):
                    i+=1
                    comp_map[-i,:,:] = i
            elif self.seedmode=='y':
                comp_map = np.zeros_like(self.fluid)
                for i in range(comp_map.shape[1]):
                    comp_map[:,i,:] = i+1
            elif self.seedmode=='-y':
                comp_map = np.zeros_like(self.fluid)
                for i in range(comp_map.shape[1]):
                    comp_map[:,-i,:] = i+1
            elif self.seedmode=='x':
                comp_map = np.zeros_like(self.fluid)
                for i in range(comp_map.shape[2]):
                    comp_map[:,:,i] = i+1
            elif self.seedmode=='-x':
                comp_map = np.zeros_like(self.fluid)
                for i in range(comp_map.shape[2]):
                    comp_map[:,:,-i] = i+1
            else:
                comp_map = TMap(np.zeros_like(self.fluid), seed=self.seeds, step=self.step_size)
                comp_map.fill(mode=mode)
                comp_map = comp_map.fluid
        else: comp_map = reference

        tort_map = np.zeros_like(self.fluid, dtype=np.float32)
        for z, plane in enumerate(tort_map):
            for y, row in enumerate(plane):
                for x, cell in enumerate(row):
                    tort_map[z, y, x] = self.fluid[z, y, x] / comp_map[z, y, x]
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

    def save(self, fname):
        '''
        Saves tortuosity mapper to disk as HDF5 file.
        '''

        #Check if file already exists
        fname = os.path.abspath(fname)
        if os.path.isfile(fname):
            hf = h5py.File(fname, 'r+')
        else:
            hf = h5py.File(fname, 'w')
            hf.create_dataset(name='base', dtype=int, data=self.volume)
            hf.create_dataset(name='fluid', dtype=np.float32, shape=self.shape)
            if self.tortuosity is not None:
                hf.create_dataset('tortmap', dtype=np.float32, shape=self.shape)
            hf.create_dataset('seeds', data=self.seeds)

        hf['fluid'][:] = self.fluid[:]
        hf['fluid'].attrs.create('front', np.array(self.prop_front))
        if self.tortuosity is not None:
            hf['tortmap'][:] = self.tortuosity[:]

        hf.close()
        return

    @staticmethod
    def load(fname):
        pass

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

    ti = time.time()
    tmap = TMap(base, step=(5,5,5))
    tmap.fill('chessboard', fname='testfile.h5', save_q=0)
    tortuosity, tortuosity_map = tmap.calc_tortuosity('chessboard')
    dt = time.time() - ti
    tmap.show(title='Isotropic', mode='fluid')
    print('Isotropic tortuosity: %f'%(tortuosity))
    print('Calculated in %f seconds'%(dt))

    # tortuosities = []
    # for i in range(25):
    #     tmap = TMap(base, step=(5,10,10), seed='random')
    #     tmap.fill('euclidean')
    #     tortuosity, tortuosity_map = tmap.calc_tortuosity('euclidean')
    #     # tmap.show(title='Isotropic')
    #     tortuosities.append(tortuosity)

    # print('Randomized Isotropic tortuosity: %f'%(np.median(tortuosities)))

    # seeds = []
    # for i in range(25):
    #     for j in range(25):
    #         seeds.append((0,i,j))
    # tmap = TMap(base, seed=seeds, step=(5,10,10))
    # tmap.fill('euclidean')
    # tortuosity, tortuosity_map = tmap.calc_tortuosity('euclidean')
    # tmap.show(title='Ansotropic Z')
    # print('Ansotropic tortuosity Z: %f'%(tortuosity))

    # seeds = []
    # for i in range(25):
    #     for j in range(4):
    #         seeds.append((j,0,i))
    # tmap = TMap(base, seed=seeds, step=(5,10,10))
    # tmap.fill('euclidean')
    # tortuosity, tortuosity_map = tmap.calc_tortuosity('euclidean')
    # tmap.show(title='Ansotropic Y')
    # print('Ansotropic tortuosity Y: %f'%(tortuosity))

    # seeds = []
    # for i in range(25):
    #     for j in range(4):
    #         seeds.append((j,i,0))
    # tmap = TMap(base, seed=seeds, step=(5,10,10))
    # tmap.fill('euclidean')
    # tortuosity, tortuosity_map = tmap.calc_tortuosity('euclidean')
    # tmap.show(title='Ansotropic X')
    # print('Ansotropic tortuosity X: %f'%(tortuosity))
