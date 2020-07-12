import h5py
import numpy as np

if __name__ == '__main__':
    file = h5py.File('./dataset_2.hdf5', 'w')
    data = np.load('testset_2.npy', allow_pickle='TRUE')
    for key, value in enumerate(data):
        group = file.create_group(f'{key}')
        group.create_dataset('x_1', data=value['dataset']['x'][0])
        group.create_dataset('x_2', data=value['dataset']['x'][1])
        group.create_dataset('y_1', data=value['dataset']['y'][0])
        group.create_dataset('y_2', data=value['dataset']['y'][1])
        print(key)

