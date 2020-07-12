from tensorflow import keras as tf
from tensorflow.keras.utils import HDF5Matrix
import numpy as np
import h5py

if __name__ == '__main__':
    for elem in range(1, 3):
        dataset = h5py.File(f'./dataset_concat_{elem}.hdf5', 'w')
        f = h5py.File(f'./dataset_{elem}.hdf5', 'r')
        sort_keys = list(f.keys())
        x_stack_list_1 = list()
        x_stack_list_2 = list()
        y_data_1 = list()
        y_data_2 = list()
        for key in sort_keys:
            x_stack_list_1.append(f[f'/{key}/x_1'][:].flatten() / 255.0)
            x_stack_list_2.append(f[f'/{key}/x_2'][:].flatten() / 255.0)
            y_data_1.append(f.get(f'/{key}/y_1')[()])
            y_data_2.append(f.get(f'/{key}/y_2')[()])
            print(key)
        x_data = np.vstack((x_stack_list_1, x_stack_list_2))
        y_data_1.extend(y_data_2)
        dataset.create_dataset(f'images_{elem}', data=x_data)
        dataset.create_dataset(f'labels_{elem}', data=y_data_1)
