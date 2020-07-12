from tensorflow import keras

from tensorflow.keras.utils import HDF5Matrix
import numpy as np
import h5py


def extract():
    dataset = h5py.File('./dataset_concat.hdf5', 'w')
    for elem in range(1, 3):
        group = dataset.create_group(f'{elem}')
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
        group.create_dataset(name='images', data= x_data)
        group.create_dataset(name='labels', data= y_data_1)
        dataset['/1/images']
        print(1)

if __name__ == '__main__':
    extract()
    # dataset = h5py.File('./dataset_concat_1.hdf5', 'r')
    # dataset.keys()
    # dd = dataset['/labels']
    # print(1)
    # x_train = dataset['/images_1/'][:][800]
    # y_train = HDF5Matrix('./dataset_concat_1.hdf5', 'labels_1', end=800)
    # x_test = HDF5Matrix('./dataset_concat_1.hdf5', 'images_1', start=800)
    # y_test = HDF5Matrix('./dataset_concat_1.hdf5', 'labels_1', start=800)
    #
    # model = keras.Sequential([
    #     keras.layers.Dense(128, activation='relu'),
    #     keras.layers.Dense(4, activation='softmax')
    # ])
    #
    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model.fit(x_train, y_train, epochs=5)
