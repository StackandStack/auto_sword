import h5py
import numpy as np

if __name__ == '__main__':
    extended_list = list()
    extended_y = list()

    file = h5py.File(f'./dataset.hdf5', 'w')
    for elem in range(1, 3):
        data = np.load(f'testset_{elem}.npy', allow_pickle='TRUE')
        extended_list.extend(
            np.array([value['dataset']['x'][0] / 255.0 for key, value in enumerate(data)], dtype='float64'))
        extended_list.extend(
            np.array([value['dataset']['x'][1] / 255.0 for key, value in enumerate(data)], dtype='float64'))
        extended_y.extend(np.array([value['dataset']['y'][0] for key, value in enumerate(data)]))
        extended_y.extend(np.array([value['dataset']['y'][1] for key, value in enumerate(data)]))

    x_data = np.stack(extended_list, axis=0)
    y_data = np.array(extended_y)
    file.create_dataset(name='images', data=x_data)
    file.create_dataset(name='labels', data=y_data)
