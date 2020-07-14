import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image


# from tensorflow.keras.utils import HDF5Matrix

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
        group.create_dataset(name='images', data=x_data)
        group.create_dataset(name='labels', data=y_data_1)
        dataset['/1/images']
        print(1)


if __name__ == '__main__':
    # mnist = tf.keras.datasets.mnist
    #
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train, x_test = x_train / 255.0, x_test / 255.0
    # extract()
    dataset = h5py.File('dataset_new_.hdf5', 'r')
    end_split = int(np.round(dataset['images'].shape[0] * 0.8, 0))
    x_train = dataset['images'][:][:end_split]
    y_train = dataset['labels'][:][:end_split]
    x_test = dataset['images'][:][end_split:]
    y_test = dataset['labels'][:][end_split:]
    x_test_2 = x_test[5]
    y_labels = y_test[5]
    # img = Image.fromarray(x_test_2, 'RGB')
    # img.show()
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(16)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(16)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(140, 148, 3)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(5, activation='softmax')
    ])
    #
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, epochs=10)

    model.evaluate(test_ds, verbose=2)
    print('predict : ', str(np.argmax(model.predict(x=x_test_2.reshape([1, 140, 148, 3])))), 'real : ' + str(y_labels))
    model.save('auto_sword.h5')
