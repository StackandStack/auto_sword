import os
import re

import h5py
import mss
import numpy as np
import pyautogui as pag
from colorama import Fore
from tensorflow import keras

left_icon = {'left': 80, 'top': 490, 'width': 74, 'height': 70}
right_icon = {'left': 210, 'top': 490, 'width': 74, 'height': 70}

left_button = [66, 601]
right_button = [304, 612]
mid_button = [150, 612]

icon_colors = {
    'BOMB': Fore.RED,
    'SWORD': Fore.MAGENTA,
    'POISON': Fore.GREEN,
    'JEWEL': Fore.CYAN,
}

colors_sample = dict((
    ((196, 2, 51), "RED"),
    ((255, 165, 0), "ORANGE"),
    ((255, 205, 0), "YELLOW"),
    ((0, 128, 0), "GREEN"),
    ((0, 0, 255), "BLUE"),
    ((127, 0, 255), "VIOLET"),
    ((0, 0, 0), "BLACK"),
    ((255, 255, 255), "WHITE"),
    ((255, 134, 255), "SWORD"),
    ((50, 50, 50), "BOMB"),
    ((120, 172, 102), "POISON"),
))


# def rgb2lab(inputColor):
#     num = 0
#     RGB = [0, 0, 0]
#
#     for value in inputColor:
#         value = float(value) / 255
#
#         if value > 0.04045:
#             value = ((value + 0.055) / 1.055) ** 2.4
#         else:
#             value = value / 12.92
#
#         RGB[num] = value * 100
#         num = num + 1
#
#     XYZ = [0, 0, 0, ]
#
#     X = RGB[0] * 0.4124 + RGB[1] * 0.3576 + RGB[2] * 0.1805
#     Y = RGB[0] * 0.2126 + RGB[1] * 0.7152 + RGB[2] * 0.0722
#     Z = RGB[0] * 0.0193 + RGB[1] * 0.1192 + RGB[2] * 0.9505
#     XYZ[0] = round(X, 4)
#     XYZ[1] = round(Y, 4)
#     XYZ[2] = round(Z, 4)
#
#     XYZ[0] = float(XYZ[0]) / 95.047  # ref_X =  95.047   Observer= 2°, Illuminant= D65
#     XYZ[1] = float(XYZ[1]) / 100.0  # ref_Y = 100.000
#     XYZ[2] = float(XYZ[2]) / 108.883  # ref_Z = 108.883
#
#     num = 0
#     for value in XYZ:
#         if value > 0.008856:
#             value = value ** (0.3333333333333333)
#         else:
#             value = (7.787 * value) + (16 / 116)
#
#         XYZ[num] = value
#         num = num + 1
#
#     Lab = [0, 0, 0]
#     L = (116 * XYZ[1]) - 16
#     a = 500 * (XYZ[0] - XYZ[1])
#     b = 200 * (XYZ[1] - XYZ[2])
#
#     Lab[0] = round(L, 4)
#     Lab[1] = round(a, 4)
#     Lab[2] = round(b, 4)
#
#     return Lab


# def color_dist(c1, c2):
#     return sum((a - b) ** 2 for a, b in zip(rgb2lab(c1), rgb2lab(c2)))


# def min_color_diff(color_to_match):
#     return min((color_dist(color_to_match, test), colors_sample[test]) for test in colors_sample)


def find_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist


def get_color(img):
    mean = np.mean(img, axis=(0, 1))

    if (mean[0] > 35 and mean[0] < 45) and (mean[1] > 35 and mean[1] < 45) and (mean[2] > 35 and mean[2] < 45):
        result = 'BOMB'
    elif mean[0] > 220 and mean[1] < 90 and mean[2] > 220:
        result = 'SWORD'
    elif mean[0] > 70 and mean[0] < 110 and mean[1] > 130 and mean[2] > 50 and mean[2] < 90:
        result = 'POISON'
    elif mean[0] > 180 and mean[0] < 230 and mean[1] > 180 and mean[1] < 225 and mean[2] > 80 and mean[2] < 125:
        result = 'JEWEL'
    else:
        result = 'None'
    return result, mean
    # img = img.reshape((img.shape[0] * img.shape[1], 3))  # represent as row*column,channel number
    # clt = KMeans(n_clusters=2)
    # clt.fit(img)
    #
    # hist = find_histogram(clt)
    # dominant_colors = clt.cluster_centers_
    # hist = sorted(hist, reverse=True)
    # dominant_colors = [list(x) for _, x in sorted(zip(hist, dominant_colors))]
    # pred = min_color_diff(dominant_colors[0])
    #
    # cli_color = Fore.WHITE
    # if pred[1] in icon_colors:
    #     cli_color = icon_colors[pred[1]]
    #
    # print(hist, dominant_colors, '%s%s' % (cli_color, pred))
    #
    # if hist[0] > 0.5:
    #     if dominant_colors[0][0] < 55 and dominant_colors[0][1] < 55 and dominant_colors[0][2] < 55:
    #         return 'BOMB'
    #     elif dominant_colors[0][0] > 250 and dominant_colors[0][0] > 120 and dominant_colors[0][1] < 140 and \
    #             dominant_colors[0][2] > 250:
    #         return 'SWORD'
    #     elif dominant_colors[0][0] > 110 and dominant_colors[0][0] < 130 and dominant_colors[0][1] > 150 and \
    #             dominant_colors[0][2] > 90 and dominant_colors[0][2] < 110:
    #         return 'POISON'
    #
    # return False


def click(x, y):
    pag.moveTo(x=x, y=y, duration=0.0)
    pag.mouseDown()
    pag.mouseUp()


def make_data_set():
    file = h5py.File(f'./dataset_new_.hdf5', 'w')
    breakp = 0
    x_data = list()
    y_data = list()
    iter = 0

    while True:
        pag.PAUSE = 0.05
        with mss.mss() as sct:
            left_img = np.array(sct.grab(left_icon))[:, :, :3]
            right_img = np.array(sct.grab(right_icon))[:, :, :3]
            left_image_nm, left_mean = get_color(left_img)
            right_image_nm, right_mean = get_color(right_img)
            if (left_image_nm == 'SWORD') and (right_image_nm == 'BOMB'):
                print('TAP left')
                x_data.extend([left_img])
                y_data.extend([1])
                x_data.extend([right_img])
                y_data.extend([0])
                click(x=left_button[0], y=left_button[1])
                breakp = 0
            elif (left_image_nm == 'SWORD') and (right_image_nm == 'POISON'):
                x_data.extend([left_img])
                y_data.extend([1])
                x_data.extend([right_img])
                y_data.extend([2])
                click(x=left_button[0], y=left_button[1])
                breakp = 0
            elif (left_image_nm == 'BOMB') and (right_image_nm == 'SWORD'):
                print('TAP right')
                x_data.extend([left_img])
                y_data.extend([0])
                x_data.extend([right_img])
                y_data.extend([1])
                click(x=right_button[0], y=right_button[1])
                breakp = 0
            elif (left_image_nm == 'POISON') and (right_image_nm == 'SWORD'):
                print('TAP right')
                x_data.extend([left_img])
                y_data.extend([2])
                x_data.extend([right_img])
                y_data.extend([1])
                click(x=right_button[0], y=right_button[1])
                breakp = 0
            elif left_image_nm == 'JEWEL' and right_image_nm == 'JEWEL':
                print('FEVER')
                x_data.extend([left_img])
                y_data.extend([3])
                x_data.extend([right_img])
                y_data.extend([3])
                click(x=left_button[0], y=left_button[1])
                click(x=right_button[0], y=right_button[1])
                breakp = 0
            else:
                x_data.extend([left_img])
                y_data.extend([4])
                x_data.extend([right_img])
                y_data.extend([4])
                breakp += 1
                if breakp >= 100:
                    x_data_matrix = np.stack(x_data, axis=0)
                    y_label_data = np.array(y_data)
                    file.create_dataset(name='images', data=x_data_matrix)
                    file.create_dataset(name='labels', data=y_label_data)
                    break
                print('FAIL')
        iter += 1


def run_main():
    breakp = 0
    new_model = keras.models.load_model('auto_sword.h5')
    while True:
        pag.PAUSE = 0.02
        with mss.mss() as sct:
            left_img = np.array(sct.grab(left_icon))[:, :, :3]
            right_img = np.array(sct.grab(right_icon))[:, :, :3]
            left_chk = np.argmax(new_model.predict(x=left_img.reshape([1, 140, 148, 3])))
            right_chk = np.argmax(new_model.predict(x=right_img.reshape([1, 140, 148, 3])))
            if (left_chk == 1) and (right_chk == 0 or right_chk == 2):
                print('TAP left')
                click(x=left_button[0], y=left_button[1])
                breakp = 0
            elif (left_chk == 0 or right_chk == 2) and (right_chk == 1):
                print('TAP right')
                click(x=right_button[0], y=right_button[1])
                breakp = 0
            elif (left_chk == 3) and (right_chk == 3):
                print('FEVER')
                click(x=left_button[0], y=left_button[1])
                click(x=right_button[0], y=right_button[1])
                breakp += 1
                if breakp >= 100:
                    break
            # elif (left_chk == 2) or (right_chk == 2):
            #     print('PASS')
            #     continue
            else:
                breakp += 1
                if breakp >= 200:
                    break
                print('FAIL')


if __name__ == '__main__':
    run_main()
