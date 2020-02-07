import glob
import os
import cv2
import numpy as np


def get_folders_in_folder(d_path):
    return [o for o in os.listdir(d_path) if os.path.isdir(os.path.join(d_path, o))]


def pad_array(input_array, size):
    input_array = input_array.astype(int)
    t = size - len(input_array)
    return np.pad(input_array, pad_width=(0, t), mode='constant', constant_values=-1)


def format_data(data):
    #zero pad image data array to fix lengh accross dataset
    print()
    max_l = 0
    for item in data:
        image_array_lenght = item[0].shape[0]
        if image_array_lenght > max_l:
            max_l = image_array_lenght
    print("max array lengh is %d" % max_l)
    # now pad all the arrays
    data_formated = []
    for item in data:
        data_formated.append([pad_array(item[0], max_l), item[1]])
    del data

    return data_formated


if __name__ == '__main__':
    print("start")
    dataset_dir_path = "dataset"
    folder_list = get_folders_in_folder(dataset_dir_path)
    print(folder_list) #this is basicly the list of classes
    data = []
    for folder in folder_list:
        files = glob.glob("%s\\%s\\*.jpg" % (dataset_dir_path, folder)) #this is the list of images in the curr folder
        label = folder #label(what the image data actually represent) is the folder name
        print("found %d files for class '%s'" % (len(files), label))
        for image_path in files:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # cv2.imshow(label, image)
            # cv2.waitKey(0) # wait for key input
            training_pair = [image.flatten(), label]
            data.append(training_pair)
    data = format_data(data)
    print("dataset size is %d" % len(data))