import pickle
import glob
import imageio
import os
import cv2
import numpy as np
import random
from pytorchMetrics import *


# Save dictionary to file
def save(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)

# Load dictionary from file
def load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

# Create the gif given the dictionary and its size
def create_gif(directory, size=100):
    files = glob.glob(directory + 'gif/' + '*.png')
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    images = []

    number = int(len(files)/size)

    for i in range(size):
        images.append(imageio.imread(files[int(i*number)]))

    for i in range(int(size*.2)):
        images.append(imageio.imread(files[-1]))

    imageio.mimsave(directory + 'training.gif', images)

def load_data(path='../dataset/test/', repeat = False):
    # Load the dataset
    files = os.listdir(path)
    number_files = len(files)
    print('\nNumber of files: ', number_files)

    data = []
    for file in files:
        img = cv2.imread(path + file, 0)
        data.append(img)

    data = np.asarray(data, dtype='uint8')

    # Rescale
    data = data / 127.5 - 1.
    data = np.expand_dims(data, axis=3)

    if repeat:
        data = np.repeat(data, 3, 3)

    return data

# Calculate metrics when comparing one set of real images with another
# These values are the desirable values to achieve with GAN
def calculate_gold_metrics():
    data = load_data(repeat=True)
    metrics_list = []

    metrics = pytorchMetrics()

    for i in range(2):
        samples = data[np.random.choice(data.shape[0], 200)]
        real_1 = samples[:100]
        real_2 = samples[100:]

        metrics_list.append(metrics.compute_score(real_1, real_2))

    emd, mmd, knn, inception, mode, fid = np.array([(t.emd, t.mmd, t.knn, t.inception, t.mode, t.fid) for t in metrics_list]).T

    score = Score()
    score.emd = np.mean(emd)
    score.mmd = np.mean(mmd)
    score.knn = np.mean(knn)
    score.inception = np.mean(inception)
    score.mode = np.mean(mode)
    score.fid = np.mean(fid)

    return score



calculate_gold_metrics()