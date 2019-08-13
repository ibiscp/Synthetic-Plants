import pickle
import glob
import imageio
import os
import cv2
import numpy as np
# import random
from pytorchMetrics import *
import matplotlib
# ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg',
 # 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
matplotlib.use("WebAgg")
matplotlib.rcParams['savefig.pad_inches'] = 0
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from matplotlib.backends.backend_agg import FigureCanvas
# from matplotlib.figure import Figure
import matplotlib.image as mpimg
from PIL import Image

# Save dictionary to file
def save(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)

# Load dictionary from file
def load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

# Create the gif given the dictionary and its size
def create_gif(directory, duration=10):
    files = glob.glob(directory + 'gif/' + '*.png')
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    frames = []
    images = []

    # Get gif images
    for f in files:
        img = cv2.imread(f, 1)
        img = cv2.resize(img, (400, 400))
        images.append(img)

    # Construct graph
    graphs = generate_graphs(directory)

    # new_im = Image.new('RGB', (total_width, max_height))
    for i, image in enumerate(images):
        graph = graphs[i]
        graph = graph[3:3 + 400, 5:5 + 400]
        new_im = np.hstack((image, graph))
        frames.append(new_im)

    # Repeat last frames
    for i in range(int(len(files)*.5)):
        frames.append(frames[-1])

    # Calculate time between frames
    time = duration/len(frames)

    # Create gif
    imageio.mimsave(directory + 'training.gif', frames, format='GIF', duration=time)

def generate_graphs(directory):

    # Load metrics
    metrics = load(directory + 'checkpoint.pkl')
    metrics = metrics['metrics']

    # List of metrics
    names = ['emd', 'fid', 'inception', 'knn', 'mmd', 'mode']

    emd = []
    fid = []
    inception = []
    knn = []
    mmd = []
    mode = []
    for m in metrics:
        emd.append(m.emd)
        fid.append(m.fid)
        inception.append(m.inception)
        knn.append(m.knn)
        mmd.append(m.mmd)
        mode.append(m.mode)

    metrics = [emd, fid, inception, knn, mmd, mode]
    num_metrics = len(metrics)
    epochs = len(emd)

    frames = []

    # Calculate gold metrics
    gold_metrics = calculate_gold_metrics()

    # Graph size
    width = 420
    height = 420
    dpi = 100

    for epoch in range(epochs):

        fig, ax = plt.subplots(num_metrics, figsize=(width/dpi, height/dpi), dpi=dpi)
        fig.suptitle('Epoch: ' + str(epoch+1), x=0.11, y=.96, horizontalalignment='left', verticalalignment='top', fontsize=14)
        # fig.patch.set_visible(False)
        # fig.axes([0,0,1,1], frameon=False)
        # fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi, frameon=False, tight_layout=True)

        for i in range(num_metrics):
            horizontal = getattr(gold_metrics, names[i])
            max_ = max(max(metrics[i]), horizontal)
            min_ = min(min(metrics[i]), horizontal)
            offset = (max_ - min_) * 0.1

            # ax = fig.add_subplot(num_metrics, 1, i + 1)
            ax[i].axhline(y=horizontal, color='r', linestyle=':')
            ax[i].set_xlim([0, epochs])
            ax[i].set_ylim([min_ - offset, max_ + offset])
            ax[i].set_ylabel(names[i])
            ax[i].yaxis.set_label_position("right")
            ax[i].plot(metrics[i][:epoch])

            if i != num_metrics-1:
                ax[i].axes.get_xaxis().set_visible(False)

        fig.canvas.draw()
        image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
        frames.append(image)

        plt.close(fig)
    return frames

def load_dataset_list(directory):
    # Load the dataset
    files = glob.glob(directory + '*.png')
    number_files = len(files)
    print('\nNumber of files: ', number_files)

    image = cv2.imread(files[0], 0)
    image = np.expand_dims(image, axis=3)
    shape = image.shape

    return files, shape


def load_data(files, repeat=False):
    data = []
    for file in files:
        img = cv2.imread(file, 0)
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
    files, _ = load_dataset_list('../dataset/SugarBeets/test/mask/')
    data = load_data(files, repeat=True)
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



# calculate_gold_metrics()
# generate_graphs('../resources/wgangp/')
# create_gif('../resources/DCGAN-batch_size-128-d_beta_1-0.5-d_lr-0.0002-epochs-100-g_beta_1-0.5-g_lr-0.0002-latent_dim-100/')