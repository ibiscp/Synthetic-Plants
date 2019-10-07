from scipy import misc
import pickle
from glob import glob
import imageio
import os
import cv2
import numpy as np
from pytorchMetrics import *
import matplotlib
matplotlib.use("WebAgg")
matplotlib.rcParams['savefig.pad_inches'] = 0
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

GIF_MATRIX = 5

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def imsave(image, path):
    return misc.imsave(path, image)

# Save dictionary to file
def save(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)

# Load dictionary from file
def load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

# Convert to the range [-1, 1]
def preprocessing(x):
    x = x/127.5 - 1
    return x

# Convert to the range [0, 255]
def postprocessing(x):
    x = ((x + 1) / 2) * 255.0
    return x

def plot_gif(images, epoch, gif_dir, type):

    plt.figure(figsize=(GIF_MATRIX*3, GIF_MATRIX*3))
    gs1 = gridspec.GridSpec(GIF_MATRIX, GIF_MATRIX)
    gs1.update(wspace=0.025, hspace=0.025)
    for i in range(images.shape[0]):
        ax = plt.subplot(gs1[i])
        if type == 'mask':
            ax.imshow(images[i,:,:,0], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        elif  type == 'nir':
            ax.imshow(images[i,:,:], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        else:
            ax.imshow(images[i,:,:,:], vmin=0, vmax=255, interpolation='nearest')
        ax.axis('off')
        ax.set_aspect('equal')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)

    name = type + "_%d.png" % epoch
    plt.savefig(os.path.join(gif_dir, name), bbox_inches='tight', pad_inches=0.025)
    plt.close()

# Create the gif given the dictionary and its size
def create_gif(images_directory, metrics, test_dataset, type, duration=10):

    files = glob(os.path.join(images_directory, '*_*.png'))
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    frames = []
    images = []

    size = (1100, 1100)

    # Get gif images
    for f in files:
        img = cv2.imread(f, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        images.append(img)

    # Construct graph
    graphs = generate_graphs(metrics, test_dataset, size, type)

    for i, image in enumerate(images):
        graph = graphs[i]
        # graph = graph[3:3 + size[0], 5:5 + size[1]]
        new_im = np.hstack((image, graph))
        frames.append(new_im)

    # Repeat last frames
    for i in range(int(len(files)*.5)):
        frames.append(frames[-1])

    # Calculate time between frames
    time = duration/len(frames)

    # Create gif
    imageio.mimsave(images_directory + 'training_' + type + '.gif', frames, format='GIF', duration=time)

def generate_graphs(metrics, test_dataset, size, type):

    # List of metrics
    # names = ['emd', 'fid', 'inception', 'knn', 'mmd', 'mode']
    names = ['emd', 'fid', 'inception', 'mmd', 'mode']

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

    metrics = {'emd': emd, 'fid': fid, 'inception': inception, 'knn': knn, 'mmd': mmd, 'mode': mode}
    num_metrics = len(names)
    epochs = len(emd)

    frames = []

    # Calculate gold metrics
    gold_metrics = calculate_gold_metrics(test_dataset, type)

    # Graph size
    width = size[0]
    height = size[1]
    dpi = 100

    for epoch in range(epochs):

        fig, ax = plt.subplots(num_metrics, figsize=(width/dpi, height/dpi), dpi=dpi)
        fig.suptitle('Epoch: ' + str(epoch+1), x=0.11, y=.96, horizontalalignment='left', verticalalignment='top', fontsize=14)
        # fig.patch.set_visible(False)
        # fig.axes([0,0,1,1], frameon=False)
        # fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi, frameon=False, tight_layout=True)

        for i in range(num_metrics):
            horizontal = getattr(gold_metrics, names[i])
            max_ = max(max(metrics[names[i]]), horizontal)
            min_ = min(min(metrics[names[i]]), horizontal)
            offset = (max_ - min_) * 0.1

            # ax = fig.add_subplot(num_metrics, 1, i + 1)
            ax[i].axhline(y=horizontal, color='r', linestyle=':')
            ax[i].set_xlim([0, epochs])
            ax[i].set_ylim([min_ - offset, max_ + offset])
            ax[i].set_ylabel(names[i])
            ax[i].yaxis.set_label_position("right")
            ax[i].plot(metrics[names[i]][:epoch])

            if i != num_metrics-1:
                ax[i].axes.get_xaxis().set_visible(False)

        fig.canvas.draw()
        image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
        frames.append(image)

        plt.close(fig)
    return frames

# Calculate metrics when comparing one set of real images with another
# These values are the desirable values to achieve with GAN
def calculate_gold_metrics(test_dataset, type):
    # files, _ = load_dataset_list(test_directory + 'mask/')
    if type == 'mask' or type == 'nir':
        data = load_data(test_dataset, type, repeat=True)
    else:
        data = load_data(test_dataset, type, repeat=False)

    metrics_list = []

    metrics = pytorchMetrics()

    for i in range(10):
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

# Load data given a file list
# Input:
#   - files: list of files
#   - repeat: repeat third chanel
def load_data(files, type, repeat=False, scale=False):

    data = []
    for file in files:
        if type == 'mask' or type == 'nir':
            img = cv2.imread(file, 0)
        else:
            img = cv2.imread(file, -1)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data.append(img)

    data = np.asarray(data, dtype='uint8')

    # Rescale
    if scale:
        data = data / 127.5 - 1.

    if type == 'mask' or type == 'nir':
        data = np.expand_dims(data, axis=3)

    if repeat:
        data = np.repeat(data, 3, 3)

    return data

# Load list of files of a dictionary with image shape
def load_dataset_list(directory, type):
    # Load the dataset
    files = glob(directory + '*.png')
    # number_files = len(files)
    # print('\nNumber of files: ', number_files)

    if type == 'mask' or type == 'nir':
        image = cv2.imread(files[0], 0)
        image = np.expand_dims(image, axis=3)
    else:
        image = cv2.imread(files[0], -1)

    shape = image.shape

    return files, shape