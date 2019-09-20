from scipy import misc
import pickle
from glob import glob
import imageio
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("WebAgg")
matplotlib.rcParams['savefig.pad_inches'] = 0
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils import *
import csv

COLUMNS = 6
ROWS = 8

def gen_log_space(limit, n):
    result = [1]
    if n>1:  # just a check to avoid ZeroDivisionError
        ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    while len(result)<n:
        next_value = result[-1]*ratio
        if next_value - result[-1] >= 1:
            # safe zone. next_value will be a different integer
            result.append(next_value)
        else:
            # problem! same integer. we need to find next_value by artificially incrementing previous value
            result.append(result[-1]+1)
            # recalculate the ratio so that the remaining values will scale correctly
            ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
    return np.array(list(map(lambda x: round(x)-1, result)), dtype=np.uint64)

def plot_grid(path, type, number):

    files = glob(path + type + '_' + str(number) + '_*.png')
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    images = []
    indexes = gen_log_space(len(files), COLUMNS * ROWS)

    # Get gif images
    for i in indexes:
        f = files[i]
        img = cv2.imread(f, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)

    plt.figure(figsize=(COLUMNS * 3, ROWS * 3))
    gs1 = gridspec.GridSpec(ROWS, COLUMNS)
    gs1.update(hspace=0.025, wspace=0.025, top=1)
    for i, image in enumerate(images):
        ax = plt.subplot(gs1[i])
        ax.imshow(image, vmin=0, vmax=255, interpolation='nearest')
        ax.set_title(str(int(indexes[i]+1)), fontsize=20, fontweight='bold')
        ax.axis('off')
        ax.set_aspect('equal')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)

    gif_dir=''
    name = type + ".png"
    plt.savefig(os.path.join(gif_dir, name), bbox_inches='tight', pad_inches=0.025)
    plt.close()

def load_bla(type):
    test_dataset = '../../plants_dataset/SugarBeets_256/test/'
    files = glob(test_dataset + type + '/' + '*.png')
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    return files

def create_csv(metrics, type):
    names = ['Inception', 'Mode', 'MMD', 'EMD', 'FID', 'KNN']
    # with open(type+'.csv', 'w') as csvFile:
    #     writer = csv.writer(csvFile)
    #
    #     writer.writerow(names)
    #
    #     for i, m in enumerate(data):
    #         writer.writerow([i, m.emd, m.fid, m.inception, m.knn, m.mmd, m.mode])

    # Calculate gold metrics
    gold_metrics = calculate_gold_metrics(load_bla(type), type)

    # gold_metrics=load('gold.pkl')

    # # Print variables
    # for n in names[1:]:
    #     value = getattr(gold_metrics, n)
    #     print(r"\newcommand{" + "\\" + type + n.capitalize() + r"}{" + str(value) + "}")
    # print(r"\newcommand{" + "\\" + type + 'Epochs' + r"}{" + str(len(data)) + "}")

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

    size = (426*3, 674*3)
    width = size[0]
    height = size[1]
    dpi = 100

    fig, ax = plt.subplots(num_metrics, figsize=(width / dpi, height / dpi), dpi=dpi)
    # fig.suptitle('Epoch: ' + str(epoch + 1), x=0.11, y=.96, horizontalalignment='left', verticalalignment='top',
    #              fontsize=14)
    # fig.patch.set_visible(False)
    # fig.axes([0,0,1,1], frameon=False)
    # fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi, frameon=False, tight_layout=True)

    for i in range(num_metrics):
        horizontal = getattr(gold_metrics, names[i].lower())
        max_ = max(max(metrics[names[i].lower()]), horizontal)
        min_ = min(min(metrics[names[i].lower()]), horizontal)
        offset = (max_ - min_) * 0.1

        # ax = fig.add_subplot(num_metrics, 1, i + 1)
        ax[i].axhline(y=horizontal, color='r', linestyle=':')
        ax[i].set_xlim([0, epochs])
        ax[i].set_ylim([min_ - offset, max_ + offset])
        ax[i].set_ylabel(names[i], fontsize=18)
        ax[i].yaxis.set_label_position("right")
        ax[i].plot(metrics[names[i].lower()])
        # ax[i].axes.get_xaxis().set_fontsize(18)
        # ax[i].axes.get_yaxis().set_fontsize(18)
        ax[i].tick_params(axis='both', which='major', labelsize=18)

        if i != num_metrics - 1:
            ax[i].axes.get_xaxis().set_visible(False)

    fig.canvas.draw()
    # image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    # frames.append(image)

    gif_dir=''
    name = type + ".png"
    plt.savefig(os.path.join(gif_dir, name), bbox_inches='tight', pad_inches=0.025)
    # plt.close()

    plt.close(fig)



# # Mask
# path = '../../plants_output/blur/samples/DCGAN-batch_size-32-d_beta_1-0.5-d_ld-0.001-d_lr-0.0002-epochs-500-g_beta_1-0.5-g_ld-0.001-g_lr-0.0002-latent_dim-100/'
# type = 'mask'
# number = 13
# plot_grid(path, type, number)
#
# # RGB
# path = '/Volumes/MAXTOR/spade_final_training/samples/'
# type = 'rgb'
# number = 24
# plot_grid(path, type, number)
#
# # NIR
# path = '/Volumes/MAXTOR/spade_final_training/samples/'
# type = 'nir'
# number = 24
# plot_grid(path, type, number)

# metrics_mask = load(os.path.join('../../plants_output/blur/model/DCGAN-batch_size-32-d_beta_1-0.5-d_ld-0.001-d_lr-0.0002-epochs-500-g_beta_1-0.5-g_ld-0.001-g_lr-0.0002-latent_dim-100/', 'checkpoint.pkl'))['metrics']
# create_csv(metrics_mask, 'mask')

[metrics_rgb, metrics_nir] = load(os.path.join('/Volumes/MAXTOR/spade_final_training/model/SPADE_load_test_SugarBeets_256_hinge_2multi_4dis_1_1_10_10_0.05_sn_TTUR_more/', 'metrics.pkl'))
create_csv(metrics_rgb, 'rgb')

create_csv(metrics_nir, 'nir')
