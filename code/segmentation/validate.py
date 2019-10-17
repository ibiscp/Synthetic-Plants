import os
import cv2
from glob import glob
import numpy as np
import keras_segmentation
from keras_segmentation.pretrained import *
from glob import glob
import os
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use("WebAgg")
matplotlib.rcParams['savefig.pad_inches'] = 0
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random

os.environ['KMP_DUPLICATE_LIB_OK']='True'

types = ['original', 'synthetic', 'half-original-synthetic', 'original-synthetic']
# types = ['original', 'synthetic', 'original-synthetic', 'original-synthetic']
names = ['9783 Original\n', '21844 Synthetic\n', '4898 Original +\n10924 Synthetic\n', '9783 Original +\n10924 Synthetic\n']


path = "/media/sf_MAXTOR/"
segmentation_path = os.path.join(path, "Final Files/segmentation/")
test_dataset = os.path.join(path, 'Segmentation/test/original/')

samples = 12
shape = (1296, 966)
out_shape = (648, 483)
original_file = ['image_358_1.png']
size = (out_shape[1] * samples, out_shape[0] * 6)
width = size[1]
height = size[0]
dpi = 100

# Get random files from test
random.seed(8299)
files = glob(test_dataset + 'image/*.png')
files = random.choices(files, k=samples)

files = ['image_718_3.png', 'image_74_3.png', 'image_822_2.png', 'image_322_1.png', 'image_2396_1.png',
         'image_2820_2.png', 'image_162_3.png', 'image_823_2.png', 'image_5_0.png', 'image_815_2.png',
         'image_2257_3.png', 'image_603_1.png']
# 'image_411_0.png'
# 'image_1986_1.png'
# 'image_1896_1.png'
# 'image_401_2.png'
# image_2310_3.png
# image_603_1.png





# Create image
plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
gs1 = gridspec.GridSpec(samples, 6)
gs1.update(hspace=0.025, wspace=0.025)

# Plot real images and original masks
for i, file in enumerate(files):
    file = file.split("/")[-1]

    # Input images
    rgb = cv2.imread(os.path.join(test_dataset, 'image', file), flags=cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, out_shape, interpolation=cv2.INTER_NEAREST)
    mask_ = cv2.imread(os.path.join(test_dataset, 'mask', file), flags=cv2.IMREAD_GRAYSCALE)
    mask = np.zeros((shape[1], shape[0], 3)).astype('uint8')
    mask[:, :, 1] = ((mask_[:, :] == 2) * 255).astype('uint8')
    mask[:, :, 0] = ((mask_[:, :] == 1) * 255).astype('uint8')
    mask = cv2.resize(mask, out_shape, interpolation=cv2.INTER_NEAREST)

    # Plot rgb image
    ax = plt.subplot(gs1[i*6])
    ax.imshow(rgb, vmin=0, vmax=255, interpolation='nearest')
    ax.axis('off')
    ax.set_aspect('equal')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    if i == 0:
        ax.set_title('Original RGB \n', fontsize=25, fontweight='bold')

    # Plot mask image
    ax = plt.subplot(gs1[i*6+1])
    ax.imshow(mask, vmin=0, vmax=255, interpolation='nearest')
    ax.axis('off')
    ax.set_aspect('equal')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    if i == 0:
        ax.set_title('Ground truth mask \n', fontsize=25, fontweight='bold')


for i, type in enumerate(types):

    # Get weeights name
    latest_weights = os.path.join(segmentation_path, type, "resources-" + type + "-.4")

    # Network parameters
    model_config = {
            "model_class": "resnet50_unet",
            "n_classes": 3,
            "input_height": 416,
            "input_width": 608}

    # Load network
    model = model_from_checkpoint_path(model_config, latest_weights)

    # Plot real images and original masks
    for j, file in enumerate(files):
        file = file.split("/")[-1]

        # Output
        output_ = model.predict_segmentation(inp=os.path.join(test_dataset, 'image', file))
        shape_seg = (model.output_width, model.output_height)
        output = np.zeros((shape_seg[1], shape_seg[0], 3)).astype('uint8')
        output[:, :, 1] = ((output_[:, :] == 2) * 255).astype('uint8')
        output[:, :, 0] = ((output_[:, :] == 1) * 255).astype('uint8')
        output = cv2.resize(output, out_shape, interpolation=cv2.INTER_NEAREST)

        # Plot mask image
        ax = plt.subplot(gs1[j*6+2+i])
        ax.imshow(output, vmin=0, vmax=255, interpolation='nearest')
        ax.axis('off')
        ax.set_aspect('equal')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)

        if j == 0:
            ax.set_title(names[i], fontsize=25, fontweight='bold')

plt.savefig("segmentation_result.png", bbox_inches='tight', pad_inches=0.025)

plt.close()

for i, file in enumerate(files):
    print(file.split("/")[-1])



# for i in range(num_metrics):
#
#     for type in types:
#
#         latest_weights = os.path.join(path, type, "resources-" + type + "-.4")
#
#         #
#
#         model_config = {
#                 "model_class": "resnet50_unet",
#                 "n_classes": 3,
#                 "input_height": 416,
#                 "input_width": 608}
#
#         model = model_from_checkpoint_path(model_config, latest_weights)
#
#         # shape = (model.output_width, model.output_height)
#
#         # files = glob(path + type + '*.png')
#         file_path = '/Volumes/MAXTOR/Segmentation/train/original/image/' + original_file
#         mask_path = '/Volumes/MAXTOR/Segmentation/train/original/mask/' + original_file
#         output_path = os.path.join(path, type, 'images/')
#
#
#         # for i, f in enumerate(files):
#         # Input image
#         input = cv2.imread(file_path)
#         # input = cv2.resize(input, shape, interpolation=cv2.INTER_AREA)
#
#         # Original mask
#         mask_ = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         mask = np.zeros((shape[1], shape[0], 3))
#         mask[:, :, 1] = ((mask_[:, :] == 2) * 255).astype('uint8')
#         mask[:, :, 2] = ((mask_[:, :] == 1) * 255).astype('uint8')
#
#         # Output
#         output_ = model.predict_segmentation(inp=file_path)
#         shape_seg = (model.output_width, model.output_height)
#         output = np.zeros((shape_seg[1], shape_seg[0], 3))
#         output[:, :, 1] = ((output_[:, :] == 2) * 255).astype('uint8')
#         output[:, :, 2] = ((output_[:, :] == 1) * 255).astype('uint8')
#         output = cv2.resize(output, shape, interpolation=cv2.INTER_NEAREST)
#
#         if not os.path.exists(output_path):
#             os.makedirs(output_path)
#
#         cv2.imwrite(output_path + '_image.png', input)
#         cv2.imwrite(output_path + '_generated.png', output)
#         cv2.imwrite(output_path + '_mask.png', mask)
#
#
#
#
#         ax = plt.subplot(gs1[i])
#         horizontal = getattr(gold_metrics, names[i].lower())
#         max_ = max(max(metrics[names[i].lower()]), horizontal)
#         min_ = min(min(metrics[names[i].lower()]), horizontal)
#         offset = (max_ - min_) * 0.1
#
#         # ax = fig.add_subplot(num_metrics, 1, i + 1)
#         ax.axhline(y=horizontal, color='r', linestyle=':')
#         ax.set_xlim([0, epochs])
#         ax.set_ylim([min_ - offset, max_ + offset])
#         ax.set_ylabel(names[i], fontsize=18)
#         ax.yaxis.set_label_position("right")
#         ax.plot(metrics[names[i].lower()])
#         # ax[i].axes.get_xaxis().set_fontsize(18)
#         # ax[i].axes.get_yaxis().set_fontsize(18)
#         ax.tick_params(axis='both', which='major', labelsize=18)
#
#         if i != num_metrics - 1 and i != num_metrics - 2:
#             ax.axes.get_xaxis().set_visible(False)
#
#     # plt.canvas.draw()
#     # image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
#     # frames.append(image)
#
#     gif_dir=''
#     name = type + ".png"
#     plt.savefig(os.path.join(gif_dir, name), bbox_inches='tight', pad_inches=0.025)
#     # plt.close()
#
#     plt.close()