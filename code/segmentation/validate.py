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
os.environ['KMP_DUPLICATE_LIB_OK']='True'

types = ['original', 'original-synthetic', 'synthetic']

path = "/Volumes/MAXTOR/Final Files/segmentation/"

shape = (1296, 966)

original_file = 'image_358_1.png'

for type in types:

    latest_weights = os.path.join(path, type, "resources-" + type + "-.4")

    #

    model_config = {
            "model_class": "resnet50_unet",
            "n_classes": 3,
            "input_height": 416,
            "input_width": 608}

    model = model_from_checkpoint_path(model_config, latest_weights)

    # shape = (model.output_width, model.output_height)

    # files = glob(path + type + '*.png')
    file_path = '/Volumes/MAXTOR/Segmentation/train/original/image/' + original_file
    mask_path = '/Volumes/MAXTOR/Segmentation/train/original/mask/' + original_file
    output_path = os.path.join(path, type, 'images/')


    # for i, f in enumerate(files):
    # Input image
    input = cv2.imread(file_path)
    # input = cv2.resize(input, shape, interpolation=cv2.INTER_AREA)

    # Original mask
    mask_ = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = np.zeros((shape[1], shape[0], 3))
    mask[:, :, 1] = ((mask_[:, :] == 2) * 255).astype('uint8')
    mask[:, :, 2] = ((mask_[:, :] == 1) * 255).astype('uint8')

    # Output
    output_ = model.predict_segmentation(inp=file_path)
    shape_seg = (model.output_width, model.output_height)
    output = np.zeros((shape_seg[1], shape_seg[0], 3))
    output[:, :, 1] = ((output_[:, :] == 2) * 255).astype('uint8')
    output[:, :, 2] = ((output_[:, :] == 1) * 255).astype('uint8')
    output = cv2.resize(output, shape, interpolation=cv2.INTER_NEAREST)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cv2.imwrite(output_path + '_image.png', input)
    cv2.imwrite(output_path + '_generated.png', output)
    cv2.imwrite(output_path + '_mask.png', mask)