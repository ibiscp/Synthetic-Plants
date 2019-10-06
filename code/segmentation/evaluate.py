import keras_segmentation
from keras_segmentation.predict import *
from keras_segmentation.pretrained import *
from glob import glob
import os
from tqdm import tqdm
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from segmentation import *

def evaluate(path, type):

    model_config = {
        "model_class": "resnet50_unet",
        "n_classes": 3,
        "input_height": 416,
        "input_width": 608}

    latest_weights = "resources/" + type + "resources-original-.4"

    model = model_from_checkpoint_path(model_config, latest_weights)

    # Calculate IoU
    images = glob(os.path.join(path, 'test/original/image/', '*.png'))
    images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    masks = glob(os.path.join(path, 'test/original/mask/', '*.png'))
    masks.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    ious = evaluate(model=model, inp_inmges=images, annotations=masks)

    print("\n" + type)
    print("Class wise IoU ", np.mean(ious, axis=0))
    print("Total  IoU ", np.mean(ious))

# Train synthetic data
# train_segmentation("../../../dataset/Segmentation/", "original/")
evaluate("/Volumes/MAXTOR/Segmentation/", "original-synthetic/")