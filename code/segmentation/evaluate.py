import keras_segmentation
from keras_segmentation.predict import *
from keras_segmentation.pretrained import *
from glob import glob
import os
from tqdm import tqdm
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# from segmentation import *

EPS = 1e-12

def get_iou( gt , pr , n_classes ):
    class_wise = np.zeros(n_classes)
    for cl in range(n_classes):
        target = ( gt == cl )
        prediction = ( pr == cl )

        intersection = np.logical_and(target, prediction)
        union = np.logical_or(target, prediction)
        iou_score = np.sum(intersection) / (np.sum(union) + EPS )
        class_wise[cl] = iou_score
    return class_wise

def get_segmentation_arr(path, nClasses, width, height, no_reshape=False):

    if type(path) is np.ndarray:
        img = path
    else:
        img = cv2.imread(path, 0)

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST).flatten()

    return img

def evaluate(model=None, inp_inmges=None, annotations=None, checkpoints_path=None):

    ious = []
    for inp, ann in tqdm(zip(inp_inmges, annotations)):
        pr = predict(model, inp).flatten()
        gt = get_segmentation_arr(ann, model.n_classes, model.output_width, model.output_height)
        iou = get_iou(gt, pr, model.n_classes)
        ious.append(iou)
    ious = np.array(ious)

    return ious

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