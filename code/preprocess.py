import os
import yaml
import cv2
import numpy as np

# Paths
path = '../dataset/Bonn 2016 Median/CKA_160517/'
annotationsPath = 'annotations/YAML/'
nirImagesPath = 'images/nir/'
rgbImagesPath = 'images/rgb/'
maskNirPath = 'annotations/dlp/iMap/'
maskRgbPath = 'annotations/dlp/color/'

# Get files
files = os.listdir(path + annotationsPath)
number_files = len(files)
print('Number of files: ', number_files)

# for file in files:

file = 'bonirob_2016-05-17-11-42-26_14_frame1.yaml'
with open(path + annotationsPath + file, 'r') as stream:
    # Image name
    imageName = os.path.splitext(file)[0]

    # Open images
    rbgimg = cv2.imread(path + rgbImagesPath + imageName + '.png', 1)
    maskNir = cv2.imread(path + maskNirPath + imageName + '.png', 1)
    maskRgb = cv2.imread(path + maskRgbPath + imageName + '.png', 1)
    imgrayRgb = cv2.cvtColor(maskRgb, cv2.COLOR_RGB2GRAY)
    imgrayNir = cv2.cvtColor(maskNir, cv2.COLOR_RGB2GRAY)

    # Get content from yaml file
    content = yaml.safe_load(stream)

    # For each
    for ann in content["annotation"]:
        if ann['type'] == 'SugarBeets':
            print('stem x', ann['stem']['x'])
            print('stem y', ann['stem']['y'])
            x = ann['contours'][0]['x']
            y = ann['contours'][0]['y']

            # Get stem
            stem_x = ann['stem']['x']
            stem_y = ann['stem']['y']

            # Contour mask (roughy position of the plant)
            mask = np.zeros(shape=(rbgimg.shape[0], rbgimg.shape[1]), dtype="uint8")
            cv2.drawContours(mask, [np.array(list(zip(x, y)), dtype=np.int32)], -1, (255, 255, 255), -1)

            # Bitwise with RGB mask
            bitRgb = cv2.bitwise_and(imgrayRgb, imgrayRgb, mask=mask)
            _, contoursRgb, _ = cv2.findContours(bitRgb, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Bitwise with NIR mask
            bitNir = cv2.bitwise_and(imgrayNir, imgrayNir, mask=mask)
            _, contoursNir, _ = cv2.findContours(bitNir, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Final mask
            finalMask = np.zeros(shape=(rbgimg.shape[0], rbgimg.shape[1]), dtype="uint8")
            cv2.drawContours(finalMask, contoursRgb, -1, (255, 255, 255), -1)

            # Final Image
            final = cv2.bitwise_and(rbgimg, rbgimg, mask=finalMask)

            cv2.imshow('image', final)
            cv2.waitKey(0)
            cv2.destroyAllWindows()