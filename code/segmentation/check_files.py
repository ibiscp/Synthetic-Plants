import os
import cv2
from glob import glob
import numpy as np

path = "/Volumes/MAXTOR/Segmentation/"

type = "original/"

train_images = os.path.join(path, "test/", type, 'mask/')
files = glob(train_images + '*.png')
size = len(files)
error = 0

for i, file in enumerate(files):
    print(str(i) + '/' + str(size) + ' ' + file)
    mask = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    if np.max(mask) > 2:
        error += 1
        print('Error', str(error))

        # mask[mask > 2] = 2
        #
        # cv2.imwrite(file, mask)

print('Total error:', str(error))
