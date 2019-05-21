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

for file in files:
    #file = 'bonirob_2016-05-17-11-42-26_14_frame196.yaml'
    with open(path + annotationsPath + file, 'r') as stream:
        # Image name
        imageName = os.path.splitext(file)[0]
        print(imageName)

        # Open images
        rgbimg = cv2.imread(path + rgbImagesPath + imageName + '.png', 1)
        nirimg = cv2.imread(path + nirImagesPath + imageName + '.png', 1)
        maskNir = cv2.imread(path + maskNirPath + imageName + '.png', 1)
        maskRgb = cv2.imread(path + maskRgbPath + imageName + '.png', 1)
        # if rgbimg is None or nirimg is None or maskNir is None or maskRgb is None:
        #     continue
        imgrayRgb = maskRgb[:,:,1]
        imgrayNir = cv2.cvtColor(maskNir, cv2.COLOR_RGB2GRAY)

        # Image shape
        shape = rgbimg.shape

        # Get content from yaml file
        content = yaml.safe_load(stream)

        # For each
        for ann in content["annotation"]:
            if ann['type'] == 'SugarBeets':

                # Contours
                x = ann['contours'][0]['x']
                y = ann['contours'][0]['y']

                # Get stem
                stem_x = ann['stem']['x']
                stem_y = ann['stem']['y']

                # Plant id
                id = ann['plant_id']

                # Only consider if image is inside picture
                if (stem_y > 0 and stem_x > 0):

                    # Contour mask (roughy position of the plant)
                    mask = np.zeros(shape=(rgbimg.shape[0], rgbimg.shape[1]), dtype="uint8")
                    cv2.drawContours(mask, [np.array(list(zip(x, y)), dtype=np.int32)], -1, (255, 255, 255), -1)

                    # Bitwise with RGB mask and most extreme points along the contour
                    bitRgb = cv2.bitwise_and(imgrayRgb, imgrayRgb, mask=mask)
                    _, contoursRgb, _ = cv2.findContours(bitRgb, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                    if not contoursRgb:
                        continue
                    # cv2.drawContours(rgbimg, [np.array(list(zip(x, y)), dtype=np.int32)], 0, (0, 255, 0), 3)
                    # cv2.imshow('image', rgbimg)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    c = max(contoursRgb, key=cv2.contourArea)
                    leftRgb = tuple(c[c[:, :, 0].argmin()][0])[0]
                    rightRgb = tuple(c[c[:, :, 0].argmax()][0])[0]
                    topRgb = tuple(c[c[:, :, 1].argmin()][0])[1]
                    botRgb = tuple(c[c[:, :, 1].argmax()][0])[1]

                    # Bitwise with NIR mask and most extreme points along the contour
                    bitNir = cv2.bitwise_and(imgrayNir, imgrayNir, mask=mask)
                    _, contoursNir, _ = cv2.findContours(bitNir, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                    c = max(contoursNir, key=cv2.contourArea)
                    leftNir = tuple(c[c[:, :, 0].argmin()][0])[0]
                    rightNir = tuple(c[c[:, :, 0].argmax()][0])[0]
                    botNir = tuple(c[c[:, :, 1].argmin()][0])[1]
                    topNir = tuple(c[c[:, :, 1].argmax()][0])[1]

                    # Final mask
                    finalMask = np.zeros(shape=(shape[0], shape[1]), dtype="uint8")
                    cv2.drawContours(finalMask, contoursRgb, -1, (255, 255, 255), -1)
                    cv2.drawContours(finalMask, contoursNir, -1, (255, 255, 255), -1)

                    # Final image
                    finalRgb = cv2.bitwise_and(rgbimg, rgbimg, mask=finalMask)
                    finalNir = cv2.bitwise_and(nirimg, nirimg, mask=finalMask)

                    maxHorizontal = max(rightRgb - stem_x, rightNir - stem_x, stem_x - leftRgb, stem_x - leftNir)
                    maxVertical = max(topRgb - stem_y, topNir - stem_y, stem_y - botRgb, stem_y - botNir)
                    maxTot = int(max(maxHorizontal, maxVertical) * 1.1)
                    right = stem_x + maxTot
                    left = stem_x - maxTot
                    top = stem_y + maxTot
                    bot = stem_y - maxTot

                    print(maxTot)

                    if bot > 0 and top < shape[0] and left > 0 and right < shape[1] and maxTot > 200:
                        # Crop images
                        cropRgb = finalRgb[bot:top, left:right, :]
                        cropNir = finalNir[bot:top, left:right, :]

                        # backtorgb = cv2.cvtColor(bitNir, cv2.COLOR_GRAY2RGB)
                        cv2.rectangle(rgbimg, (right, bot), (left, top), (0, 255, 0), 1)
                        cv2.circle(rgbimg, (stem_x, stem_y), maxTot, (0, 255, 0), 1)

                        # cv2.imshow('image', rgbimg)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()

                        # Write image
                        cv2.imwrite('../dataset/RGB/' + imageName + '_' + str(id) + '.png', cropRgb)
                        cv2.imwrite('../dataset/NIR/' + imageName + '_' + str(id) + '.png', cropNir)