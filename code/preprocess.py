import os
import yaml
import cv2
import numpy as np
from argparse import ArgumentParser
import glob
import math
import shutil
import random

import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt

RGB_CAMERA_MATRIX = np.matrix([[2150.03686, 0.000000, 649.2291], [0.000000, 2150.03686, 480.680634], [0.000000, 0.000000, 1.000000]])
RGB_PROJECTION_MATRIX = np.matrix([[2150.03686, 0.000000, 649.2291, 0.000000], [0.000000, 2150.03686, 480.680634, 0.000000], [0.000000, 0.000000, 1.000000, 0.000000]])
RGB_DISTORT_COEFFICIENTS = np.matrix([0.0, 0.0, 0.0, 0.0, 0.0])

NIR_CAMERA_MATRIX = np.matrix([[2162.2948, 0.000000, 650.22019], [0.000000, 2162.2948, 481.20451], [0.000000, 0.000000, 1.000000]])
NIR_PROJECTION_MATRIX = np.matrix([[2162.2948, 0.000000, 650.22019, 0.000000], [0.000000, 2162.2948, 481.20451, 0.000000], [0.000000, 0.000000, 1.000000, 0.000000]])
NIR_DISTORT_COEFFICIENTS = np.matrix([0.0, 0.0, 0.0, 0.0, 0.0])

TRANSLATION = RGB_CAMERA_MATRIX[0:2,2] - NIR_CAMERA_MATRIX[0:2,2]

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='../../plants_dataset/Bonn 2016/', help="Dataset path")
    # parser.add_argument("--dataset_path", type=str, default='../../plants_dataset/SugarBeets_256/train/', help="Dataset path")
    parser.add_argument("--plant_type", type=str, default='SugarBeets', help="Output path")
    parser.add_argument("--dimension", type=int, default=256, help="Image dimension")

    return parser.parse_args()

# Flip image
def flip_image(img, mode = 1):
    # Horizontal = 1
    # Vertical = 0
    # Both = -1
    return cv2.flip(img, mode)

# Rotate image
def rotate_image(img, angle, dim):
    M = cv2.getRotationMatrix2D((dim/2,dim/2), angle, 1)
    return cv2.warpAffine(img,M,(dim,dim), borderMode=1)

# Return list of augmented images given one single image
def augment_image(img, dim):
    flip_list = [-1, 0, 1]
    rotation_list = list(range(0, 360, 90))
    images = []

    for flip in flip_list:
        flip_img = flip_image(img, flip)

        for rotation in rotation_list:
            rotation_img = rotate_image(flip_img, rotation, dim)

            images.append(rotation_img)

    return images

# Calculate maximum radius in pixel given contour and center stem
def find_max_radius(contours, stem_x, stem_y):
    dist = 0
    for c in contours:
        for point in c:
            point = point[0]
            x = point[0] - stem_x
            y = point[1] - stem_y
            new_dist = math.ceil(math.sqrt(math.pow(x, 2) + math.pow(y, 2)))
            if new_dist > dist:
                dist = new_dist

    return dist

def get_alignment_parameters(img2, img1):

    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m in matches:
        if m[0].distance < 0.5 * m[1].distance:
            good.append(m)
    matches = np.asarray(good)

    if len(matches[:, 0]) >= 4:
        src = np.float32([kp1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        # print H
    else:
        print("\t\tCan't find enough keypoints.")
        H = None

    return H

def generate_dataset(path, output_path, dim = 256, type='SugarBeets', smooth = False, save_images=False):

    annotationsPath = 'annotations/YAML/'
    nirImagesPath = 'images/nir/'
    rgbImagesPath = 'images/rgb/'
    maskNirPath = 'annotations/dlp/iMap/'
    maskRgbPath = 'annotations/dlp/color/'

    imageNumber = 0

    # Get folders
    folders = glob.glob(path + '/*/')
    # folders = ['../../plants-dataset/Bonn 2016/CKA_160523/']
    print('Number of folders:', len(folders))
    radius_list = []

    for i, folder in enumerate(folders):
        # Get files
        files = os.listdir(folder + annotationsPath)
        # files = ['bonirob_2016-05-23-10-57-33_4_frame157.yaml']
        number_files = len(files)
        print('\nFolder %d/%d: %s' %(i+1,len(folders),folder))
        print('\tNumber of files:', number_files)

        for j, file in enumerate(files):
            print('\tFile %d/%d: %s' % (j + 1, len(files), file))
            with open(folder + annotationsPath + file, 'r') as stream:
                # Image name
                imageName = os.path.splitext(file)[0]

                # Open images
                rgbimg = cv2.imread(folder + rgbImagesPath + imageName + '.png', cv2.IMREAD_COLOR)
                nirimg = cv2.imread(folder + nirImagesPath + imageName + '.png', cv2.IMREAD_GRAYSCALE)
                maskNir = cv2.imread(folder + maskNirPath + imageName + '.png', cv2.IMREAD_GRAYSCALE)
                maskRgb = cv2.imread(folder + maskRgbPath + imageName + '.png', cv2.IMREAD_COLOR)
                if rgbimg is None or nirimg is None or maskNir is None or maskRgb is None:
                    continue
                maskRgb = maskRgb[:, :,1]  # Get only green channel

                # Undistort NIR
                nirimg = cv2.undistort(nirimg, NIR_CAMERA_MATRIX, NIR_DISTORT_COEFFICIENTS, None, RGB_CAMERA_MATRIX)
                maskNir = cv2.undistort(maskNir, NIR_CAMERA_MATRIX, NIR_DISTORT_COEFFICIENTS, None, RGB_CAMERA_MATRIX)

                # Image shape
                shape = rgbimg.shape

                # Get parameter to superpose images
                try:
                    H = get_alignment_parameters(cv2.cvtColor(rgbimg, cv2.COLOR_BGR2GRAY), nirimg)
                except:
                    H = None

                if H is None:
                    print('\t\tError: Empty H matrix')
                    continue

                # print(np.sum(np.abs(H)))
                if np.sum(np.abs(H)) > 10:
                    print('\t\tError: Error in alignment')
                    continue

                nirimg = cv2.warpPerspective(nirimg, H, (shape[1], shape[0]))
                maskNir = cv2.warpPerspective(maskNir, H, (shape[1], shape[0]))

                # rgbgray = cv2.cvtColor(rgbimg, cv2.COLOR_BGR2GRAY)
                # nirgray = nirimg
                #
                # scale_percent = 90  # percent of original size
                # width = int(shape[1] * scale_percent / 100)
                # height = int(shape[0] * scale_percent / 100)
                # dim = (width, height)
                #
                # #### BEFORE #####
                # before = cv2.add(rgbgray, nirgray)
                #
                # cv2.imshow('BEFORE', cv2.resize(before, dim, interpolation=cv2.INTER_AREA))
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                #
                # #### AFTER #####
                # # rgbUndistorted = cv2.undistort(rgbgray, RGB_CAMERA_MATRIX, RGB_DISTORT_COEFFICIENTS, None, RGB_CAMERA_MATRIX)
                # # nirUndistorted = cv2.undistort(nirimg, NIR_CAMERA_MATRIX, NIR_DISTORT_COEFFICIENTS, None, RGB_CAMERA_MATRIX)
                #
                # after = cv2.add(rgbgray, dst)
                #
                # cv2.imshow('AFTER', cv2.resize(after, dim, interpolation=cv2.INTER_AREA))
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                #
                # # print(np.sum(nirgray))
                # # print(np.sum(nirUndistorted))

                # Get content from yaml file
                content = yaml.safe_load(stream)

                # For each
                try:
                    field = content["annotation"]
                except:
                    print('\t\tError: Empty Yaml')
                    continue
                for ann in field:
                    if ann['type'] == type:

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
                            cv2.imwrite('../images/mask.png', mask)
                            cv2.drawContours(mask, [np.array(list(zip(x, y)), dtype=np.int32)], -1, (255, 255, 255), -1)
                            cv2.imwrite('../images/contour.png', mask)

                            # Bitwise with RGB mask and most extreme points along the contour
                            bitRgb = cv2.bitwise_and(maskRgb, maskRgb, mask=mask)
                            cv2.imwrite('../images/bitRgb.png', bitRgb)
                            _, contoursRgb, _ = cv2.findContours(bitRgb, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                            if not contoursRgb:
                                continue

                            # test = cv2.add(cv2.cvtColor(rgbimg, cv2.COLOR_BGR2GRAY), mask)
                            # cv2.imshow('BEFORE', bitRgb)
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()

                            # Bitwise with NIR mask and most extreme points along the contour
                            bitNir = cv2.bitwise_and(maskNir, maskNir, mask=mask)
                            cv2.imwrite('../images/bitNir.png', (cv2.bitwise_and(maskNir, maskNir, mask=mask)/np.max(cv2.bitwise_and(maskNir, maskNir, mask=mask))*255).astype(int))
                            _, contoursNir, _ = cv2.findContours(bitNir, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

                            # Final mask
                            finalMask = np.zeros(shape=(shape[0], shape[1]), dtype="uint8")
                            cv2.drawContours(finalMask, contoursRgb, -1, (255, 255, 255), -1)
                            cv2.drawContours(finalMask, contoursNir, -1, (255, 255, 255), -1)
                            cv2.imwrite('../images/finalMask.png', finalMask)
                            if smooth:
                                finalMask = cv2.blur(finalMask, (5,5))

                            # Find maximum radius of the plant
                            ret, thresh = cv2.threshold(finalMask, 127, 255, 0)
                            im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)
                            radius = find_max_radius(contours, stem_x, stem_y)
                            radius_list.append(radius)
                            radius = int(radius * 1.1)

                            # Final image
                            finalRgb = rgbimg #cv2.bitwise_and(rgbimg, rgbimg, mask=finalMask)
                            finalNir = nirimg #cv2.bitwise_and(nirimg, nirimg, mask=finalMask)

                            right = stem_x + radius
                            left = stem_x - radius
                            top = stem_y + radius
                            bot = stem_y - radius

                            if bot > 0 and top < shape[0] and left > 0 and right < shape[1] and radius >= dim/2:# and radius < dim:
                                # Crop images
                                cropRgb = finalRgb[bot:top, left:right, :]
                                cropNir = finalNir[bot:top, left:right]
                                cropMask = finalMask[bot:top, left:right]

                                cv2.imwrite('../images/cropRgb.png', cropRgb)
                                cv2.imwrite('../images/cropNir.png', cropNir)
                                cv2.imwrite('../images/cropMask.png', cropMask)

                                # cv2.imwrite(
                                #     output_path + 'train/mask/' + imageName + '_' + str(id) + '_' + '.png',
                                #     cropMask)

                                # Resize image
                                cropRgb = cv2.resize(cropRgb, (dim, dim), interpolation=cv2.INTER_AREA)
                                cropNir = cv2.resize(cropNir, (dim, dim), interpolation=cv2.INTER_AREA)
                                cropMask = cv2.resize(cropMask, (dim, dim), interpolation=cv2.INTER_NEAREST)

                                # Augment images
                                cropRgb_ = augment_image(cropRgb, dim)
                                cropNir_ = augment_image(cropNir, dim)
                                cropMask_ = augment_image(cropMask, dim)

                                # cropRgb_ = [cropRgb]
                                # cropNir_ = [cropNir]
                                # cropMask_ = [cropMask]

                                # Write image
                                if save_images:
                                    for k in range(len(cropMask_)):
                                        # cv2.imwrite(output_path + 'train/rgb/' + imageName + '_' + str(id) + '_' + str(k) + '.png', cropRgb_[k])
                                        # cv2.imwrite(output_path + 'train/nir/' + imageName + '_' + str(id) + '_' + str(k) + '.png', cropNir_[k])
                                        # cv2.imwrite(output_path + 'train/mask/' + imageName + '_' + str(id) + '_' + str(k) + '.png', cropMask_[k])

                                        # cv2.imwrite(output_path + 'train/rgb/rgb_' + str(imageNumber) + '.png', cropRgb_[k])
                                        # cv2.imwrite(output_path + 'train/nir/nir_' + str(imageNumber) + '.png', cropNir_[k])
                                        cv2.imwrite(output_path + 'train/mask/mask_' + str(imageNumber) + '.png', cropMask_[k])
                                        imageNumber += 1

                                        cv2.imwrite('../images/cropRgb' + str(imageNumber) + '.png', cropRgb_[k])
                                        cv2.imwrite('../images/cropNir' + str(imageNumber) + '.png', cropNir_[k])
                                        cv2.imwrite('../images/cropMask' + str(imageNumber) + '.png', cropMask_[k])

                                        print('done')

                                        # cv2.imwrite(output_path + 'train/rgb/rgb_' + str(imageNumber) + '.png', cropRgb_[k])
                                        # imageNumber += 1
                                        # cv2.imwrite(output_path + 'train/rgb/rgb_' + str(imageNumber) + '.png', cropNir_[k])
                                        # imageNumber += 1

if __name__ == '__main__':
    args = parse_args()

    folders = ['train/', 'test/']
    subfolers = ['mask/', 'rgb/', 'nir/']

    output_path = '../dataset/' + args.plant_type + '_' + str(args.dimension) + '/'
    # Create folders if do not exist
    if os.path.exists(output_path):
        print('\nFolder', output_path, 'already exist, delete it before continue!\n')
    else:
        print('\nCreating folders!\n')

        os.makedirs(output_path)
        for f in folders:
            for s in subfolers:
                os.makedirs(output_path + f + s)

        # Generate data
        generate_dataset(path=args.dataset_path, output_path=output_path, dim=args.dimension, type=args.plant_type, smooth = False, save_images=True)

        # Split train and test files
        files = os.listdir(output_path + folders[0] + 'mask/')
        cut_files = random.sample(files, 200)

        for c in cut_files:
            fileNumber = c.split("_")[1]
            for s in subfolers:
                folderName = s[0:-1]
                shutil.move(output_path + folders[0] + s + s[0:-1] + '_' + fileNumber, output_path + folders[1] + s + s[0:-1] + '_' + fileNumber)

    # # Split train and test files
    # files = os.listdir(args.dataset_path + 'blur/')
    # cut_files = random.sample(files, 200)
    #
    # for s in cut_files:
    #     shutil.move('../../plants_dataset/SugarBeets_256/train/blur/' + s,
    #                 '../../plants_dataset/SugarBeets_256/test/blur/' + s)
    #
    # # Split train and test files
    # files = os.listdir(output_path + folders[0] + 'mask/')
    # cut_files = random.sample(files, 200)

    # for c in cut_files:
    #             for s in subfolers:
    #                 shutil.move(output_path + folders[0] + s + c, output_path + folders[1] + s + c)