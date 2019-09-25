import os
import yaml
import cv2
import numpy as np
from argparse import ArgumentParser
from glob import glob
import math
import shutil
import random
from main import *

def parseArgs():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='../../../plants_dataset/Bonn 2016/', help="Dataset path")
    parser.add_argument("--dataset_type", type=str, default='original', help="Types available [original, combined, synthetic]")

    return parser.parse_args()

# Flip image
def flip_image(img, mode = 1):
    # Horizontal = 1
    # Vertical = 0
    # Both = -1
    return cv2.flip(img, mode)

# Rotate image
def rotate_image(img, angle, shape):
    height = shape[0]
    width = shape[1]
    channels = shape[2]
    M = cv2.getRotationMatrix2D((width/2,height/2), angle, 1)
    return cv2.warpAffine(img,M,(width,height), borderMode=1)

# Return list of augmented images given one single image
def augment_image(img, shape):
    flip_list = [-1, 0, 1]
    images = [img]

    for flip in flip_list:
        flip_img = flip_image(img, flip)

        images.append(flip_img)

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

def generate_dataset(path, output_path, type="SugarBeets"):

    annotationsPath = 'annotations/YAML/'
    rgbImagesPath = 'images/rgb/'
    maskRgbPath = 'annotations/dlp/color/'

    imageNumber = 0

    dim = 256

    # Get folders
    folders = glob(path + '/*/')
    print('Number of folders:', len(folders))
    complete_radius_list = []
    cutted_images = 0

    # Load args from SPADE
    args = parse_args()
    if args is None:
      exit()

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        gan = spade(sess, args)

        # build graph
        gan.build_model()

        # Load model
        gan.load_model()

        for i, folder in enumerate(folders):
            # Get files
            files = os.listdir(folder + annotationsPath)
            print('\nFolder %d/%d: %s' %(i+1,len(folders),folder))

            for j, file in enumerate(files):
                print('\tFile %d/%d: %s' % (j + 1, len(files), file))
                with open(folder + annotationsPath + file, 'r') as stream:
                    # Image name
                    imageName = os.path.splitext(file)[0]

                    # Open images
                    rgbimg = cv2.imread(folder + rgbImagesPath + imageName + '.png', cv2.IMREAD_COLOR)
                    maskRgb = cv2.imread(folder + maskRgbPath + imageName + '.png', cv2.IMREAD_COLOR)
                    if rgbimg is None or maskRgb is None:
                        continue
                    maskRed = maskRgb[:, :, 2]  # Get only red channel
                    maskGreen = maskRgb[:, :, 1]  # Get only green channel

                    shape = rgbimg.shape

                    # Get content from yaml file
                    content = yaml.safe_load(stream)

                    # For each
                    try:
                        field = content["annotation"]
                    except:
                        print('\t\tError: Empty Yaml')
                        continue

                    # Blank mask
                    fullMask = np.zeros(shape=(rgbimg.shape[0], rgbimg.shape[1]), dtype="uint8")

                    # Radius list
                    radius_list = []

                    for ann in field:
                        if ann['type'] == type:

                            # Contours
                            x = ann['contours'][0]['x']
                            y = ann['contours'][0]['y']

                            # Get stem
                            stem_x = ann['stem']['x']
                            stem_y = ann['stem']['y']

                            # Draw plant on mask
                            cv2.drawContours(fullMask, [np.array(list(zip(x, y)), dtype=np.int32)], -1, (255, 255, 255), -1)

                            # Only consider if image is inside picture
                            if (stem_y > 0 and stem_x > 0):

                                # Contour mask (roughy position of the plant)
                                mask = np.zeros(shape=(rgbimg.shape[0], rgbimg.shape[1]), dtype="uint8")
                                cv2.drawContours(mask, [np.array(list(zip(x, y)), dtype=np.int32)], -1, (255, 255, 255), -1)

                                # Bitwise with RGB mask and most extreme points along the contour
                                bitRgb = cv2.bitwise_and(maskGreen, maskGreen, mask=mask)
                                _, contoursRgb, _ = cv2.findContours(bitRgb, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                                if not contoursRgb:
                                    continue

                                # Find maximum radius of the plant
                                ret, thresh = cv2.threshold(bitRgb, 127, 255, 0)
                                im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)
                                radius = find_max_radius(contours, stem_x, stem_y)
                                radius = int(radius * 1.1)

                                right = stem_x + radius
                                left = stem_x - radius
                                top = stem_y + radius
                                bot = stem_y - radius

                                # Check if crop is fully inside image
                                if bot > 0 and top < shape[0] and left > 0 and right < shape[1]:
                                    complete_radius_list.append(radius)

                                    # Crop images
                                    cropMask = bitRgb[bot:top, left:right]
                                    cropMaskInv = cv2.bitwise_not(cropMask)

                                    # Resize mask
                                    cropMaskResized = cv2.resize(cropMask, (dim, dim), interpolation=cv2.INTER_NEAREST)

                                    radius_list.append([right, left, top, bot, radius, cropMask, cropMaskInv, cropMaskResized])

                                else:
                                    cutted_images += 1

                            else:
                                cutted_images += 1

                    # Bitwise with RGB mask and most extreme points along the contour
                    fullMask = cv2.bitwise_not(fullMask)
                    maskRed = cv2.bitwise_and(maskRed, maskRed, mask=fullMask)


                    ones = np.ones(shape=(rgbimg.shape[0], rgbimg.shape[1]), dtype="uint8")
                    bitRed = cv2.bitwise_and(ones, ones, mask=maskRed)
                    bitGreen = cv2.bitwise_and(ones*2, ones*2, mask=maskRgb[:,:,1])

                    maskRgb = bitRed + bitGreen

                    # Augment images
                    rgbimg_ = augment_image(rgbimg, shape)
                    maskRgb_ = augment_image(maskRgb, shape)

                    # Number of folds for the original dataset compared to the original one
                    if len(radius_list) > 0:
                        for fold in range(4):
                            rgbimgCopy = rgbimg.copy()
                            for [right, left, top, bot, radius, cropMask, cropMaskInv, cropMaskResized] in radius_list:

                                # Generate image
                                synthetic = gan.generate_sample(cropMaskResized)
                                synthetic = cv2.cvtColor(synthetic, cv2.COLOR_BGR2RGB)
                                # synthetic = np.expand_dims(cropMaskResized, axis=2)
                                # synthetic = np.repeat(synthetic, 3, axis=2)
                                synthetic = cv2.resize(synthetic, (radius*2, radius*2), interpolation=cv2.INTER_AREA)

                                original = cv2.bitwise_and(rgbimgCopy[bot:top,left:right,:], rgbimgCopy[bot:top,left:right,:], mask=cropMaskInv)
                                synthetic = cv2.bitwise_and(synthetic, synthetic, mask=cropMask)
                                rgbimgCopy[bot:top,left:right,:] = cv2.add(original,synthetic)

                            # Augment generated image
                            rgbimg_ = augment_image(rgbimgCopy, shape)

                            # Write image
                            for k in range(len(maskRgb_)):
                                cv2.imwrite(output_path + 'train/synthetic/image/image_' + str(imageNumber) + '_' + str(fold) + '_' + str(k) + '.png', rgbimg_[k])
                                cv2.imwrite(output_path + 'train/synthetic/mask/image_' + str(imageNumber) + '_' + str(fold) + '_' + str(k) + '.png', maskRgb_[k])

                    for k in range(len(rgbimg_)):

                        cv2.imwrite(output_path + 'train/original/image/image_' + str(imageNumber) + '_' + str(k) + '.png', rgbimg_[k])
                        cv2.imwrite(output_path + 'train/original/mask/image_' + str(imageNumber) + '_' + str(k) + '.png', maskRgb_[k])
                    imageNumber += 1

    print(complete_radius_list)

    print("Cut images: ", str(cutted_images))

    above = 0
    below = 0

    for i in complete_radius_list:
        if i >= 128:
            above += 1

        else:
            below += 1

    print("Above: ", str(above))
    print("Below: ", str(below))

if __name__ == '__main__':
    args = parseArgs()

    folders = ['train/', 'test/']
    subfolers = ['original/', 'synthetic/']
    subsubfolers = ['image/', 'mask/']

    # output_path = '../../dataset/Segmentation/'
    output_path = '/Volumes/MAXTOR/Segmentation/'
    # Create folders if do not exist
    if os.path.exists(output_path):
        print('\nFolder', output_path, 'already exist, delete it before continue!\n')
    else:
        print('\nCreating folders!\n')

        os.makedirs(output_path)
        for f in folders:
            for s in subfolers:
                for w in subsubfolers:
                    os.makedirs(output_path + f + s + w)

        # Generate data
        generate_dataset(path=args.dataset_path, output_path=output_path)

        # Split train and test files
        for s in subfolers:
            files = os.listdir(output_path + folders[0] + s + subsubfolers[0])
            cut_files = random.sample(files, int(len(files)*0.2))

            for c in cut_files:
                for ss in subsubfolers:
                    shutil.move(output_path + folders[0] + s + ss + c, output_path + folders[1] + s + ss + c)