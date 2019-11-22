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

import sys
sys.path.append('../')
from utils import *

def parseArgs():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='../../../plants_dataset/Bonn 2016/', help="Dataset path")
    parser.add_argument("--annotation_path", type=str, default='../../../sugar_beet_annotation/', help="Annotation path")
    parser.add_argument("--output_path", type=str, default='../../../plants_dataset/Segmentation/', help="Output path")
    parser.add_argument("--background", type=str2bool, default=False, help="Keep (true) or remove (false) background")
    parser.add_argument("--blur", type=str2bool, default=True, help="Remove background with blur")

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

# Calculate stem if crop is outside image and max and min values for coordinates
def calculateStem(contours, stem_x, stem_y):

    m_x = [10e3, -10e3]
    m_y = [10e3, -10e3]

    for c in contours:
        for point in c:
            point = point[0]

            m_x = [min(m_x[0], point[0]), max(m_x[1], point[0])]
            m_y = [min(m_y[0], point[1]), max(m_y[1], point[1])]

    # if stem_x < 0:
    #     stem_x = int(m_x[0] + (m_x[1] - m_x[0])/2)
    #
    # if stem_y < 0:
    #     stem_y = int(m_y[0] + (m_y[1] - m_y[0]) / 2)

    return stem_x, stem_y, m_x, m_y

def generate_dataset(path, output_path, annotation_path, background, blur, type="SugarBeets"):

    annotationsPath = os.path.join(annotation_path, 'yamls/')
    nirImagesPath = 'images/nir/'
    rgbImagesPath = 'images/rgb/'
    maskNirPath = os.path.join(annotation_path, 'masks/iMap/')
    maskRgbPath = os.path.join(annotation_path, 'masks/color/')

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
            files = os.listdir(folder + rgbImagesPath)
            print('\nFolder %d/%d: %s' %(i+1,len(folders),folder))

            for j, file in enumerate(files):

                yaml_file = annotationsPath + file.split('.')[0] + '.yaml'

                print('\tFile %d/%d: %s' % (j + 1, len(files), yaml_file))

                if not os.path.isfile(yaml_file):
                    print('\t\tError: YAML does not exist')
                    continue

                with open(yaml_file, 'r') as stream:
                    # Image name
                    imageName = os.path.splitext(file)[0]

                    # Open images
                    rgbimg = cv2.imread(folder + rgbImagesPath + imageName + '.png', cv2.IMREAD_COLOR)
                    nirimg = cv2.imread(folder + nirImagesPath + imageName + '.png', cv2.IMREAD_GRAYSCALE)
                    maskRgb = cv2.imread(maskRgbPath + imageName + '.png', cv2.IMREAD_COLOR)
                    maskNir = cv2.imread(maskNirPath + imageName + '.png', cv2.IMREAD_GRAYSCALE)

                    if rgbimg is None or nirimg is None or maskNir is None or maskRgb is None:
                        print('\t\tError: Image does not exist')
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

                    # Undistort images
                    flag, nirimg, maskNir = align_images(rgbimg, nirimg, maskNir)

                    if flag:
                        continue

                    # Blank mask
                    maskCrop = np.zeros(shape=(rgbimg.shape[0], rgbimg.shape[1]), dtype="uint8")

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
                            cv2.drawContours(maskCrop, [np.array(list(zip(x, y)), dtype=np.int32)], -1, (255, 255, 255), -1)

                            # # Only consider if image is inside picture
                            if (stem_y > 0 and stem_x > 0):

                                # Contour mask (roughy position of the plant)
                                mask = np.zeros(shape=(rgbimg.shape[0], rgbimg.shape[1]), dtype="uint8")
                                cv2.drawContours(mask, [np.array(list(zip(x, y)), dtype=np.int32)], -1, (255, 255, 255), -1)

                                # Bitwise with RGB mask and most extreme points along the contour
                                bitRgb = cv2.bitwise_and(maskGreen, maskGreen, mask=mask)
                                _, contoursRgb, _ = cv2.findContours(bitRgb, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                                if not contoursRgb:
                                    continue

                                # Bitwise with NIR mask and most extreme points along the contour
                                bitNir = cv2.bitwise_and(maskNir, maskNir, mask=mask)
                                _, contoursNir, _ = cv2.findContours(bitNir, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

                                # Final mask
                                finalMask = np.zeros(shape=(shape[0], shape[1]), dtype="uint8")
                                cv2.drawContours(finalMask, contoursRgb, -1, (255, 255, 255), -1)
                                cv2.drawContours(finalMask, contoursNir, -1, (255, 255, 255), -1)

                                # Find maximum radius of the plant
                                ret, thresh = cv2.threshold(finalMask, 127, 255, 0)
                                im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)

                                # Calculate stem if not given
                                stem_x, stem_y, m_x, m_y = calculateStem(contours, stem_x, stem_y)

                                radius = find_max_radius(contours, stem_x, stem_y)
                                radius = int(radius * 1.1)

                                right = m_x[1]
                                left = m_x[0]
                                top = m_y[1]
                                bot = m_y[0]

                                complete_radius_list.append(radius)

                                # Crop images
                                cropMask = np.zeros(shape=(2*radius, 2*radius), dtype="uint8")
                                cropMask[radius-(stem_y-bot):radius+(top-stem_y), radius-(stem_x-left):radius+(right-stem_x)]=finalMask[bot:top, left:right]

                                # Resize mask
                                cropMaskResized = cv2.resize(cropMask, (dim, dim), interpolation=cv2.INTER_NEAREST)

                                radius_list.append([m_x, m_y, stem_x, stem_y, radius, cropMask, cropMaskResized])
                            else:
                                cutted_images += 1


                    # Bitwise with RGB mask and most extreme points along the contour
                    maskWeed = cv2.bitwise_not(maskCrop) # not crop

                    crop = cv2.bitwise_and(maskGreen, maskGreen, mask=maskCrop)
                    weed = cv2.bitwise_and(maskRed, maskRed, mask=maskWeed)

                    maskRgb = (crop/127.5 + weed/255).astype(np.uint8)

                    # Augment images
                    rgbimg_ = augment_image(rgbimg, shape)
                    nirimg_ = augment_image(nirimg, shape)
                    mask_ = augment_image(maskRgb, shape)

                    # Original images
                    for k in range(len(mask_)):

                        cv2.imwrite(output_path + 'train/original/rgb/image_' + str(imageNumber) + '_' + str(k) + '.png', rgbimg_[k])
                        cv2.imwrite(output_path + 'train/original/nir/image_' + str(imageNumber) + '_' + str(k) + '.png', nirimg_[k])
                        cv2.imwrite(output_path + 'train/original/mask/image_' + str(imageNumber) + '_' + str(k) + '.png', mask_[k])

                    # Number of folds for the original dataset compared to the original one
                    if len(radius_list) > 0:
                        for fold in range(4):
                            rgbimgCopy = rgbimg.copy()
                            nirimgCopy = nirimg.copy()
                            for [m_x, m_y, stem_x, stem_y, radius, cropMask, cropMaskResized] in radius_list:

                                right = m_x[1]
                                left = m_x[0]
                                top = m_y[1]
                                bot = m_y[0]

                                # # Generate image
                                synthetic_rgb, synthetic_nir = gan.generate_sample(cropMaskResized)
                                synthetic_rgb = cv2.cvtColor(synthetic_rgb, cv2.COLOR_BGR2RGB)
                                # Used for test only
                                # synthetic_nir = cropMaskResized
                                # synthetic_rgb = np.expand_dims(synthetic_nir, axis=2)
                                # synthetic_rgb = np.repeat(synthetic_rgb, 3, axis=2)

                                synthetic_rgb = cv2.resize(synthetic_rgb, (radius*2, radius*2), interpolation=cv2.INTER_AREA)
                                synthetic_rgb = synthetic_rgb[radius - (stem_y - bot): radius + (top - stem_y),
                                            radius - (stem_x - left): radius + (right - stem_x), :]

                                synthetic_nir = cv2.resize(synthetic_nir, (radius*2, radius*2), interpolation=cv2.INTER_AREA)
                                synthetic_nir = synthetic_nir[radius - (stem_y - bot): radius + (top - stem_y),
                                            radius - (stem_x - left): radius + (right - stem_x)]

                                if blur:
                                    cropMask = cv2.blur(cropMask, (5, 5))

                                if not background:

                                    original_rgb = rgbimgCopy[bot:top, left:right, :]
                                    original_nir = nirimgCopy[bot:top, left:right]

                                    original = np.concatenate((original_rgb, np.expand_dims(original_nir, axis=2)), axis=2)
                                    synthetic = np.concatenate((synthetic_rgb, np.expand_dims(synthetic_nir, axis=2)), axis=2)

                                    mask = cropMask[radius - (stem_y - bot): radius + (top - stem_y),
                                                    radius - (stem_x - left): radius + (right - stem_x)]

                                    blended = blend_with_mask_matrix(synthetic, original, mask)
                                    rgbimgCopy[bot:top,left:right,:] = blended[:,:,0:3]
                                    nirimgCopy[bot:top,left:right] = blended[:,:,3]

                                else:
                                    rgbimgCopy[bot:top, left:right, :] = synthetic_rgb
                                    nirimgCopy[bot:top, left:right] = synthetic_nir

                                # cv2.imshow('BEFORE', rgbimgCopy)
                                # cv2.waitKey(0)
                                # cv2.destroyAllWindows()

                            # Augment generated image
                            rgbimg_ = augment_image(rgbimgCopy, shape)
                            nirimg_ = augment_image(nirimgCopy, shape)

                            # Write image
                            for k in range(len(mask_)):
                                cv2.imwrite(output_path + 'train/synthetic/rgb/image_' + str(imageNumber) + '_' + str(fold) + '_' + str(k) + '.png', rgbimg_[k])
                                cv2.imwrite(output_path + 'train/synthetic/nir/image_' + str(imageNumber) + '_' + str(fold) + '_' + str(k) + '.png', nirimg_[k])
                                cv2.imwrite(output_path + 'train/synthetic/mask/image_' + str(imageNumber) + '_' + str(fold) + '_' + str(k) + '.png', mask_[k])

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
    subsubfolers = ['rgb/', 'nir/', 'mask/']

    output_path = args.output_path # #'../../dataset/Segmentation/'
    # output_path = '/Volumes/MAXTOR/Segmentation/'
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
        generate_dataset(path=args.dataset_path, output_path=output_path, annotation_path=args.annotation_path, background=args.background, blur=args.blur)

        # Split original train and test files
        # for s in subfolers:
        s = 'original/'
        files = os.listdir(output_path + folders[0] + s + subsubfolers[0])
        cut_files = random.sample(files, int(len(files)*0.2))

        for c in cut_files:
            for ss in subsubfolers:
                shutil.move(output_path + folders[0] + s + ss + c, output_path + folders[1] + s + ss + c)
