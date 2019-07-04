import os
import yaml
import cv2
import numpy as np
from argparse import ArgumentParser
import glob
import math
import shutil
import random

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("dataset_path", nargs='?', default='../dataset/Bonn 2016/', help="Dataset path")
    parser.add_argument("output_path", nargs='?', default='../dataset/Processed/', help="Output path")

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
    rotation_list = list(range(0, 360, 30))
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

def generate_dataset(path, output_path, dim = 512, smooth = False, save_images=False):

    annotationsPath = 'annotations/YAML/'
    nirImagesPath = 'images/nir/'
    rgbImagesPath = 'images/rgb/'
    maskNirPath = 'annotations/dlp/iMap/'
    maskRgbPath = 'annotations/dlp/color/'

    # Get folders
    folders = glob.glob(path + '/*/')
    print('Number of folders:', len(folders))
    radius_list = []

    for i, folder in enumerate(folders):
        # Get files
        files = os.listdir(folder + annotationsPath)
        number_files = len(files)
        print('\nFolder %d/%d: %s' %(i+1,len(folders),folder))
        print('\tNumber of files:', number_files)

        for j, file in enumerate(files):
            # file = 'bonirob_2016-05-18-10-59-09_4_frame1.yaml'
            print('\tFile %d/%d: %s' % (j + 1, len(files), file))
            with open(folder + annotationsPath + file, 'r') as stream:
                # Image name
                imageName = os.path.splitext(file)[0]

                # Open images
                rgbimg = cv2.imread(folder + rgbImagesPath + imageName + '.png', cv2.IMREAD_COLOR)
                nirimg = cv2.imread(folder + nirImagesPath + imageName + '.png', cv2.IMREAD_GRAYSCALE)
                maskNir = cv2.imread(folder + maskNirPath + imageName + '.png', cv2.IMREAD_GRAYSCALE)
                maskRgb = cv2.imread(folder + maskRgbPath + imageName + '.png', cv2.IMREAD_COLOR) # Get only green channel
                if rgbimg is None or nirimg is None or maskNir is None or maskRgb is None:
                    continue
                maskRgb = maskRgb[:, :,1]  # Get only green channel

                # Image shape
                shape = rgbimg.shape

                # Get content from yaml file
                content = yaml.safe_load(stream)

                # For each
                try:
                    field = content["annotation"]
                except:
                    print('\t\tError: Empty Yaml')
                    continue
                for ann in field:
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
                            bitRgb = cv2.bitwise_and(maskRgb, maskRgb, mask=mask)
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
                            if smooth:
                                finalMask = cv2.blur(finalMask, (5,5))

                            # Find maximum radius of the plant
                            ret, thresh = cv2.threshold(finalMask, 127, 255, 0)
                            im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)
                            radius = find_max_radius(contours, stem_x, stem_y)
                            radius_list.append(radius)
                            radius = int(radius * 1.1)

                            # Final image
                            finalRgb = cv2.bitwise_and(rgbimg, rgbimg, mask=finalMask)
                            finalNir = cv2.bitwise_and(nirimg, nirimg, mask=finalMask)

                            right = stem_x + radius
                            left = stem_x - radius
                            top = stem_y + radius
                            bot = stem_y - radius

                            if bot > 0 and top < shape[0] and left > 0 and right < shape[1] and radius > dim/2:
                                # Crop images
                                cropRgb = finalRgb[bot:top, left:right, :]
                                cropNir = finalNir[bot:top, left:right]
                                cropMask = finalMask[bot:top, left:right]

                                # Resize image
                                cropRgb = cv2.resize(cropRgb, (dim, dim), interpolation=cv2.INTER_AREA)
                                cropNir = cv2.resize(cropNir, (dim, dim), interpolation=cv2.INTER_AREA)
                                cropMask = cv2.resize(cropMask, (dim, dim), interpolation=cv2.INTER_AREA)

                                # Augment images
                                cropRgb_ = augment_image(cropRgb, dim)
                                cropNir_ = augment_image(cropNir, dim)
                                cropMask_ = augment_image(cropMask, dim)

                                # Write image
                                if save_images:
                                    for k in range(len(cropMask_)):
                                        cv2.imwrite(output_path + 'train/rgb/' + imageName + '_' + str(id) + '_' + str(k) + '.png', cropRgb_[k])
                                        cv2.imwrite(output_path + 'train/nir/' + imageName + '_' + str(id) + '_' + str(k) + '.png', cropNir_[k])
                                        cv2.imwrite(output_path + 'train/mask/' + imageName + '_' + str(id) + '_' + str(k) + '.png', cropMask_[k])

if __name__ == '__main__':
    args = parse_args()

    folders = ['train/', 'test/']
    subfolers = ['mask/', 'rgb/', 'nir/']
    # Create folders if do not exist
    if os.path.exists(args.output_path):
        print('\nFolder', args.output_path, 'already exist, delete it before continue!\n')
    else:
        print('\nCreating folders!\n')

        os.makedirs(args.output_path)
        for f in folders:
            for s in subfolers:
                os.makedirs(args.output_path + f + s)

        # Generate data
        generate_dataset(path=args.dataset_path, output_path=args.output_path, save_images=True)

        # Split train and test files
        files = os.listdir(args.output_path + folders[0] + 'mask/')
        cut_files = random.sample(files, 100)

        for c in cut_files:
            for s in subfolers:
                shutil.move(args.output_path + folders[0] + s + c, args.output_path + folders[1] + s + c)
