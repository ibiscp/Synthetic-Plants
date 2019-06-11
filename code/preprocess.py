import os
import yaml
import cv2
import numpy as np
from argparse import ArgumentParser
import tensorflow as tf
import sys
import glob

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("dataset_path", nargs='?', default='../dataset/Bonn 2016 Median/CKA_160517/', help="Dataset path")
    parser.add_argument("output_path", nargs='?', default='../dataset/Smoother/', help="Output path")

    return parser.parse_args()

def flip_image(img, mode = 1):
    # Horizontal = 1
    # Vertical = 0
    # Both = -1
    return cv2.flip(img, mode)

def rotate_image(img, angle):
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
    return cv2.warpAffine(img,M,(cols,rows), borderMode=1)

def augment_image(img):
    flip_list = [-1, 0, 1]
    rotation_list = [0, 90, 180, 270]
    images = []

    for flip in flip_list:
        flip_img = flip_image(img, flip)

        for rotation in rotation_list:
            rotation_img = rotate_image(flip_img, rotation)

            images.append(rotation_img)

    return images

def generate_dataset(path, output_path, dim = (128, 128), smooth = True, save_images=False):

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
            #print(imageName)

            # Open images
            rgbimg = cv2.cvtColor(cv2.imread(path + rgbImagesPath + imageName + '.png', 1), cv2.COLOR_BGR2RGB)
            nirimg = cv2.cvtColor(cv2.imread(path + nirImagesPath + imageName + '.png', 1), cv2.COLOR_BGR2RGB) #TODO convert to grayscale
            maskNir = cv2.imread(path + maskNirPath + imageName + '.png', 1)
            maskRgb = cv2.imread(path + maskRgbPath + imageName + '.png', 1)
            # if rgbimg is None or nirimg is None or maskNir is None or maskRgb is None:
            #     continue
            imgrayRgb = maskRgb[:, :, 1]
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
                        if smooth:
                            finalMask = cv2.blur(finalMask, (5,5))

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
                            cropMask = finalMask[bot:top, left:right]

                            # backtorgb = cv2.cvtColor(bitNir, cv2.COLOR_GRAY2RGB)
                            # cv2.rectangle(rgbimg, (right, bot), (left, top), (0, 255, 0), 1)
                            # cv2.circle(rgbimg, (stem_x, stem_y), maxTot, (0, 255, 0), 1)

                            # cv2.imshow('image', rgbimg)
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()

                            # Resize image
                            cropRgb = cv2.resize(cropRgb, dim, interpolation=cv2.INTER_AREA)
                            cropNir = cv2.resize(cropNir, dim, interpolation=cv2.INTER_AREA)
                            cropMask = cv2.resize(cropMask, dim, interpolation=cv2.INTER_AREA)

                            cropMask_ = augment_image(cropMask)

                            for i in range(len(cropMask_)):
                                cv2.imwrite('../dataset/test/' + imageName + '_' + str(id) + '_' + str(i) + '.png', cropMask_[i])

                            # # Write image
                            # if save_images:
                            #     #cv2.imwrite(output_path + 'RGB/' + imageName + '_' + str(id) + '.png', cropRgb)
                            #     #cv2.imwrite(output_path + 'NIR/' + imageName + '_' + str(id) + '.png', cropNir)
                            #     #cv2.imwrite(output_path + 'Mask/' + imageName + '_' + str(id) + '.png', cropMask)
                            #     cv2.imwrite('../dataset/test/' + imageName + '_' + str(id) + '.png', cropMask)

                            #yield cropRgb, cropNir, cropMask, imageName + '_' + str(id) + '.png'

def image2tfrecords(img):
    str = tf.compat.as_bytes(img.tostring())
    bts = tf.train.BytesList(value=[str])
    feature = tf.train.Feature(bytes_list=bts)

    return feature

if __name__ == '__main__':
    args = parse_args()

    # open the TFRecords file
    data_path = '../dataset/train.tfrecords'  # address to save the TFRecords file

    # Load data
    generate_dataset(path = args.dataset_path, output_path=args.output_path, save_images=True)

    #writer = tf.python_io.TFRecordWriter(data_path)

    # for i in generator:
    #     rgb = i[0]
    #     nir = i[1]
    #     mask = i[2]
    #     name = i[3]
    #
    #     feature = {name + '/rgb': image2tfrecords(rgb),
    #                name + '/nir': image2tfrecords(nir),
    #                name + '/mask': image2tfrecords(mask)}
    #
    #     # Create an example protocol buffer
    #     example = tf.train.Example(features=tf.train.Features(feature=feature))
    #
    #     # Serialize to string and write on the file
    #     writer.write(example.SerializeToString())
    #
    # writer.close()
    # sys.stdout.flush()




    # with tf.Session() as sess:
    #     feature = {'train/image': tf.FixedLenFeature([], tf.string),
    #                'train/label': tf.FixedLenFeature([], tf.int64)}
    #
    #     # Create a list of filenames and pass it to a queue
    #     filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
    #
    #     # Define a reader and read the next record
    #     reader = tf.TFRecordReader()
    #     _, serialized_example = reader.read(filename_queue)
    #
    #     # Decode the record read by the reader
    #     features = tf.parse_single_example(serialized_example, features=feature)
    #
    #     # Convert the image data from string back to the numbers
    #     image = tf.decode_raw(features['train/image'], tf.float32)
    #
    #     # Reshape image data into the original shape
    #     image = tf.reshape(image, [224, 224, 3])
    #
    #     # Any preprocessing here ...
    #
    #     # Creates batches by randomly shuffling tensors
    #     images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1,
    #                                             min_after_dequeue=10)
