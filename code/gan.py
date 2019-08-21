# Import GANs
from dcgan import DCGAN
from wgangp import WGANGP

# Other libraries
import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow.contrib.gan as tfgan
import time
import gc
import glob
import re
from help import *
import math
from pytorchMetrics import *
from operator import itemgetter

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

GIF_MATRIX = 5

class GAN():
    def __init__(self, architecture, train_dataset, test_dataset, shape, **kwargs):

        self.architecture = architecture
        self.latent_dim = kwargs['latent_dim']
        self.batch_size = kwargs['batch_size']
        self.epochs = kwargs['epochs']
        self.kwargs = kwargs

        # Name
        self.name = architecture + ''.join('-{}-{}'.format(key, val) for key, val in sorted(kwargs.items()))

        # Directories
        self.directory = '../resources/' + self.name + '/'
        self.gif_dir = self.directory + 'gif/'
        self.model_dir = self.directory + 'model/'
        self.summary = '../resources/summary/' + self.name

        # Training parameters
        self.epoch = 0
        self.batch = 0
        self.metrics = []

        # Create directories
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            os.makedirs(self.gif_dir)
            os.makedirs(self.model_dir)

        # Load data
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.kwargs['img_shape'] = shape

        # Load the seed for the gif
        if not os.path.isfile(self.directory + '../' + str(self.latent_dim) + '.pkl'):
            self.gif_generator = np.random.normal(0, 1, (GIF_MATRIX ** 2, self.latent_dim))
            save(self.gif_generator, self.directory + '../' + str(self.latent_dim) + '.pkl')
        else:
            self.gif_generator = load(self.directory + '../' + str(self.latent_dim) + '.pkl')

        # Load the correct gan
        self.load_gan()

    def load_gan(self):

        # Load gan
        self.gan = globals()[self.architecture](**self.kwargs)

        # Get all the models saved
        files = glob.glob(self.model_dir + '*.h5')

        # Load Checkpoint if found
        if files:
            versions = []
            regex = re.compile(r'\d+')

            for filename in files:
                versions.append([int(x) for x in regex.findall(filename)][0])
            version = max(versions)

            self.gan.load(self.model_dir, version)

            self.load_checkpoint()

    def save_checkpoint(self):

        data = vars(self).copy()
        del data['gif_generator']
        del data['gan']
        del data['train_dataset']
        del data['test_dataset']

        save(data, self.directory + 'checkpoint.pkl')

    def load_checkpoint(self):

        data = load(self.directory + 'checkpoint.pkl')

        for key, value in data.items():
            setattr(self, key, value)

    def plot_gif(self, epoch):

        gen_imgs = self.gan.generator.predict(self.gif_generator)
        gen_imgs = np.sign(gen_imgs)
        gen_imgs = (0.5 * gen_imgs + 0.5) * 255

        plt.figure(figsize=(GIF_MATRIX, GIF_MATRIX))
        gs1 = gridspec.GridSpec(GIF_MATRIX, GIF_MATRIX)
        gs1.update(wspace=0.025, hspace=0.025)
        for i in range(gen_imgs.shape[0]):
            ax = plt.subplot(gs1[i])
            ax.imshow(gen_imgs[i,:,:,0], cmap='gray', interpolation='nearest')
            ax.axis('off')
            ax.set_aspect('equal')
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_frame_on(False)

        plt.savefig(self.gif_dir + "%d.png" % epoch, bbox_inches='tight', pad_inches=0.025)
        plt.close()

    def batch_generator(self, samples):

        ids = np.arange(len(self.train_dataset))
        # l = len(ids)
        # miss = np.random.choice(ids, self.batch_size - l % self.batch_size)
        # ids = np.hstack((ids, miss))

        np.random.shuffle(ids)

        for batch in range(0, samples, self.batch_size):

            batch_ids = ids[batch:batch + self.batch_size]

            files = [self.train_dataset[i] for i in batch_ids]

            data = load_data(files)

            yield data

    def train(self, samples=2048):

        metrics = pytorchMetrics()
        wallclocktime = 0

        # Summary
        summary_writer = tf.summary.FileWriter(self.summary, max_queue=1)

        for epoch in range(self.epoch, self.epochs):

            self.epoch = epoch

            print("\tEpoch %d/%d" % (epoch + 1, self.epochs))

            batch_numbers = math.ceil(samples/self.batch_size)

            # for real_images in self.batch_generator(samples):
            #
            #     start = time.time()
            #
            #     # Train epoch
            #     self.gan.train_batch(real_images, self.batch)
            #
            #     # Save iteration time
            #     wallclocktime += time.time() - start
            #
            #     print("\t\tBatch %d/%d - time: %.2f seconds" % ((self.batch % batch_numbers) + 1, batch_numbers, time.time() - start))
            #     self.batch += 1

            # Save epoch summary
            summary = tf.Summary()
            summary.value.add(tag="wallclocktime", simple_value=wallclocktime)
            summary_writer.add_summary(summary, global_step=self.epoch)

            self.plot_gif(epoch)

            ## Run metrics and save model
            # Select true images
            test_samples = 64
            idx = np.random.randint(0, len(self.test_dataset), test_samples)
            files = [self.test_dataset[i] for i in idx]
            true = load_data(files, repeat=True)

            # Select false images
            noise = np.random.normal(0, 1, (test_samples, self.latent_dim))
            false = self.gan.generator.predict(noise)
            false = np.sign(false)
            false = (0.5 * false + 0.5) * 255
            false = np.repeat(false, 3, 3)

            score = metrics.compute_score(true, false)
            self.metrics.append(score)

            # Save evaluation summary
            summary = tf.Summary()
            summary.value.add(tag="emd", simple_value=score.emd)
            summary.value.add(tag="fid", simple_value=score.fid)
            summary.value.add(tag="inception", simple_value=score.inception)
            summary.value.add(tag="knn", simple_value=score.knn)
            summary.value.add(tag="mmd", simple_value=score.mmd)
            summary.value.add(tag="mode", simple_value=score.mode)
            summary_writer.add_summary(summary, global_step=self.epoch)

            # Save model
            self.gan.save(self.model_dir, self.epoch)
            self.save_checkpoint()

        return score

