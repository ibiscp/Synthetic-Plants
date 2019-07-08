# Import GANs
from dcgan import DCGAN
from wgangp import WGANGP
from tensorflow.keras.models import load_model

# Other libraries
import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use("TKAgg")
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

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class GAN():
    def __init__(self, name, architecture, latent_dim=100, batch_size=64, img_shape=(128, 128, 1), g_lr=0.0002, g_beta_1=0.5,
                 d_lr=0.0002, d_beta_1=0.5):
        # Names
        self.name = name # TODO change name based on parameters
        self.architecture = architecture

        # Parameters
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.g_beta_1 = g_beta_1
        self.d_beta_1 = d_beta_1
        self.batch_size = batch_size

        # Directories
        self.directory = '../resources/' + name + '/'
        self.gif_dir = self.directory + 'gif/'
        self.model_dir = self.directory + 'model/'
        self.summary = '../resources/summary/' + name

        # Gif
        self.gif_matrix = 5
        self.gif_generator = np.random.normal(0, 1, (self.gif_matrix ** 2, self.latent_dim))

        # Training parameters
        self.epoch = 0
        self.batch = 0
        self.wallclocktime = 0
        self.metrics = []

        # Create directories
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            os.makedirs(self.gif_dir)
            os.makedirs(self.model_dir)

        # Load data
        self.data = load_data()

        # Load the seed for the gif
        if not os.path.isfile(self.directory + '../' + 'seed.pkl'):
            save(self.gif_generator, self.directory + '../' + 'seed.pkl')
        else:
            self.gif_generator = load(self.directory + '../' + 'seed.pkl')

        # Load the correct gan
        self.load_gan()

        # print('\n\nGenerator')
        # self.generator.summary()
        #
        # print('\n\nDiscriminator')
        # self.discriminator.summary()
        #
        # print('\n\nGAN')
        # self.combined.summary()

    def load_gan(self):

        if self.architecture == 'dcgan':
            self.gan = DCGAN(latent_dim=self.latent_dim, batch_size=self.batch_size, img_shape=self.img_shape, g_lr=self.g_lr, g_beta_1=self.g_beta_1,
                        d_lr=self.d_lr, d_beta_1=self.d_beta_1)
            # self.generator = self.gan.generator
            # self.discriminator = self.gan.discriminator
            # self.combined = gan.gan
        elif self.architecture == 'wgangp':
            self.gan = WGANGP(latent_dim=self.latent_dim, img_shape=self.img_shape, g_lr=self.g_lr, g_beta_1=self.g_beta_1,
                        d_lr=self.d_lr, d_beta_1=self.d_beta_1)
            # self.generator = gan.generator
            # self.discriminator = gan.discriminator
            # self.critic_model = gan.critic_model
            # self.generator_model = gan.generator_model


        # Get all the models saved
        files = glob.glob(self.model_dir + '*.h5')

        # Load Checkpoint if found
        if files:
            versions = []
            regex = re.compile(r'\d+')

            for filename in files:
                versions.append([int(x) for x in regex.findall(filename)][0])
            version = max(versions)

            # self.generator = load_model(self.model_dir + 'generator_' + str(version) + '.h5')
            # self.discriminator = load_model((self.model_dir + 'discriminator_' + str(version) + '.h5'))
            self.gan.load(self.model_dir, version)

            self.load_checkpoint()

    def save_checkpoint(self):

        data = {}
        data['architecture'] = self.architecture
        data['img_shape'] = self.img_shape
        data['latent_dim'] = self.latent_dim
        data['g_lr'] = self.g_lr
        data['d_lr'] = self.d_lr
        data['g_beta_1'] = self.g_beta_1
        data['d_beta_1'] = self.d_beta_1
        data['epoch'] = self.epoch
        data['batch'] = self.batch
        data['wallclocktime'] = self.wallclocktime
        data['metrics'] = self.metrics

        save(data, self.directory + 'checkpoint.pkl')

    def load_checkpoint(self):

        data = load(self.directory + 'checkpoint.pkl')

        self.architecture = data['architecture']
        self.img_shape = data['img_shape']
        self.latent_dim = data['latent_dim']
        self.g_lr = data['g_lr']
        self.d_lr = data['d_lr']
        self.g_beta_1 = data['g_beta_1']
        self.d_beta_1 = data['d_beta_1']
        self.epoch = data['epoch'] + 1
        self.batch = data['batch']
        self.wallclocktime = data['wallclocktime']
        self.metrics = data['metrics']

    def plot_gif(self, epoch):

        gen_imgs = self.gan.generator.predict(self.gif_generator)
        gen_imgs = np.sign(gen_imgs)
        gen_imgs = (0.5 * gen_imgs + 0.5) * 255

        plt.figure(figsize=(self.gif_matrix, self.gif_matrix))
        gs1 = gridspec.GridSpec(self.gif_matrix, self.gif_matrix)
        gs1.update(wspace=0.025, hspace=0.025)
        for i in range(gen_imgs.shape[0]):
            ax = plt.subplot(gs1[i])
            ax.imshow(gen_imgs[i,:,:,0], cmap='gray', interpolation='nearest')
            ax.axis('off')
            ax.set_aspect('equal')
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_frame_on(False)

        plt.savefig(self.gif_dir + "%d.png" % epoch, bbox_inches='tight', pad_inches = 0.025)
        plt.close()

    def batch_generator(self):

        ids = np.arange(self.data.shape[0])
        l = len(ids)
        miss = np.random.choice(ids, self.batch_size - l % self.batch_size)
        ids = np.hstack((ids, miss))

        np.random.shuffle(ids)

        for batch in range(0, l, self.batch_size):

            batch_ids = ids[batch:batch + self.batch_size]

            yield self.data[batch_ids]

    def train(self, epochs=100):

        metrics = pytorchMetrics()

        # Summary
        summary_writer = tf.summary.FileWriter(self.summary, max_queue=1)

        for epoch in range(self.epoch, epochs):

            self.epoch = epoch

            print("\n\tEpoch %d" % epoch)

            batch_numbers = math.ceil(self.data.shape[0]/self.batch_size)

            for real_images in self.batch_generator():

                start = time.time()

                # Train epoch
                self.gan.train_batch(real_images)

                # Save iteration time
                self.wallclocktime += time.time() - start

                print("\t\tBatch %d/%d - time: %.2f seconds" % ((self.batch % batch_numbers) + 1, batch_numbers, time.time() - start))
                self.batch += 1

            # Save epoch summary
            summary = tf.Summary()
            summary.value.add(tag="wallclocktime", simple_value=self.wallclocktime)
            summary_writer.add_summary(summary, global_step=self.epoch)

            self.plot_gif(epoch)

            ## Run metrics and save model
            # Select true images
            idx = np.random.randint(0, self.data.shape[0], self.latent_dim)
            true = self.data[idx]
            true = (0.5 * true + 0.5) * 255
            true = np.repeat(true, 3, 3)

            # Select false images
            noise = np.random.normal(0, 1, (100, self.latent_dim))
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

ibis = GAN('dcganload', 'dcgan', batch_size=64)
ibis.train(epochs=100)

