# Import GANs
from dcgan import DCGAN
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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class GAN():
    def __init__(self, name, architecture, latent_dim=100, img_shape=(128, 128, 1), g_lr=0.0002, g_beta_1=0.5,
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

        print('\n\nGenerator')
        self.generator.summary()

        print('\n\nDiscriminator')
        self.discriminator.summary()

        print('\n\nGAN')
        self.combined.summary()

    def load_gan(self):

        if self.architecture == 'dcgan':
            gan = DCGAN(latent_dim=self.latent_dim, img_shape=self.img_shape, g_lr=self.g_lr, g_beta_1=self.g_beta_1,
                        d_lr=self.d_lr, d_beta_1=self.d_beta_1)
            self.generator = gan.generator
            self.discriminator = gan.discriminator
            self.combined = gan.gan


        # Get all the models saved
        files = glob.glob(self.model_dir + '*.h5')

        # Load Checkpoint if found
        if files:
            versions = []
            regex = re.compile(r'\d+')

            for filename in files:
                versions.append([int(x) for x in regex.findall(filename)][0])
            version = max(versions)

            self.generator = load_model(self.model_dir + 'generator_' + str(version) + '.h5')
            self.discriminator = load_model((self.model_dir + 'discriminator_' + str(version) + '.h5'))

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
        self.epoch = data['epoch']
        self.batch = data['batch']
        self.wallclocktime = data['wallclocktime']

    def plot_gif(self, epoch):

        gen_imgs = self.generator.predict(self.gif_generator)
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

    def batch_generator(self, batch_size):

        ids = np.arange(self.data.shape[0])
        np.random.shuffle(ids)
        l = len(ids)

        for batch in range(0, l, batch_size):
            batch_ids = ids[batch:min(batch + batch_size, l)]
            yield self.data[batch_ids]


    def train(self, epochs=100, batch_size=128, save_every=1):

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # Frechet distance
        # ground_truth = tf.placeholder(tf.uint8, shape=[None, 128, 128, 3])
        # generated = tf.placeholder(tf.uint8, shape=[None, 128, 128, 3])
        #
        # true_pre = tfgan.eval.preprocess_image(ground_truth)
        # generated_pre = tfgan.eval.preprocess_image(generated)
        #
        # true_act = tfgan.eval.run_inception(true_pre)
        # generated_act = tfgan.eval.run_inception(generated_pre)
        #
        # frechet = tfgan.eval.mean_only_frechet_classifier_distance_from_activations(true_act, generated_act)

        metrics = pytorchMetrics()

        # Summary
        summary_writer = tf.summary.FileWriter(self.summary, max_queue=1)

        for epoch in range(self.epoch, epochs):

            self.epoch = epoch

            print("\n\tEpoch %d" % epoch)

            batch_numbers = math.ceil(self.data.shape[0]/batch_size)

            for real_images in self.batch_generator(batch_size):

                start = time.time()

                # Generate fake images
                noise = np.random.normal(0, 1, [batch_size, 100])
                generated_images = self.generator.predict(noise)

                # Get real images size
                l = real_images.shape[0]

                # Train the Discriminator
                # self.discriminator.trainable = True
                d_loss_real, d_acc_real = self.discriminator.train_on_batch(real_images[0:l], valid[0:l])
                d_loss_fake, d_acc_fake = self.discriminator.train_on_batch(generated_images[0:l], fake[0:l])
                d_loss = 0.5 * (d_loss_real + d_loss_fake)
                d_acc = 0.5 * (d_acc_real + d_acc_fake)

                # TODO train generator before discriminator and get accuracy
                # Train Generator
                # self.discriminator.trainable = False
                # noise = np.random.normal(0, 1, [batch_size, 100])
                g_loss = self.combined.train_on_batch(noise, valid)

                # Save iteration time
                self.wallclocktime += time.time() - start

                # Save batch summary
                summary = tf.Summary()
                summary.value.add(tag="d_loss", simple_value=d_loss)
                summary.value.add(tag="g_loss", simple_value=g_loss)
                summary.value.add(tag="d_acc", simple_value=d_acc)
                summary.value.add(tag="g_acc", simple_value=1-d_acc)
                summary_writer.add_summary(summary, global_step=self.batch)

                print("\t\tBatch %d/%d - time: %.2f seconds" % ((self.batch % batch_numbers) + 1, batch_numbers, time.time() - start))
                self.batch += 1

            # Save epoch summary
            summary = tf.Summary()
            summary.value.add(tag="wallclocktime", simple_value=self.wallclocktime)
            summary_writer.add_summary(summary, global_step=self.epoch)

            self.plot_gif(epoch)

            # Run metrics and save model
            # if epoch == 0 or epoch+1 % save_every == 0:
            # Select true images
            idx = np.random.randint(0, self.data.shape[0], 100)
            true = self.data[idx]
            true = (0.5 * true + 0.5) * 255
            true = np.repeat(true, 3, 3)

            # Select false images
            noise = np.random.normal(0, 1, (100, self.latent_dim))
            false = self.generator.predict(noise)
            false = np.sign(false)
            false = (0.5 * false + 0.5) * 255
            false = np.repeat(false, 3, 3)

            # Run evaluation
            # with tf.Session() as sess:
            #     start = time.time()
            #     fid = sess.run(frechet, feed_dict={ground_truth: true, generated: false})
            #     print("\t\tFid: %.2f - time: %.2f seconds" % (fid, time.time() - start))
            #
            #     # Save evaluation summary
            #     summary = tf.Summary()
            #     summary.value.add(tag="fid", simple_value=fid)
            #     summary_writer.add_summary(summary, global_step=self.epoch)
            #
            # gc.collect()

            score = metrics.compute_score(true, false)

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
            self.generator.save(self.model_dir + 'generator_' + str(epoch) + '.h5')
            self.discriminator.save(self.model_dir + 'discriminator_' + str(epoch) + '.h5')
            self.save_checkpoint()

ibis = GAN('ganseparated', 'dcgan')
ibis.train(epochs=100, batch_size=64)

