from __future__ import print_function, division

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imageio
from tqdm import tqdm
import tensorflow.contrib.gan as tfgan
import time
import gc
import glob
import re
from help import *
import math

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class GAN():
    def __init__(self, name, latent_dim=100, channels=1, img_shape=(128, 128, 1)):
        self.name = name
        self.directory = '../resources/' + name + '/'
        self.gif_dir = self.directory + 'gif/'
        self.model_dir = self.directory + 'model/'
        self.summary = '../resources/summary/' + name
        self.latent_dim = latent_dim
        self.channels = channels
        self.img_shape = img_shape
        self.gif_matrix = 5
        self.gif_generator = np.random.normal(0, 1, (self.gif_matrix ** 2, self.latent_dim))
        self.epoch = 1
        self.batch = 0

        # Create directories
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            os.makedirs(self.gif_dir)
            os.makedirs(self.model_dir)

        # Load data
        self.data = self.load_data()

        # Get all the models saved
        files = glob.glob(self.model_dir + '*.h5')

        # Load the seed for plotting
        if not os.path.isfile(self.directory + '../' + 'seed.pkl'):
            save(self.gif_generator, self.directory + '../' + 'seed.pkl')
        else:
            self.gif_generator = load(self.directory + '../' + 'seed.pkl')

        if files:
            versions = []
            regex = re.compile(r'\d+')

            for filename in files:
                versions.append([int(x) for x in regex.findall(filename)][0])
            version = max(versions)

            self.generator = load_model(self.model_dir + 'generator_' + str(version) + '.h5')
            self.discriminator = load_model((self.model_dir + 'discriminator_' + str(version) + '.h5'))

            self.load_checkpoint()

        else:
            # Build generator
            self.generator = self.generator()

            # Build discriminator
            self.discriminator = self.discriminator()

        print('\n\nGenerator')
        self.generator.summary()

        print('\n\nDiscriminator')
        self.discriminator.summary()

        # Create GAN
        self.gan = self.create_gan()
        print('\n\nGAN')
        self.gan.summary()

    def save_checkpoint(self):

        data = [self.gif_generator, self.epoch] # TODO add loss, optimizer and other stuff

        save(data, self.directory + 'checkpoint.pkl')

    def load_checkpoint(self):

        self.gif_generator, self.epoch = load(self.directory + 'checkpoint.pkl')

    def load_data(self):
        # Load the dataset
        files = os.listdir('../dataset/test/')
        number_files = len(files)
        print('Number of files: ', number_files)

        X_train = []
        for file in files:
            img = cv2.imread('../dataset/test/' + file, 0)
            X_train.append(img)

        X_train = np.asarray(X_train, dtype='uint8')

        # Rescale
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        return X_train

    def generator(self):

        model = Sequential()

        model.add(Dense(128 * 32 * 32, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((32, 32, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        return model

    def discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

        return model

    def create_gan(self):

        self.discriminator.trainable = False

        # Input and Output of GAN
        gan_input = Input(shape=(100,))
        gan_output = self.discriminator(self.generator(gan_input))

        gan = Model(inputs=gan_input, outputs=gan_output)

        gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

        return gan

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


    def train(self, epochs=100, batch_size=128, save_every=5):

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # Frechet distance
        ground_truth = tf.placeholder(tf.uint8, shape=[None, 128, 128, 3])
        generated = tf.placeholder(tf.uint8, shape=[None, 128, 128, 3])

        true_pre = tfgan.eval.preprocess_image(ground_truth)
        generated_pre = tfgan.eval.preprocess_image(generated)

        true_act = tfgan.eval.run_inception(true_pre)
        generated_act = tfgan.eval.run_inception(generated_pre)

        frechet = tfgan.eval.mean_only_frechet_classifier_distance_from_activations(true_act, generated_act)

        # Summary
        summary_writer = tf.summary.FileWriter(self.summary, max_queue=1)

        for epoch in range(self.epoch, epochs + 1):

            self.epoch = epoch

            print("\nEpoch %d" % epoch)

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
                d_loss_real = self.discriminator.train_on_batch(real_images[0:l], valid[0:l])
                d_loss_fake = self.discriminator.train_on_batch(generated_images[0:l], fake[0:l])
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Train Generator
                # self.discriminator.trainable = False
                # noise = np.random.normal(0, 1, [batch_size, 100])
                g_loss = self.gan.train_on_batch(noise, valid)

                # Save batch summary
                summary = tf.Summary()
                summary.value.add(tag="d_loss", simple_value=d_loss)
                summary.value.add(tag="g_loss", simple_value=g_loss)
                summary_writer.add_summary(summary, global_step=self.batch)
                self.batch += 1

                print("\tBatch %d/%d - time: %.2f seconds" % (self.batch % batch_numbers, batch_numbers, time.time() - start))

            self.plot_gif(epoch)

            if epoch == 1 or epoch % save_every == 0:
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
                with tf.Session() as sess:
                    start = time.time()
                    actual_fid = sess.run(frechet, feed_dict={ground_truth: true, generated: false})
                    print("\tFid: %.2f,\t time: %.2f seconds" % (actual_fid, time.time() - start))
                    summary = tf.Summary()
                    summary.value.add(tag="fid", simple_value=actual_fid)
                    summary_writer.add_summary(summary, global_step=epoch)

                gc.collect()

                # Save model
                self.generator.save(self.model_dir + 'generator_' + str(epoch) + '.h5')
                self.discriminator.save(self.model_dir + 'discriminator_' + str(epoch) + '.h5')
                self.save_checkpoint()

ibis = GAN('train3')
ibis.train(epochs=100, batch_size=64)
#ibis.train2(epochs=4000, batch_size=32, save_interval=50)
#ibis.create_gif()

#ibis.save_checkpoint()