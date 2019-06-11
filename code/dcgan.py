from __future__ import print_function, division

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import os
import cv2
import tensorflow.contrib.gan as tfgan

os.environ['KMP_DUPLICATE_LIB_OK']='True'


# Just disables the warning
import warnings
import os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt


import sys

import numpy as np
import time

import gc

class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        self.frechet = self.calculate_fid()

    def build_generator(self):

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

        print('\n\nGenerator')
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

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

        print('\n\nDiscriminator')
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def calculate_fid(self):

        # Load the dataset
        files = os.listdir('../dataset/test/')
        number_files = len(files)
        print('Number of files: ', number_files)

        X_train = []
        for file in files:
            img = cv2.imread('../dataset/test/' + file, 0)
            X_train.append(img)

        X_train = np.asarray(X_train, dtype='uint8')

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Frechet distance
        ground_truth = tf.placeholder(tf.uint8, shape=[None, 128, 128, 3])
        generated = tf.placeholder(tf.uint8, shape=[None, 128, 128, 3])

        true_pre = tfgan.eval.preprocess_image(ground_truth)
        generated_pre = tfgan.eval.preprocess_image(generated)

        true_act = tfgan.eval.run_inception(true_pre)
        generated_act = tfgan.eval.run_inception(generated_pre)

        frechet = tfgan.eval.mean_only_frechet_classifier_distance_from_activations(true_act, generated_act)

        with tf.Session() as sess:
            for i in range(10):
                # Select true images
                idx = np.random.randint(0, X_train.shape[0], 100)
                true = X_train[idx]
                true = (0.5 * true + 0.5) * 255
                true = np.repeat(true, 3, 3)

                idx = np.random.randint(0, X_train.shape[0], 100)
                false = X_train[idx]
                false = (0.5 * false + 0.5) * 255
                false = np.repeat(false, 3, 3)

                start = time.time()
                actual_fid = sess.run(frechet, feed_dict={ground_truth: true, generated: false})
                print("\nfid: %.2f,\t time: %.2f seconds" % (actual_fid, time.time() - start))

        gc.collect()

        return 0

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        files = os.listdir('../dataset/test/')
        number_files = len(files)
        print('Number of files: ', number_files)

        X_train = []
        for file in files:
            img = cv2.imread('../dataset/test/' + file, 0)
            X_train.append(img)

        X_train = np.asarray(X_train, dtype='uint8')

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

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

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if (epoch) % save_interval == 0:

                # Select true images
                idx = np.random.randint(0, X_train.shape[0], 100)
                true = X_train[idx]
                true = (0.5 * true + 0.5) * 255
                true = np.repeat(true, 3, 3)

                # Select false images
                noise = np.random.normal(0, 1, (100, self.latent_dim))
                false = self.generator.predict(noise)
                false = np.sign(false)
                false = (0.5 * false + 0.5) * 255
                false = np.repeat(false, 3, 3)

                with tf.Session() as sess:
                    start = time.time()
                    actual_fid = sess.run(frechet, feed_dict={ground_truth:true, generated:false})
                    print("\nfid: %.2f,\t time: %.2f seconds" % (actual_fid, time.time() - start))

                gc.collect()


                self.save_imgs(epoch+1, false)

                # self.calculate_fid(true, false)

    def save_imgs(self, epoch, gen_imgs):
        r, c = 5, 5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("../dataset/generated/mask_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=4000, batch_size=32, save_interval=50)

    #dcgan.calculate_fid()