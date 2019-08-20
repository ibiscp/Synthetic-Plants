import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from help import *

# Remove warnings
tf.logging.set_verbosity(tf.logging.ERROR)


class DCGAN():
    def __init__(self, **kwargs):
        # Input parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Input shape
        self.img_rows = self.img_shape[0]
        self.img_cols = self.img_shape[1]
        self.channels = self.img_shape[2]

        assert self.img_rows % 4 == 0, "output image size must be divisible by 4 and square"
        assert self.img_cols % 4 == 0, "output image size must be divisible by 4 and square"

        with tf.device('gpu'):
            self.generator = self.generator()
            self.discriminator = self.discriminator()
            self.combined = self.combined()

    def generator(self):

        input_size = int(self.img_rows / 4)

        model = Sequential()

        model.add(Dense(128 * input_size * input_size, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((input_size, input_size, 128)))
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

        model.compile(loss='binary_crossentropy', optimizer=Adam(self.d_lr, self.d_beta_1), metrics=['accuracy'])

        return model

    def combined(self):

        self.discriminator.trainable = False

        # Input and Output of GAN
        gan_input = Input(shape=(self.latent_dim,))
        gan_output = self.discriminator(self.generator(gan_input))

        gan = Model(inputs=gan_input, outputs=gan_output)

        gan.compile(loss='binary_crossentropy', optimizer=Adam(self.g_lr, self.g_beta_1))

        self.discriminator.trainable = True

        return gan

    def train_batch(self, real_images, batch):

        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        # Get real images size
        l = real_images.shape[0]

        # Generate fake images
        noise = np.random.normal(0, 1, [l, self.latent_dim])
        generated_images = self.generator.predict(noise)

        # TRAIN DISCRIMINATOR
        d_loss_real, d_acc_real = self.discriminator.train_on_batch(real_images, valid[:l])
        d_loss_fake, d_acc_fake = self.discriminator.train_on_batch(generated_images, fake[:l])
        self.d_loss = 0.5 * (d_loss_real + d_loss_fake)
        d_acc = 0.5 * (d_acc_real + d_acc_fake)

        # TRAIN GENERATOR
        self.g_loss = self.combined.train_on_batch(noise, valid)

    def load(self, dir, version):
        self.generator.load_weights(dir + 'generator_' + str(version) + '.h5')
        self.discriminator.load_weights(dir + 'discriminator_' + str(version) + '.h5')
        self.combined.load_weights(dir + 'combined_' + str(version) + '.h5')

    def save(self, dir, version):
        self.generator.save_weights(dir + 'generator_' + str(version) + '.h5')
        self.discriminator.save_weights(dir + 'discriminator_' + str(version) + '.h5')
        self.combined.save_weights(dir + 'combined_' + str(version) + '.h5')