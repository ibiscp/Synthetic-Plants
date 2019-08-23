import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from help import *
import math

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

        assert isPowerOfTwo(self.img_rows) == True, "output image size must be power of 2"
        assert isPowerOfTwo(self.img_cols) == True, "output image size must be power of 2"
        assert self.img_rows == self.img_cols, "output image size must be square"

        # Possible parameters
        self.startingSize = 8
        self.outputFilter = 8
        self.kernel_size = 3

        self.upSamplingLayer = int(math.log2(self.img_rows) - math.log2(self.startingSize))

        with tf.device('gpu'):
            self.generator = self.generator()
            self.discriminator = self.discriminator()
            self.combined = self.combined()

    def generator(self):

        model = Sequential()

        starting = self.outputFilter * (2 ** (self.upSamplingLayer + 1))
        model.add(Dense(starting * self.startingSize ** 2, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((self.startingSize, self.startingSize, starting)))
        model.add(BatchNormalization(momentum=0.8))

        # 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256 -> 512x512
        for i in range(self.upSamplingLayer):
            model.add(UpSampling2D())
            # print(self.outputFilter * (2 ** (self.upSamplingLayer - i)))
            model.add(Conv2D(self.outputFilter * (2 ** (self.upSamplingLayer - i)), kernel_size=self.kernel_size, padding="same"))
            model.add(Activation("relu"))
            model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(self.outputFilter, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        # print("\nGenerator")
        # model.summary()

        return model

    def discriminator(self):

        model = Sequential()

        model.add(Conv2D(self.outputFilter, kernel_size=self.kernel_size, strides=2, input_shape=self.img_shape, padding="same"))  # 256 -> 128
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        # 128 -> 64 -> 32 -> 16 -> 8
        for i in range(self.upSamplingLayer):
            # print(self.outputFilter * (2 ** i))
            model.add(Conv2D(self.outputFilter * (2 ** (i+1)), kernel_size=self.kernel_size, strides=2, padding="same"))  # 128 -> 64
            # model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.25))
            model.add(BatchNormalization(momentum=0.8))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer=Adam(self.d_lr, self.d_beta_1), metrics=['accuracy'])

        # print("\nDiscriminator")
        # model.summary()

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
        # print(real_images.shape)
        d_loss_real, d_acc_real = self.discriminator.train_on_batch(real_images, valid[:l])
        # print(real_images.shape)
        d_loss_fake, d_acc_fake = self.discriminator.train_on_batch(generated_images, fake[:l])
        self.d_loss = 0.5 * (d_loss_real + d_loss_fake)
        d_acc = 0.5 * (d_acc_real + d_acc_fake)

        # TRAIN GENERATOR
        self.g_loss = self.combined.train_on_batch(noise, valid)

    def load(self, dir, version):
        # self.generator.load_weights(dir + 'generator_' + str(version) + '.h5')
        # self.discriminator.load_weights(dir + 'discriminator_' + str(version) + '.h5')
        # self.combined.load_weights(dir + 'combined_' + str(version) + '.h5')

        self.generator.load_weights(dir + 'generator' + '.h5')
        self.discriminator.load_weights(dir + 'discriminator' + '.h5')
        self.combined.load_weights(dir + 'combined' + '.h5')

    def save(self, dir, version):
        # self.generator.save_weights(dir + 'generator_' + str(version) + '.h5')
        # self.discriminator.save_weights(dir + 'discriminator_' + str(version) + '.h5')
        # self.combined.save_weights(dir + 'combined_' + str(version) + '.h5')

        self.generator.save_weights(dir + 'generator' + '.h5')
        self.discriminator.save_weights(dir + 'discriminator' + '.h5')
        self.combined.save_weights(dir + 'combined' + '.h5')