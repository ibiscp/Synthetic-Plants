from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop
from functools import partial
import tensorflow.keras.backend as K
import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class RandomWeightedAverage(Add):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((64, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class WGANGP():
    def __init__(self, batch_size=64, latent_dim=100, img_shape=(128, 128, 1), n_critic=5, g_lr=0.00005, c_lr=0.00005):
        # Input shape
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.img_rows = img_shape[0]
        self.img_cols = img_shape[1]
        self.channels = img_shape[2]

        self.n_critic = n_critic
        self.g_lr = g_lr
        self.c_lr = c_lr

        assert self.img_rows % 4 == 0, "output image size must be divisible by 4 and square"
        assert self.img_cols % 4 == 0, "output image size must be divisible by 4 and square"

        self.generator = self.generator()
        self.critic = self.critic()
        self.critic_model, self.generator_model = self.gan()

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def generator(self):

        input_size = int(self.img_rows / 4)

        model = Sequential()

        model.add(Dense(128 * input_size * input_size, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((input_size, input_size, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def critic(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def gan(self):

        # CONSTRUCT COMPUTATIONAL GRAPH FOR THE CRITIC

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))

        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])

        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss, averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        # Defines critic model
        critic_model = Model(inputs=[real_img, z_disc], outputs=[valid, fake, validity_interpolated])
        critic_model.compile(loss=[self.wasserstein_loss, self.wasserstein_loss, partial_gp_loss], optimizer=RMSprop(lr=self.c_lr), loss_weights=[1, 1, 10])

        # CONSTRUCT COMPUTATIONAL GRAPH FOR GENERATOR

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,))

        # Generate images based of noise
        img = self.generator(z_gen)

        # Discriminator determines validity
        valid = self.critic(img)

        # Defines generator model
        generator_model = Model(z_gen, valid)
        generator_model.compile(loss=self.wasserstein_loss, optimizer=RMSprop(lr=self.g_lr))

        return critic_model, generator_model

    def train_batch(self, real_images, batch):

        # Adversarial ground truths
        valid = -np.ones((self.batch_size, 1))
        fake = np.ones((self.batch_size, 1))
        dummy = np.zeros((self.batch_size, 1)) # Dummy gt for gradient penalty

        # Get real images size
        l = real_images.shape[0]

        # Sample generator input
        noise = np.random.normal(0, 1, (l, self.latent_dim))

        # TRAIN CRITIC
        self.d_loss = self.critic_model.train_on_batch([real_images, noise], [valid[:l], fake[:l], dummy[:l]])

        # TRAIN GENERATOR
        if batch % self.n_critic == 0:
            self.g_loss = self.generator_model.train_on_batch(noise, valid)

    def load(self, dir, version):
        self.generator.load_weights(dir + 'generator_' + str(version) + '.h5')
        self.critic.load_weights(dir + 'critic_' + str(version) + '.h5')
        self.critic_model.load_weights(dir + 'critic_model_' + str(version) + '.h5')
        self.generator_model.load_weights(dir + 'generator_model_' + str(version) + '.h5')

    def save(self, dir, version):
        self.generator.save_weights(dir + 'generator_' + str(version) + '.h5')
        self.critic.save_weights(dir + 'critic_' + str(version) + '.h5')
        self.critic_model.save_weights(dir + 'critic_model_' + str(version) + '.h5')
        self.generator_model.save_weights(dir + 'generator_model_' + str(version) + '.h5')

