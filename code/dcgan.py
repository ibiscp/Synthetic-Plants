import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

# Remove warnings
tf.logging.set_verbosity(tf.logging.ERROR)


class DCGAN():
    def __init__(self, latent_dim=100, img_shape=(128, 128, 1), g_lr=0.0002, g_beta_1=0.5, d_lr=0.0002, d_beta_1=0.5):
        # Input shape
        self.img_rows = img_shape[0]
        self.img_cols = img_shape[1]
        self.channels = img_shape[2]
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.g_beta_1 = g_beta_1
        self.d_beta_1 = d_beta_1

        assert self.img_rows % 4 == 0, "output image size must be divisible by 4 and square"
        assert self.img_cols % 4 == 0, "output image size must be divisible by 4 and square"

        self.generator = self.generator()
        self.discriminator = self.discriminator()
        self.gan = self.gan()

    def generator(self):

        input_size = int(self.img_rows /4)

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

        model.compile(loss='binary_crossentropy', optimizer=Adam(self.d_lr, self.d_beta_1))

        return model

    def gan(self):

        self.discriminator.trainable = False

        # Input and Output of GAN
        gan_input = Input(shape=(self.latent_dim,))
        gan_output = self.discriminator(self.generator(gan_input))

        gan = Model(inputs=gan_input, outputs=gan_output)

        gan.compile(loss='binary_crossentropy', optimizer=Adam(self.g_lr, self.g_beta_1))

        return gan
