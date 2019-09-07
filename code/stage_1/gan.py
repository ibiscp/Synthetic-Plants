# Import GANs
from dcgan import DCGAN
from wgangp import WGANGP

# Other libraries
import os
import numpy as np
import tensorflow as tf
import time
from glob import glob
import re
from help import *
import math
import sys
sys.path.append('../')
from pytorchMetrics import *
from utils import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class GAN():
    def __init__(self, architecture, train_dataset, test_dataset, shape, **kwargs):

        self.architecture = architecture
        self.latent_dim = kwargs['latent_dim']
        self.batch_size = kwargs['batch_size']
        self.epochs = kwargs['epochs']
        self.kwargs = kwargs

        # Name
        self.name = architecture + ''.join('-{}-{}'.format(key, val) for key, val in sorted(kwargs.items())) + '/'

        # Directories
        self.resources = 'resources/'
        self.gif_dir = os.path.join(self.resources, 'gif/', self.name)
        self.model_dir = os.path.join(self.resources, 'model/', self.name)
        self.logs = os.path.join(self.resources, 'logs/', self.name)
        self.samples = os.path.join(self.resources, 'samples/', self.name)

        # Create directories
        check_folder(self.resources)
        check_folder(self.gif_dir)
        check_folder(self.model_dir)
        check_folder(self.logs)
        check_folder(self.samples)

        # Training parameters
        self.epoch = 0
        self.batch = 0
        self.metrics = []

        # Load data
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.kwargs['img_shape'] = shape

        # Load the seed for the gif
        if not os.path.isfile(self.resources + str(self.latent_dim) + '.pkl'):
            self.gif_generator = np.random.normal(0, 1, (GIF_MATRIX ** 2, self.latent_dim))
            save(self.gif_generator, self.resources + str(self.latent_dim) + '.pkl')
        else:
            self.gif_generator = load(self.resources + str(self.latent_dim) + '.pkl')

        # Load the correct gan
        self.load_gan()

    def load_gan(self):

        # Load gan
        self.gan = globals()[self.architecture](**self.kwargs)

        # Get all the models saved
        files = glob(self.model_dir + '*.h5')

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

        save(data, self.model_dir + 'checkpoint.pkl')

    def load_checkpoint(self):

        data = load(self.model_dir + 'checkpoint.pkl')

        for key, value in data.items():
            setattr(self, key, value)

    def predict_generator(self, noise):

        # samples_batch = self.batch_size
        samples = []

        for i in range(math.ceil(len(noise)/self.batch_size)):
            n = noise[i*self.batch_size:min((i+1)*self.batch_size, len(noise))]
            output = self.gan.generator.predict(n)
            samples.append(output)

        samples = np.vstack(samples)

        return samples

    def batch_generator(self, samples):

        ids = np.arange(len(self.train_dataset))
        # l = len(ids)
        # miss = np.random.choice(ids, self.batch_size - l % self.batch_size)
        # ids = np.hstack((ids, miss))

        np.random.shuffle(ids)

        for batch in range(0, samples, self.batch_size):

            batch_ids = ids[batch:batch + self.batch_size]

            files = [self.train_dataset[i] for i in batch_ids]

            data = load_data(files, type='mask', scale=True)

            yield data

    def train(self, samples=1):#2048):

        metrics = pytorchMetrics()
        wallclocktime = 0

        # Summary
        summary_writer = tf.summary.FileWriter(self.logs, max_queue=1)

        for epoch in range(self.epoch, self.epochs):

            self.epoch = epoch

            print("\tEpoch %d/%d" % (epoch + 1, self.epochs))

            batch_numbers = math.ceil(samples/self.batch_size)

            for real_images in self.batch_generator(samples):

                start = time.time()

                # Train epoch
                self.gan.train_batch(real_images, self.batch)

                # Save iteration time
                wallclocktime += time.time() - start

                print("\t\tBatch %d/%d - time: %.2f seconds" % ((self.batch % batch_numbers) + 1, batch_numbers, time.time() - start))
                self.batch += 1

            self.gan.reduce_lr(epoch)

            # Save epoch summary
            summary = tf.Summary()
            summary.value.add(tag="wallclocktime", simple_value=wallclocktime)
            summary_writer.add_summary(summary, global_step=self.epoch)

            # Plot gif
            gen_imgs = self.predict_generator(self.gif_generator)
            gen_imgs = np.sign(gen_imgs)
            gen_imgs = postprocessing(gen_imgs)
            plot_gif(gen_imgs, epoch, self.gif_dir, type='mask')

            ## Run metrics and save model
            # Select true images
            test_samples = 128
            idx = np.random.randint(0, len(self.test_dataset), test_samples)
            files = [self.test_dataset[i] for i in idx]
            true = load_data(files, type='mask', repeat=True)
            # print(np.max(true))
            # print(np.min(true))

            # Select false images
            noise = np.random.normal(0, 1, (test_samples, self.latent_dim))
            false = self.predict_generator(noise)
            false = np.sign(false)
            false = postprocessing(false)
            false = np.repeat(false, 3, 3)
            # print(np.max(false))
            # print(np.min(false))

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
            if (epoch + 1) % 10 == 0:
                self.gan.save(self.model_dir, self.epoch)
                self.save_checkpoint()

            # Print samples
            gen_imgs = np.rint(gen_imgs).astype(int)
            gen_imgs = np.squeeze(gen_imgs, axis=3)

            # Save images separately
            for i, img in enumerate(gen_imgs):
                imsave(img, os.path.join(self.samples, 'mask_%d_%d.png' %(i, epoch)))

        # Save last model
        self.gan.save(self.model_dir, self.epoch)
        self.save_checkpoint()

        # Create gif
        create_gif(self.gif_dir, self.metrics, self.test_dataset, type='mask')

        return score

