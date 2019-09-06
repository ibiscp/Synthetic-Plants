from argparse import ArgumentParser
from gridSearch import *
from help import *

import sys
sys.path.append('../')
from utils import *


# '../dataset/SugarBeets/train/mask/'

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("dataset_path", nargs='?', default='../../plants_dataset/SugarBeets_256/', help="Name of the dataset path to use")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # Load dataset list
    train_dataset, shape = load_dataset_list(directory=args.dataset_path + 'train/mask/', type='mask')
    test_dataset, _ = load_dataset_list(directory=args.dataset_path + 'test/mask/', type='mask')

    # Define the base grid search parameters
    base = {'epochs': [150], 'latent_dim': [100], 'batch_size': [128]}

    # DCGAN
    DCGAN = {'g_lr': [0.0002], 'g_ld': [0.001], 'g_beta_1': [0.5], 'd_lr': [0.0002], 'd_ld': [0.001], 'd_beta_1': [0.5]}
    DCGAN.update(base)

    # WGANGP
    WGANGP = {'g_lr': [0.0002], 'c_lr': [0.0002], 'n_critic': [5]}
    WGANGP.update(base)

    # Train
    grid = gridSearch(train_dataset=train_dataset, test_dataset=test_dataset, shape=shape, parameters=DCGAN)
    grid.fit()

    # Print grid search summary
    grid.summary()