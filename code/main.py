from argparse import ArgumentParser
from gridSearch import *
from help import *

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("dataset_path", nargs='?', default='../dataset/test/', help="Name of the dataset path to use")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # Load sentences
    dataset = load_data(path=args.dataset_path)

    # Define the base grid search parameters
    base = {'epochs': [2], 'latent_dim': [100, 200], 'batch_size': [64]}

    # DCGAN
    DCGAN = {'g_lr': [0.0002], 'g_beta_1': [0.5], 'd_lr': [0.0002], 'd_beta_1': [0.5]}
    DCGAN.update(base)

    # WGANGP
    WGANGP = {'g_lr': [0.0002], 'c_lr': [0.0002], 'n_critic': [5]}
    WGANGP.update(base)

    # Train
    grid = gridSearch(dataset=dataset, parameters=WGANGP)
    grid.fit()

    # Print grid search summary
    grid.summary()