from argparse import ArgumentParser
from gridSearch import *
from help import *

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("dataset_path", nargs='?', default='../dataset/SugarBeets/train/mask/', help="Name of the dataset path to use")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # Load sentences
    dataset = load_data(path=args.dataset_path)

    # Define the base grid search parameters
    base = {'epochs': [100], 'latent_dim': [100], 'batch_size': [128]}

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