import pickle
import glob
import imageio

# Save dictionary to file
def save(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)

# Load dictionary from file
def load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def create_gif(self, directory, size=100):
    files = glob.glob(directory + 'gif/' + '*.png')
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    images = []

    number = int(len(files)/size)

    for i in range(size):
        images.append(imageio.imread(files[int(i*number)]))

    for i in range(int(size*.2)):
        images.append(imageio.imread(files[-1]))

    imageio.mimsave(directory + 'training.gif', images)