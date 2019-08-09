import numpy as np
import tensorflow as tf
from datetime import datetime

tf.logging.set_verbosity(tf.logging.ERROR)

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Choose which device you want to test on: either 'cpu' or 'gpu'
devices = ['cpu', 'gpu']

# Choose size of the matrix to be used.
# Make it bigger to see bigger benefits of parallel computation
shapes = [(50, 50), (100, 100), (500, 500), (1000, 1000), (10000, 10000)]


def compute_operations(device, shape):
    """Run a simple set of operations on a matrix of given shape on given device

    Parameters
    ----------
    device : the type of device to use, either 'cpu' or 'gpu'
    shape : a tuple for the shape of a 2d tensor, e.g. (10, 10)

    Returns
    -------
    out : results of the operations as the time taken
    """

    # Define operations to be computed on selected device
    with tf.device(device):
        random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
        dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
        sum_operation = tf.reduce_sum(dot_operation)

    # Time the actual runtime of the operations
    start_time = datetime.now()
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as session:
            result = session.run(sum_operation)
    elapsed_time = datetime.now() - start_time

    return result, elapsed_time



if __name__ == '__main__':

    # Run the computations and print summary of each run
    for device in devices:
        print("--" * 20)

        for shape in shapes:
            _, time_taken = compute_operations(device, shape)

            # Print the result and also the time taken on the selected device
            print("Input shape:", shape, "using Device:", device, "took: {:.2f}".format(time_taken.seconds + time_taken.microseconds/1e6))
            #print("Computation on shape:", shape, "using Device:", device, "took:")

    print("--" * 20)