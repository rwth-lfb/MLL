from six.moves import urllib
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os

class MNIST:
    def __init__(self):
        # Alternative method to load MNIST, if mldata.org is down
        from scipy.io import loadmat
        import os
        dset_path = os.path.join(os.path.abspath(os.environ["HOME"]), 'datasets')
        mnist_path = os.path.join(dset_path, "mnist-original.mat")
        try:
            mnist_raw = loadmat(mnist_path)
        except FileNotFoundError:
            if not os.path.isdir(dset_path):
                os.makedirs(dset_path)
            mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
            response = urllib.request.urlopen(mnist_alternative_url)
            with open(mnist_path, "wb") as f:
                content = response.read()
                f.write(content)
            mnist_raw = loadmat(mnist_path)

        self.data = mnist_raw["data"].T

        onehot_encoder = OneHotEncoder(n_values=10, sparse=False)
        labels = mnist_raw["label"][0].reshape(len(mnist_raw["label"][0]), 1)
        labels = onehot_encoder.fit_transform(labels)
        
        self.target = labels