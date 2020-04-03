import numpy as np
import matplotlib.pyplot as plt

def plot_batch(images, n=36):
    """ Visualize a single batch of images
    
    Parameters
    ----------
    images : numpy.ndarray
        images of shape (b, c, x, y)
    n : int, default: 36
        number of images to display. Must be a power of 2
    """
    rowcols = np.sqrt(n)
    plt.figure(figsize=(rowcols, rowcols))
    for index in range(n):
        plt.subplot(rowcols, rowcols, index + 1)
        plt.imshow(images[index, 0, :, :], cmap="binary")
        plt.axis("off")
    plt.show()