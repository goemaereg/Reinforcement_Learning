import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib import pyplot as plt

def img2np(filepath, show=False):
    # Open the image form working directory
    image = Image.open(filepath).convert('L')

    if show:
        # show the image
        image_mpl = mpl.image.imread(filepath)
        plt.imshow(image_mpl)
        plt.show()

    importnp_image = np.asarray(image)
    np_image = importnp_image/255
    np_image = np_image.astype(int) # array of 0s and 1s
    np_image = 1 - np_image
    return np_image

if __name__ == '__main__':
    img2np('gridworlds_img/parking_gridworld.png')
