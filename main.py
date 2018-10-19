import matplotlib.pyplot as plt
from matplotlib.image import imsave, imread
import numpy as np

from image import Image

pic = imread("img/face.png")

image = Image(pic)

image.rgb2gray()

image.to_binary()


plt.imshow(image.get_matrix(), cmap='Greys_r')
plt.show()

image.bin_dilation()

plt.imshow(image.get_matrix(), cmap='Greys_r')
plt.show()

