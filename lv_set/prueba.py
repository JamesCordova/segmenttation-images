import numpy as np
from skimage.io import imread
from scipy.ndimage import gaussian_filter

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True,precision=8)
#   img = imread('im2.bmp', True)
img = imread('test.bmp', True)
print(img.shape)
print(img[0][0])
img = imread('gourd.bmp', True)
img = np.interp(img, [np.min(img), np.max(img)], [0, 255])
print(img.shape)
c0 = 2
initial_lsf = c0 * np.ones(img.shape)
initial_lsf[24:35, 19:25] = -c0
initial_lsf[24:35, 39:50] = -c0

img_smooth = gaussian_filter(img, 0.8)  # smooth image by Gaussian convolution
[Iy, Ix] = np.gradient(img_smooth)
phi = initial_lsf.copy()
# print(phi)

# test = [[1, 2, 3, 4, 5, 6],
#  [7, 8, 9, 10, 11, 12],
#  [13, 14, 15, 16, 17, 18]]
# test[1:-1, np.ix_([0, -1])] = test[1:-1, np.ix_([2, -3])]
# print(test)