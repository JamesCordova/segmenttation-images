import numpy as np
from scipy.ndimage import gaussian_filter

from lv_set.drlse_algo import drlse_edge
from lv_set.potential_func import DOUBLE_WELL, SINGLE_WELL
from PIL import Image as im
import cv2

threshold_value = 15

def find_lsf(img: np.ndarray, initial_lsf: np.ndarray, timestep=1, iter_inner=10, iter_outer=30, lmda=5,
             alfa=-3, epsilon=1.5, sigma=2, potential_function=DOUBLE_WELL, threads = False):
    
    """
    :param img: Input image as a grey scale uint8 array (0-255)
    :param initial_lsf: Array as same size as the img that contains the seed points for the LSF.
    :param timestep: Time Step
    :param iter_inner: How many iterations to run drlse before showing the output
    :param iter_outer: How many iterations to run the iter_inner
    :param lmda: coefficient of the weighted length term L(phi)
    :param alfa: coefficient of the weighted area term A(phi)
    :param epsilon: parameter that specifies the width of the DiracDelta function
    :param sigma: scale parameter in Gaussian kernal
    :param potential_function: The potential function to use in drlse algorithm. Should be SINGLE_WELL or DOUBLE_WELL
    """
    if len(img.shape) != 2:
        raise Exception("Input image should be a gray scale one")

    if len(img.shape) != len(initial_lsf.shape):
        raise Exception("Input image and the initial LSF should be in the same shape")

    if np.max(img) <= 1:
        raise Exception("Please make sure the image data is in the range [0, 255]")

     
    noise_level = estimate_noise_level(img)
    
    # Ajustar sigma en función del nivel de ruido estimado
    print(noise_level)
    if noise_level < threshold_value:
        sigma = 0.8  # Usar un valor predeterminado si el ruido es bajo
    else:
        sigma =  3
    print(sigma)

    # parameters
    mu = 0.2 / timestep  # coefficient of the distance regularization term R(phi)
    img = np.array(img, dtype='float32')
    img_smooth = gaussian_filter(img, sigma)  # smooth image by Gaussian convolution
    data = im.fromarray(img_smooth)
    if data.mode != 'RGB':
        data = data.convert('RGB')
    data.save('img_with_filter_gauss.bmp')
    [Iy, Ix] = np.gradient(img_smooth)
    f = np.square(Ix) + np.square(Iy)
    g = 1 / (1 + f)  #  .

    # initialize LSF as binary step function
    
    phi = initial_lsf.copy()
    
    if not threads:
        from lv_set.show_fig import show_fig1, show_fig2, draw_all
        show_fig1(phi)
        show_fig2(phi, img)
    # ----> print('show fig 2 first time')
    
    if potential_function != SINGLE_WELL:
        potential_function = DOUBLE_WELL  # default choice of potential function

    # start level set evolution
    for n in range(iter_outer):
        phi = drlse_edge(phi, g, lmda, mu, alfa, epsilon, timestep, iter_inner, potential_function)
        #-----> print(phi)
        #----> print('show fig 2 for %i time' % n)
        if not threads:
            draw_all(phi, img)

    # refine the zero level contour by further level set evolution with alfa=0
    alfa = 0
    iter_refine = 10
    phi = drlse_edge(phi, g, lmda, mu, alfa, epsilon, timestep, iter_refine, potential_function)
    return phi

def estimate_noise_level(img: np.ndarray) -> float:
    """
    Estimate the noise level in the input image.
    
    :param img: Input image as a numpy array.
    :return: Estimated noise level.
    """
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    noise = np.mean(np.abs(img - blurred))
    return noise