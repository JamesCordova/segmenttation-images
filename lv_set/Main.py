import numpy as np
from skimage.io import imread
import cv2 as cv
from lv_set.find_lsf import find_lsf
from lv_set.potential_func import *
from lv_set.show_fig import draw_all
from PIL import Image, ImageFilter

current_img = None
def custom_image_cells(img):
    global current_img
    current_img = img
    img = imread(current_img, True)
    if len(img.shape) == 3:
        img = Image.rgb2gray(img)
    img = np.interp(img, [np.min(img), np.max(img)], [0, 255])

    # initialize LSF as binary step function
    c0 = 2
    initial_lsf = c0 * np.ones(img.shape)
    # generate the initial
    # generate rectangles for the initial region R0
    row_index = int(40 * initial_lsf.shape[0] / 100)
    col_index = int(40 * initial_lsf.shape[1] / 100)
    initial_lsf[row_index:-row_index, col_index:-col_index] = -c0

    # parameters
    return {
        'img': img,
        'initial_lsf': initial_lsf,
        'timestep': 5,  # time step
        'iter_inner': 5,
        'iter_outer': 60,
        'lmda': 5,  # coefficient of the weighted length term L(phi)
        'alfa': 1.5,  # coefficient of the weighted area term A(phi)
        'epsilon': 1.5,  # parameter that specifies the width of the DiracDelta function
        'sigma': 1.5,  # scale parameter in Gaussian kernel
        'potential_function': DOUBLE_WELL,
    }

def custom_image_gourd(img):
    global current_img
    current_img = img
    img = imread(current_img, True)
    if len(img.shape) == 3:
        img = Image.rgb2gray(img)
    img = np.interp(img, [np.min(img), np.max(img)], [0, 255])

    # initialize LSF as binary step function
    c0 = 2
    initial_lsf = c0 * np.ones(img.shape)
    # generate the initial
    # generate rectangles for the initial region R0
    initial_lsf[24:35, 19:25] = -c0
    initial_lsf[24:35, 39:50] = -c0

    # parameters
    return {
        'img': img,
        'initial_lsf': initial_lsf,
        'timestep': 1,  # time step
        'iter_inner': 10,
        'iter_outer': 30,
        'lmda': 5,  # coefficient of the weighted length term L(phi)
        'alfa': -3,  # coefficient of the weighted area term A(phi)
        'epsilon': 1.5,  # parameter that specifies the width of the DiracDelta function
        'sigma': 0.8,  # scale parameter in Gaussian kernel
        'potential_function': DOUBLE_WELL,
    }

def gourd_params():
    global current_img
    # img = imread('im2.bmp', True)
    current_img = 'gourd.bmp'
    img = imread(current_img, True) # img matrix has values from 0 to 1 to express grayscale
    img = np.interp(img, [np.min(img), np.max(img)], [0, 255]) # Convert all values from 0-1 to 0-255 

    # initialize LSF as binary step function
    c0 = 2 # constant, it could be any number
    initial_lsf = c0 * np.ones(img.shape) # From an matrix of ones to c0 = 2 2 2; 2 2 2 ..
    # generate the initial region R0 as two rectangles
    initial_lsf[24:35, 19:25] = -c0
    initial_lsf[24:35, 39:50] = -c0

    # parameters
    return {
        'img': img,
        'initial_lsf': initial_lsf,
        'timestep': 1,  # time step
        'iter_inner': 10,
        'iter_outer': 30,
        'lmda': 5,  # coefficient of the weighted length term L(phi)
        'alfa': -3,  # coefficient of the weighted area term A(phi)
        'epsilon': 1.5,  # parameter that specifies the width of the DiracDelta function
        'sigma': 0.8,  # scale parameter in Gaussian kernel
        'potential_function': DOUBLE_WELL,
    }

def jsf_params():
    img = imread('jsf_img_2.bmp', True)
    img = np.interp(img, [np.min(img), np.max(img)], [0, 255])

    # initialize LSF as binary step function
    c0 = 2
    initial_lsf = c0 * np.ones(img.shape)
    # generate the initial region R0 as two rectangles
    initial_lsf[24:35, 19:25] = -c0
#   initial_lsf[24:35, 39:50] = -c0

    # parameters
    return {
        'img': img,
        'initial_lsf': initial_lsf,
        'timestep': 1,  # time step
        'iter_inner': 10,
        'iter_outer': 30,
        'lmda': 5,  # coefficient of the weighted length term L(phi)
        'alfa': -3,  # coefficient of the weighted area term A(phi)
        'epsilon': 1.5,  # parameter that specifies the width of the DiracDelta function
        'sigma': 0.8,  # scale parameter in Gaussian kernel
        'potential_function': DOUBLE_WELL,
    }


def two_cells_params():
    global current_img
    current_img = 'twocells.bmp'
    img = imread(current_img, True)
    img = np.interp(img, [np.min(img), np.max(img)], [0, 255])

    # initialize LSF as binary step function
    c0 = 2
    initial_lsf = c0 * np.ones(img.shape)
    # generate the initial region R0 as two rectangles
    initial_lsf[9:55, 9:75] = -c0

    # parameters
    return {
        'img': img,
        'initial_lsf': initial_lsf,
        'timestep': 5,  # time step
        'iter_inner': 5,
        'iter_outer': 40,
        'lmda': 5,  # coefficient of the weighted length term L(phi)
        'alfa': 1.5,  # coefficient of the weighted area term A(phi)
        'epsilon': 1.5,  # parameter that specifies the width of the DiracDelta function
        'sigma': 1.5,  # scale parameter in Gaussian kernel
        'potential_function': DOUBLE_WELL,
    }

def four_shapes():
    global current_img
    current_img = 'cross.jpg'
    img = imread(current_img, True)
    img = np.interp(img, [np.min(img), np.max(img)], [0, 255])

    # initialize LSF as binary step function
    c0 = 2
    initial_lsf = c0 * np.ones(img.shape)
    # generate the initial region R0 as two rectangles
    initial_lsf[9:55, 9:75] = -c0

    # parameters
    return {
        'img': img,
        'initial_lsf': initial_lsf,
        'timestep': 5,  # time step
        'iter_inner': 5,
        'iter_outer': 50,
        'lmda': 5,  # coefficient of the weighted length term L(phi)
        'alfa': 1.5,  # coefficient of the weighted area term A(phi)
        'epsilon': 1.5,  # parameter that specifies the width of the DiracDelta function
        'sigma': 1.5,  # scale parameter in Gaussian kernel
        'potential_function': DOUBLE_WELL,
    }

def four_gourd_params():
    # img = imread('im2.bmp', True)
    global current_img
    current_img = 'cross.jpg'
    img = imread(current_img, True) # img matrix has values from 0 to 1 to express grayscale
    img = np.interp(img, [np.min(img), np.max(img)], [0, 255]) # Convert all values from 0-1 to 0-255 

    # initialize LSF as binary step function
    c0 = 2 # constant, it could be any number
    initial_lsf = c0 * np.ones(img.shape) # From an matrix of ones to c0 = 2 2 2; 2 2 2 ..
    # generate the initial region R0 as two rectangles
    # initial_lsf[24:35, 19:25] = -c0
    initial_lsf[18:25, 19:25] = -c0
    initial_lsf[24:35, 39:50] = -c0

    # parameters
    return {
        'img': img,
        'initial_lsf': initial_lsf,
        'timestep': 1,  # time step
        'iter_inner': 10,
        'iter_outer': 50,
        'lmda': 5,  # coefficient of the weighted length term L(phi)
        'alfa': -3,  # coefficient of the weighted area term A(phi)
        'epsilon': 1.5,  # parameter that specifies the width of the DiracDelta function
        'sigma': 0.8,  # scale parameter in Gaussian kernel
        'potential_function': DOUBLE_WELL,
    }

def custom_shapes_param():
    global current_img
    current_img = '2.bmp'
    img = imread(current_img, True)
    img = np.interp(img, [np.min(img), np.max(img)], [0, 255])

    # initialize LSF as binary step function
    c0 = 2
    initial_lsf = c0 * np.ones(img.shape)
    # generate the initial region R0 as two rectangles
    initial_lsf[20:40, 20:30] = -c0 # forma estrella
    initial_lsf[13:30, 46:60] = -c0 # cilindro
    initial_lsf[50:55, 50:65] = -c0 # elipse

    # parameters
    return {
        'img': img,
        'initial_lsf': initial_lsf,
        'timestep': 5,  # time step
        'iter_inner': 5,
        'iter_outer': 50,
        'lmda': 5,  # coefficient of the weighted length term L(phi)
        'alfa': -4,  # coefficient of the weighted area term A(phi)
        'epsilon': 1.5,  # parameter that specifies the width of the DiracDelta function
        'sigma': 1.5,  # scale parameter in Gaussian kernel
        'potential_function': DOUBLE_WELL,
    }

def custom_shapes_separated_param():
    global current_img
    current_img = '2.bmp'
    img = imread(current_img, True)
    img = np.interp(img, [np.min(img), np.max(img)], [0, 255])

    # initialize LSF as binary step function
    c0 = 2
    initial_lsf = c0 * np.ones(img.shape)
    # generate the initial region R0 as two rectangles
    initial_lsf[10:60, 10:40] = -c0 # forma estrella
    initial_lsf[3:40, 46:70] = -c0 # cilindro
    # initial_lsf[40:65, 40:75] = -c0 # elipse

    # parameters
    return {
        'img': img,
        'initial_lsf': initial_lsf,
        'timestep': 5,  # time step
        'iter_inner': 5,
        'iter_outer': 50,
        'lmda': 5,  # coefficient of the weighted length term L(phi)
        'alfa': 1.5,  # coefficient of the weighted area term A(phi)
        'epsilon': 1.5,  # parameter that specifies the width of the DiracDelta function
        'sigma': 1.5,  # scale parameter in Gaussian kernel
        'potential_function': DOUBLE_WELL,
    }

def heart_param():
    global current_img
    current_img = 'heart_ct.bmp'
    img = imread(current_img, True)
    img = np.interp(img, [np.min(img), np.max(img)], [0, 255])

    # initialize LSF as binary step function
    c0 = 2
    initial_lsf = c0 * np.ones(img.shape)
    # generate the initial region R0 as two rectangles
    initial_lsf[10:100, 45:60] = -c0 # area de interes
    initial_lsf[53:86, 93:137] = -c0 # 
    # initial_lsf[40:65, 40:75] = -c0 # 

    # parameters
    return {
        'img': img,
        'initial_lsf': initial_lsf,
        'timestep': 5,  # time step
        'iter_inner': 5,
        'iter_outer': 50,
        'lmda': 5,  # coefficient of the weighted length term L(phi)
        'alfa': -3,  # coefficient of the weighted area term A(phi)
        'epsilon': 1.5,  # parameter that specifies the width of the DiracDelta function
        'sigma': 1.5,  # scale parameter in Gaussian kernel
        'potential_function': DOUBLE_WELL,
    }
    
def semi_circles_param():
    global current_img
    current_img = 'semi_circle_6.jpg'
    img = imread(current_img, True)
    img = np.interp(img, [np.min(img), np.max(img)], [0, 255])

    # initialize LSF as binary step function
    c0 = 2
    initial_lsf = c0 * np.ones(img.shape)
    # generate the initial region R0 as two rectangles
    initial_lsf[25:35, 25:38] = -c0 # 
    initial_lsf[25:35, 46:55] = -c0 # 

    # parameters
    return {
        'img': img,
        'initial_lsf': initial_lsf,
        'timestep': 5,  # time step
        'iter_inner': 5,
        'iter_outer': 40,
        'lmda': 5,  # coefficient of the weighted length term L(phi)
        'alfa': -4,  # coefficient of the weighted area term A(phi)
        'epsilon': 1.5,  # parameter that specifies the width of the DiracDelta function
        'sigma': 1.5,  # scale parameter in Gaussian kernel
        'potential_function': DOUBLE_WELL,
    }
    
def semi_circles_outside_param():
    global current_img
    current_img = 'semi_circle_6.jpg'
    img = imread(current_img, True)
    img = np.interp(img, [np.min(img), np.max(img)], [0, 255])

    # initialize LSF as binary step function
    c0 = 2
    initial_lsf = c0 * np.ones(img.shape)
    # generate the initial region R0 as two rectangles
    initial_lsf[15:55, 15:70] = -c0 # 
    # initial_lsf[25:35, 46:55] = -c0 # 

    # parameters
    return {
        'img': img,
        'initial_lsf': initial_lsf,
        'timestep': 5,  # time step
        'iter_inner': 5,
        'iter_outer': 200,
        'lmda': 5,  # coefficient of the weighted length term L(phi)
        'alfa': 2,  # coefficient of the weighted area term A(phi)
        'epsilon': 1.5,  # parameter that specifies the width of the DiracDelta function
        'sigma': 1.5,  # scale parameter in Gaussian kernel
        'potential_function': DOUBLE_WELL,
    }
    
def execution_with_threads():
    from lv_set.MainCopy import process_and_show_images
    process_and_show_images()

def denoise(img):
    # im1 = img.filter(ImageFilter.BLUR)
    im1 = img.filter(ImageFilter.MinFilter(3))
    # im1 = img.filter(ImageFilter.MinFilter)
    im1.save("testing.bmp")
    return im1


# params = jsf_params()
params = gourd_params()
# params = two_cells_params()

# params = four_shapes()
# params = four_gourd_params()

# params = custom_shapes_param()
# params = custom_shapes_separated_param()

# params = heart_param()

# params = semi_circles_param()
# params = semi_circles_outside_param()

# params = custom_image_gourd('circle_0_noise.jpg')
# params = custom_image_gourd('circle_with_noise.jpg')
# params = custom_image_gourd('circle_noiseness.jpg')

# params = custom_image_cells('scan_knee.jpg')

list_options = [gourd_params, two_cells_params, four_shapes, 
                four_gourd_params, custom_shapes_param, custom_shapes_separated_param, 
                heart_param, semi_circles_param, semi_circles_outside_param, custom_image_gourd, 
                custom_image_cells, execution_with_threads,exit]

# A menu trough console to select the method to use and exit
# by choicing a number and show the result
print('Select the method to use:')
print('1. gourd_params')
print('2. two_cells_params')
print('2. four_shapes')
print('4. four_gourd_params')
print('5. custom_shapes_param')
print('6. custom_shapes_separated_param')
print('7. heart_param')
print('8. semi_circles_param')
print('9. semi_circles_outside_param')
print('10. custom_image_gourd')
print('11. custom_image_cells')
print('12. Execution with threads')
print('13. Exit')
menu = input('Enter a number: ')

# now execute the option
params = list_options[int(menu) - 1]()

denoise(Image.open(current_img))
phi = find_lsf(**params)

print('Show final output')
draw_all(phi, params['img'], 100)

# image = Image.open("gourd.bmp")
image = Image.open(current_img)
mode = image.mode
image_array = np.array(image)
# print(image_array)
if mode == 'L':
    print('The image is grayscale')
    # phi mask
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(suppress=True,precision=8)
    # phi_truncated = np.where(np.logical_and(phi != 2, np.abs(phi - np.round(phi)) > 1e-10), np.round(phi), phi)
    # phi_truncated = np.trunc(phi * 100) / 100
    # phi_truncated = np.ceil(phi * 100) / 100
    phi_truncated = np.floor(phi)
    # print(phi)
    # print(phi_truncated)
    # print(image_array[phi_truncated <= 0])
    image_array[phi_truncated <= 0] = 255 - image_array[phi_truncated <= 0]
    modified_image = Image.fromarray(image_array)

    # Save the modified image
    modified_image.save('modified_image.jpg')
elif mode == 'RGB':
    print('The image is NOT grayscale')
    # phi mask
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(suppress=True,precision=8)
    # phi_truncated = np.where(np.logical_and(phi != 2, np.abs(phi - np.round(phi)) > 1e-10), np.round(phi), phi)
    # phi_truncated = np.trunc(phi * 100) / 100
    # phi_truncated = np.ceil(phi * 100) / 100
    phi_truncated = np.floor(phi)
    # print(phi)
    # print(phi_truncated)
    # print(image_array[phi_truncated <= 0])
    # color opuesto
    image_array[phi_truncated <= 0] = 255 - image_array[phi_truncated <= 0]
    
    modified_image = Image.fromarray(image_array)
    print('The image is RGB (no alpha channel)')
    # Save the modified image
    modified_image.save('modified_image.jpg')
elif mode == 'RGBA':
    print('The image is RGB with an alpha channel (opacity)')
else:
    print(f'The image has an unknown mode: {mode}')
