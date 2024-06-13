import numpy as np
from skimage.io import imread
from skimage import measure
import matplotlib.pyplot as plt
import cv2
import threading

from lv_set.find_lsf import find_lsf
from lv_set.potential_func import *


def gourd_params(img):
    c0 = 2
    initial_lsf = c0 * np.ones(img.shape)
    initial_lsf[24:35, 19:25] = -c0
    initial_lsf[24:35, 39:50] = -c0

    return {
        'img': img,
        'initial_lsf': initial_lsf,
        'timestep': 1,
        'iter_inner': 10,
        'iter_outer': 30,
        'lmda': 5,
        'alfa': -3,
        'epsilon': 1.5,
        'sigma': 0.8,
        'potential_function': DOUBLE_WELL,
    }

def two_cells_params(img):
    c0 = 2
    initial_lsf = c0 * np.ones(img.shape)
    initial_lsf[9:55, 9:75] = -c0

    return {
        'img': img,
        'initial_lsf': initial_lsf,
        'timestep': 5,
        'iter_inner': 5,
        'iter_outer': 30,
        'lmda': 5,
        'alfa': 2,
        'epsilon': 1.5,
        'sigma': 1.5,
        'potential_function': DOUBLE_WELL,
    }

def preprocess_image(img_path):
    img = imread(img_path, as_gray=True)
    img = np.interp(img, [np.min(img), np.max(img)], [0, 255])
    # img = cv2.bilateralFilter(img.astype(np.float32), 9, 75, 75)
    return img

def process_image(img_path, params_func, results, index):
    img = preprocess_image(img_path)
    params = params_func(img)
    phi = find_lsf(**params, threads=True)

    contours = measure.find_contours(phi, 0)
    results[index] = (img, phi, contours)

def plot_contours(ax, img, contours):
    ax.imshow(img, cmap='gray')
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0],color = 'red', linewidth=2)

def process_and_show_images():
    image_paths = [
        'circle_0_noise.jpg',
        'circle_with_noise.jpg',
        'circle_noiseness.jpg'
    ]

    # Crear listas para almacenar los resultados
    gourd_results = [None] * len(image_paths)
    two_cells_results = [None] * len(image_paths)

    # Crear y ejecutar los hilos
    threads = []
    for i, img_path in enumerate(image_paths):
        t1 = threading.Thread(target=process_image, args=(img_path, gourd_params, gourd_results, i))
        t2 = threading.Thread(target=process_image, args=(img_path, two_cells_params, two_cells_results, i))
        threads.append(t1)
        threads.append(t2)
        t1.start()
        t2.start()

    # Esperar a que todos los hilos terminen
    for t in threads:
        t.join()

    # Mostrar los resultados en el hilo principal
    plt.ion()
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    for i in range(len(image_paths)):
        img, phi, contours = gourd_results[i]
        plot_contours(axs[0, i], img, contours)
        axs[0, i].set_title(f'Gourd Params {i+1}')
        axs[0, i].axis('off')
        
        img, phi, contours = two_cells_results[i]
        plot_contours(axs[1, i], img, contours)
        axs[1, i].set_title(f'Two Cells Params {i+1}')
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.show(block=True)  # Mostrar la figura y mantenerla abierta hasta que el usuario la cierre manualmente

# Ejecutar el procesamiento y mostrar las imágenes
process_and_show_images()
