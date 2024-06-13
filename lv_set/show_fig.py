"""
This python code demonstrates an edge-based active contour model as an application of the
Distance Regularized Level Set Evolution (DRLSE) formulation in the following paper:

  C. Li, C. Xu, C. Gui, M. D. Fox, "Distance Regularized Level Set Evolution and Its Application to Image Segmentation",
     IEEE Trans. Image Processing, vol. 19 (12), pp. 3243-3254, 2010.

Author: Ramesh Pramuditha Rathnayake
E-mail: rsoft.ramesh@gmail.com

Released Under MIT License
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage import measure

plt.ioff()
fig1 = plt.figure(1)
ax1 = None  # Initialize axes variables
ax2 = None

def show_fig1(phi: np.ndarray):
    global ax1
    if ax1 is not None:
        ax1.cla()  # Clear the current axes
    else:
        ax1 = fig1.add_subplot(121, projection='3d')
        plt.subplots_adjust(wspace=0.5)
    y, x = phi.shape
    x = np.arange(0, x, 1)
    y = np.arange(0, y, 1)
    X, Y = np.meshgrid(x, y)
    # ax1.title.set_text('3D plot')
    ax1.plot_surface(X, Y, -phi, rstride=2, cstride=2, color='r', linewidth=0, alpha=0.6, antialiased=True)
    ax1.contour(X, Y, phi, 0, colors='g', linewidths=2)
    plt.draw()

def show_fig2(phi: np.ndarray, img: np.ndarray):
    global ax2
    if ax2 is not None:
        ax2.cla()  # Clear the current axes
    else:
        ax2 = fig1.add_subplot(122)
        plt.subplots_adjust(wspace=0.5)
    contours = measure.find_contours(phi, 0)
    ax2.imshow(img, interpolation='nearest', cmap=plt.get_cmap('gray'))
    for n, contour in enumerate(contours):
        ax2.plot(contour[:, 1], contour[:, 0], linewidth=2)
    plt.draw()

def draw_all(phi: np.ndarray, img: np.ndarray, pause=0.3):
    plt.ion() 
    show_fig2(phi, img)
    show_fig1(phi)
    plt.pause(pause)
