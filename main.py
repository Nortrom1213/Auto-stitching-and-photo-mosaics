import matplotlib.animation as animation
import matplotlib.pyplot as plt
import skimage.transform as sktr
from skimage.draw import polygon
from scipy.spatial import Delaunay
import skimage as sk
import numpy as np
import numpy.linalg as lin
import pickle
import math
import re
import os
from os import path


def select_points(image, image_name):
    plt.imshow(image)
    print("Select %d points." % N)
    points = plt.ginput(N, 0)
    plt.close()
    pickle_name = image_name + ".npy"
    pickle.dump(points, open(pickle_name, "wb"))
    return points


def init(image1_name, image2_name):
    image1 = plt.imread(image1_name+'.jpg')
    image2 = plt.imread(image2_name+'.jpg')
    if not (path.exists(image1_name + ".npy") and path.exists(image2_name + ".npy")):
        select_points(image1, image1_name)
        select_points(image2, image2_name)
    points1 = pickle.load(open(image1_name + '.npy', "rb"))
    points2 = pickle.load(open(image2_name + '.npy', "rb"))
    H = computeH(points1, points2)
    return image1, image2, points1, points2, H


def computeH(im1_pts, im2_pts):
    x1, y1 = im1_pts[0]
    x2, y2 = im2_pts[0]
    N = len(im1_pts)

    A = np.matrix([
        [-x2, -y2, -1, 0, 0, 0, x1 * x2, y1 * x2],
        [0, 0, 0, -x2, -y2, -1, x1 * y2, y1 * y2]
    ])

    b = np.zeros((N * 2, 1))
    for i in range(N):
        b[i * 2] = - im1_pts[i][0]
        b[i * 2 + 1] = - im1_pts[i][1]

    for i in range(1, N):
        x1, y1 = im1_pts[i]
        x2, y2 = im2_pts[i]
        next_rows = np.matrix([
            [-x2, -y2, -1, 0, 0, 0, x1 * x2, y1 * x2],
            [0, 0, 0, -x2, -y2, -1, x1 * y2, y1 * y2]
        ])
        A = np.vstack([A, next_rows])

    pts = (lin.lstsq(A, b, rcond=-1)[0]).T[0]
    H = np.matrix([[pts[0], pts[1], pts[2]],
                   [pts[3], pts[4], pts[5]],
                   [pts[6], pts[7], 1.]])
    return H

def bound():
    shape = image2.shape
    max_x = shape[1]
    max_y = shape[0]

    new_bound = [[[0], [max_y], [1]],
                        [[max_x], [max_y], [1]],
                        [[0], [0], [1]],
                        [[max_x], [0], [1]]]

    new_bound = [H * point for point in new_bound]
    new_bound = [point / point[2] for point in new_bound]
    new_max_x = max(new_bound, key=lambda x: x[0])[0].astype(np.int)
    new_max_y = max(new_bound, key=lambda x: x[1])[1].astype(np.int)
    new_min_x = min(new_bound, key=lambda x: x[0])[0].astype(np.int)
    new_min_y = min(new_bound, key=lambda x: x[1])[1].astype(np.int)
    print(new_max_x)

    return [new_max_x[0, 0], new_max_y[0, 0],
        new_min_x[0, 0], new_min_y[0, 0]]


def compute_polygon():
    maxx, maxy, minx, miny = bound()
    maxx = max(maxx, image1.shape[1], image2.shape[1])
    maxy = max(maxy, image1.shape[0], image2.shape[0])

    mask = polygon([0, maxx + abs(minx), maxx + abs(minx), 0],
                        [0, 0, maxy + abs(miny), maxy + abs(miny)])
    return np.matrix(np.vstack([mask, np.ones(len(mask[0]))]))

def apply_H_transform():
    mask = compute_polygon()
    mask_tr = (lin.inv(H) * mask)
    cc, rr, w = mask_tr

    cc = np.squeeze(np.asarray(cc))
    rr = np.squeeze(np.asarray(rr))
    w = np.squeeze(np.asarray(w))

    cc = (cc / w).astype(np.int)
    rr = (rr / w).astype(np.int)

    return [cc, rr, mask]

def compute_new_image():
    maxx, maxy, minx, miny = bound()
    maxx = max(maxx, image1.shape[1], image2.shape[1])
    maxy = max(maxy, image1.shape[0], image2.shape[0])
    new_image = np.zeros((maxy + abs(miny) + 1, maxx + abs(minx) + 1, 3), dtype="uint8")
    return new_image

def create_list_of_indices(cc, rr):
    return np.where((cc >= 0) & (cc < image2.shape[1]) &
        (rr >= 0) & (rr < image2.shape[0]))

def process_and_filter_indices(cc, rr, mask, indices):
    cc = cc[indices]
    rr = rr[indices]

    x_orig, y_orig, _ = mask
    x_orig = np.squeeze(np.asarray(x_orig))
    y_orig = np.squeeze(np.asarray(y_orig))

    x_orig = x_orig[indices].astype(np.int)
    y_orig = y_orig[indices].astype(np.int)

    _, _, minx, miny = bound()

    offset_x = abs(min(minx, 0))
    offset_y = abs(min(miny, 0))

    x_orig += offset_x
    y_orig += offset_y

    return [cc, rr, x_orig, y_orig, offset_x, offset_y]

def paste_transformed_image(new_image, cc, rr, x_orig, y_orig):
    new_image[y_orig, x_orig] = image2[rr, cc]
    return new_image

def create_alpha_gradient(exponent):
    alpha = np.cos(np.linspace(0, math.pi/2, int(image1.shape[1]/2))) ** exponent
    alpha = np.hstack([np.ones(int(image1.shape[1]/2), dtype="float64"), alpha])
    finalAlpha = alpha
    for i in range(image1.shape[0]-1):
        finalAlpha = np.vstack([finalAlpha, alpha])

    return finalAlpha.reshape((finalAlpha.shape[0], finalAlpha.shape[1], 1))

def blend_image(alphaGrad, new_image, offx, offy):
    new_image1 = image1 * alphaGrad
    range_maxx = image1.shape[1] + offx
    range_maxy = image1.shape[0] + offy


    new_image[offy:range_maxy,
    offx:range_maxx] = new_image1 * alphaGrad + \
                new_image[offy:range_maxy, offx:range_maxx] * (1 - alphaGrad)

    return new_image



def compute_morph(exponent):
    cc, rr, mask = apply_H_transform()
    new_image = compute_new_image()
    indices = create_list_of_indices(cc, rr)
    cc, rr, x_orig, y_orig, offx, offy = process_and_filter_indices(cc, rr,
                                                                            mask,
                                                                            indices)
    new_image = paste_transformed_image(new_image,
                                            cc, rr,
                                            x_orig, y_orig)
    alphaGrad = create_alpha_gradient(exponent)
    new_image = blend_image(alphaGrad, new_image, offx, offy)
    blend = new_image

    return blend

img1_name = "pic1"
img2_name = "pic2"
NUM_POINTS = 4
N = 4
image1, image2, image1_points, image2_points, H = init(img1_name, img2_name)
blend = compute_morph(2)
plt.imshow(blend)
plt.show()