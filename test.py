import math
import pickle
import re
from os import path
import skimage.io as io
import skimage.draw as draw
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import os


def select_points(num_points, image):
    plt.imshow(image)
    print("Select %d points." % num_points)
    points = np.array(plt.ginput(num_points, timeout=0))
    plt.close()
    return points

class image_warpper():

    def __init__(self, image1_name, image2_name, N=4):
        self.image1 = io.imread(image1_name)
        self.image2 = io.imread(image2_name)
        self.N = N
        if not (self.__points_exist__(image1_name) or self.__points_exist__(image2_name)):
            self.__define_points__(self.image1, image1_name)
            self.__define_points__(self.image2, image2_name)
        self.points1 = self.__load_points__(image1_name)
        self.points2 = self.__load_points__(image2_name)
        self.H = self.__homography__()
        self.blend = None

    def __define_points__(self, image, image_name):
        plt.imshow(image)
        points = plt.ginput(self.N, 0)
        plt.close()
        pickle_name = re.split("\.", image_name)[0] + ".p"
        pickle.dump(points, open(pickle_name, "wb"))

    def __load_points__(self, image_name):
        pickle_name = re.split("\.", image_name)[0] + ".p"
        points = pickle.load(open(pickle_name, "rb"))
        return points

    def __points_exist__(self, image_name):
        return path.exists(re.split("\.", image_name)[0] + ".p")



    def __homography__(self):
        im1_pts = self.points1
        im2_pts = self.points2
        x1, y1 = im1_pts[0]
        x2, y2 = im2_pts[0]

        A = np.matrix([
            [-x2, -y2, -1, 0, 0, 0, x1 * x2, y1 * x2],
            [0, 0, 0, -x2, -y2, -1, x1 * y2, y1 * y2]
        ])

        b = np.zeros((self.N * 2, 1))
        for i in range(self.N):
            b[i * 2] = - self.points1[i][0]
            b[i * 2 + 1] = - self.points1[i][1]


        for i in range(1, self.N):
            x1, y1 = im1_pts[i]
            x2, y2 = im2_pts[i]
            next_rows = np.matrix([
                [-x2, -y2, -1, 0, 0, 0, x1 * x2, y1 * x2],
                [0, 0, 0, -x2, -y2, -1, x1 * y2, y1 * y2]
            ])
            A = np.vstack([A, next_rows])

        H_arr = (lin.lstsq(A, b, rcond=-1)[0]).T[0]
        H = np.matrix([[H_arr[0], H_arr[1], H_arr[2]],
                      [H_arr[3], H_arr[4], H_arr[5]],
                       [H_arr[6], H_arr[7], 1.]])
        return H

    def __check_bounding_box__(self):
        shape = self.image2.shape
        max_x = shape[1]
        max_y = shape[0]

        lower_left = [[0], [max_y], [1]]
        lower_right = [[max_x], [max_y], [1]]
        upper_left = [[0], [0], [1]]
        upper_right = [[max_x], [0], [1]]

        new_bounds_list = [lower_left, lower_right,
                           upper_left, upper_right]


        new_bounds_list = [self.H * point for point in new_bounds_list]
        new_bounds_list = [point / point[2] for point in new_bounds_list]
        new_max_x = max(new_bounds_list, key=lambda x: x[0])[0].astype(np.int)
        new_max_y = max(new_bounds_list, key=lambda x: x[1])[1].astype(np.int)
        new_min_x = min(new_bounds_list, key=lambda x: x[0])[0].astype(np.int)
        new_min_y = min(new_bounds_list, key=lambda x: x[1])[1].astype(np.int)

        return [new_max_x[0, 0], new_max_y[0, 0],
           new_min_x[0, 0], new_min_y[0, 0]]

    def __create_polygon__(self):
        maxx, maxy, minx, miny = self.__check_bounding_box__()
        maxx = max(maxx, self.image1.shape[1], self.image2.shape[1])
        maxy = max(maxy, self.image1.shape[0], self.image2.shape[0])

        mask = draw.polygon([0, maxx + abs(minx), maxx + abs(minx), 0],
                            [0, 0, maxy + abs(miny), maxy + abs(miny)])
        return np.matrix(np.vstack([mask, np.ones(len(mask[0]))]))

    def __apply_H_transform__(self):
        mask = self.__create_polygon__()
        cc, rr, _ = mask
        mask_tr = (lin.inv(self.H) * mask)
        cc, rr, w = mask_tr

        cc = np.squeeze(np.asarray(cc))
        rr = np.squeeze(np.asarray(rr))
        w = np.squeeze(np.asarray(w))

        cc = (cc / w).astype(np.int)
        rr = (rr / w).astype(np.int)

        return [cc, rr, mask]

    def __compute_new_image__(self):
        maxx, maxy, minx, miny = self.__check_bounding_box__()
        maxx = max(maxx, self.image1.shape[1], self.image2.shape[1])
        maxy = max(maxy, self.image1.shape[0], self.image2.shape[0])
        new_image = np.zeros((maxy + abs(miny) + 1, maxx + abs(minx) + 1, 3), dtype="uint8")
        return new_image

    def __create_list_of_indices__(self, cc, rr):
        return np.where((cc >= 0) & (cc < self.image2.shape[1]) &
                   (rr >= 0) & (rr < self.image2.shape[0]))

    def __process_and_filter_indices__(self, cc, rr, mask, indices):
        cc = cc[indices]
        rr = rr[indices]

        x_orig, y_orig, _ = mask
        x_orig = np.squeeze(np.asarray(x_orig))
        y_orig = np.squeeze(np.asarray(y_orig))

        x_orig = x_orig[indices].astype(np.int)
        y_orig = y_orig[indices].astype(np.int)

        _, _, minx, miny = self.__check_bounding_box__()

        offset_x = abs(min(minx, 0))
        offset_y = abs(min(miny, 0))

        x_orig += offset_x
        y_orig += offset_y

        return [cc, rr, x_orig, y_orig, offset_x, offset_y]

    def __paste_transformed_image__(self, new_image, cc, rr, x_orig, y_orig):
        new_image[y_orig, x_orig] = self.image2[rr, cc]
        return new_image

    def __create_alpha_gradient__(self, exponent):
        alpha = np.cos(np.linspace(0, math.pi/2, int(self.image1.shape[1]/2))) ** exponent
        alpha = np.hstack([np.ones(int(self.image1.shape[1]/2), dtype="float64"), alpha])
        finalAlpha = alpha
        for i in range(self.image1.shape[0]-1):
            finalAlpha = np.vstack([finalAlpha, alpha])

        return finalAlpha.reshape((finalAlpha.shape[0], finalAlpha.shape[1], 1))

    def __blend_images__(self, alphaGrad, new_image, offx, offy):
        new_image1 = self.image1 * alphaGrad
        range_maxx = self.image1.shape[1] + offx
        range_maxy = self.image1.shape[0] + offy


        new_image[offy:range_maxy,
                  offx:range_maxx] = new_image1 * alphaGrad + \
                                                new_image[offy:range_maxy,
                                                          offx:range_maxx] * (1 - alphaGrad)

        return new_image

    def compute_morph(self, exponent):
        cc, rr, mask = self.__apply_H_transform__()
        new_image = self.__compute_new_image__()
        indices = self.__create_list_of_indices__(cc, rr)
        cc, rr, x_orig, y_orig, offx, offy = self.__process_and_filter_indices__(cc, rr,
                                                                                 mask,
                                                                                 indices)
        new_image = self.__paste_transformed_image__(new_image,
                                                     cc, rr,
                                                     x_orig, y_orig)
        alphaGrad = self.__create_alpha_gradient__(exponent)
        new_image = self.__blend_images__(alphaGrad, new_image, offx, offy)
        self.blend = new_image


    def save_result(self, filename):
        if self.blend is None :
            print("No result to save!")
            exit()

        io.imsave(filename, self.blend)

    def show_result(self):
        if self.blend is None:
            print("No result to show!")
            exit()
        io.imshow(self.blend)
        io.show()



imwrap = image_warpper("pic1.jpg", "pic2.jpg", N=4)
imwrap.compute_morph(2)
imwrap.show_result()