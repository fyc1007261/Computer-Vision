import numpy as np
import copy


class Image:
    def __init__(self, matrix):
        self._matrix = np.array(matrix)
        self._status = "color"
        self._size = self._matrix.size
        # dealing with .png
        if self._matrix.max() <= 1:
            self._matrix *= 255
        self._raw = copy.deepcopy(self._matrix)

    def get_matrix(self):
        return self._matrix

    def back_to_raw(self):
        self._matrix = copy.deepcopy(self._raw)

    def rgb2gray(self):
        if self._status == "color":
            self._matrix = np.dot(self._matrix[..., :3], [0.299, 0.587, 0.114])
            self._status = "gray"
            return 0
        else:
            return -1

    def to_binary(self):
        # use OTSU algorithm
        if self._status == "color":
            self.rgb2gray()
        hist = np.histogram(np.reshape(self._matrix, [-1]), range=(0, 256), bins=256)[0]
        threshold = 0
        tot = np.sum(hist * np.array(range(0, 256)))

        # calculate threshold
        max_var = 0
        for i in range(256):
            white_proportion = np.sum(hist[0:i]) / self._size
            black_proportion = np.sum(hist[i:]) / self._size
            if white_proportion == 0:
                continue
            if white_proportion == 1:
                break
            white_avg = np.sum(hist[0:i] * np.array(range(0, i))) / (white_proportion * self._size)
            black_avg = (tot - white_avg * white_proportion * self._size) / (self._size * black_proportion)
            var = white_proportion * black_proportion * ((white_avg - black_avg) ** 2)
            if var > max_var:
                max_var = var
                threshold = i

        # gray to binary
        for i in range(self._matrix.shape[0]):
            for j in range(self._matrix.shape[1]):
                if self._matrix[i, j] > threshold:
                    self._matrix[i, j] = 255
                else:
                    self._matrix[i, j] = 0
        self._status = "binary"

    def bin_erosion(self, kernel=np.zeros((2, 2))+255):
        k_width, k_height = kernel.shape[1], kernel.shape[0]
        new_mat = np.zeros((self._matrix.shape[0], self._matrix.shape[1]))
        self._matrix = np.hstack((self._matrix, np.zeros((self._matrix.shape[0], k_width-1))))
        self._matrix = np.vstack((self._matrix, np.zeros((k_height-1, self._matrix.shape[1]))))
        for i in range(self._matrix.shape[0] - k_height + 1):
            for j in range(self._matrix.shape[1] - k_width + 1):
                value = np.max(kernel - self._matrix[i: i+k_height, j: j+k_width])
                new_mat[i, j] = 255 - value
        self._matrix = new_mat

    def bin_dilation(self, kernel=np.zeros((2, 2))+255):
        k_width, k_height = kernel.shape[1], kernel.shape[0]
        new_mat = np.zeros((self._matrix.shape[0], self._matrix.shape[1]))
        self._matrix = np.hstack((self._matrix, np.zeros((self._matrix.shape[0], k_width - 1))))
        self._matrix = np.vstack((self._matrix, np.zeros((k_height - 1, self._matrix.shape[1]))))


        for i in range(self._matrix.shape[0] - k_height + 1):
            for j in range(self._matrix.shape[1] - k_width + 1):
                value = np.max(kernel + self._matrix[i: i + k_height, j: j + k_width])
                new_mat[i, j] = value - 255
        self._matrix = new_mat





