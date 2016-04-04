import cv2
import numpy as np
import scipy.linalg as linalg
from scipy.signal import hann


class Target(object):
    def __init__(self, path_to_file):
        self.path = path_to_file
        i = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        if i is None:
            raise Exception

        self.image = i.astype(float) / 255.0
        self.height, self.width, self.planes = np.shape(self.image)
        if self.planes < 4:
            z = np.ones((self.height, self.width, 4))
            z[:, :, 0:self.planes] = self.image
            self.image = z
            self.height, self.width, self.planes = np.shape(self.image)


class Atom(object):
    def __init__(self, path_to_file, scalar=1, windowed=True):
        self.path = path_to_file
        i = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        if i is None:
            raise Exception

        if scalar != 1:
            i = cv2.resize(i, (0, 0), fx=scalar, fy=scalar)
            self.image = i.astype(float) / 255.0
        else:
            self.image = i.astype(float) / 255.0

        self.height, self.width, self.planes = np.shape(self.image)
        if self.planes < 4:
            z = np.ones((self.height, self.width, 4))
            z[:, :, 0:self.planes] = self.image
            self.image = z
            self.height, self.width, self.planes = np.shape(self.image)

        if windowed:
            win = np.vstack(hann(self.height)) * hann(self.width)
            for i in range(self.planes):
                self.image[:, :, i] *= win

        self.image[:, :, 0:3] *= 1. / linalg.norm(self.image[:, :, 0:3])  # normalize


class Dictionary(object):
    def __init__(self, images, params=None):
        self.scalars = params.get('scalars') or [1]
        self.windowed = params.get('windowed') or True
        self.images = images
        self.max_height = 1.0
        self.max_width = 1.0
        self.atoms = []

        for f in self.images:
            for s in self.scalars:
                try:
                    a = Atom(f, scalar=s, windowed=self.windowed)
                    self.atoms.append(a)
                    if a.height > self.max_height:
                        self.max_height = a.height
                    if a.width > self.max_width:
                        self.max_width = a.width
                except Exception as e:
                    print('Error making atom {}'.format(e))
