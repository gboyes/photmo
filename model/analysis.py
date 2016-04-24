import os
import cv2
import datetime
import sys
import numpy as np

class Analysis(object):
    def __init__(self, target, dictionary, output_path, params=None):


        self.target = target
        self.dictionary = dictionary
        self.model = np.zeros(np.shape(self.target.image))
        self.residual = self.target.image.copy()
        self.output_path = output_path
        self.kIterations = params.get('iterations') or 100
        self.snap_to_x = params.get('snap_to_x') or 120
        self.snap_to_y = params.get('snap_to_y') or 160
        self.rep = params.get('rep') or sys.maxsize

    def _roundnum(self, num, mul):
        rem = num % mul
        if rem == 0:
            return num
        if num >= mul / 2:
            return (num - rem + mul)
        else:
            return num - rem

    def _indices(self, atom, x, y):
        if y + atom.height > self.target.height:
            yb = self.target.height
            ayb = atom.height - (y + atom.height - self.target.height)
        else:
            yb = y + atom.height
            ayb = atom.height

        if x + atom.width > self.target.width:
            xb = self.target.width
            axb = atom.width - (x + atom.width - self.target.width)
        else:
            xb = x + atom.width
            axb = atom.width

        return (xb, axb, yb, ayb)

    def start(self):
        count = 0
        write_count = 0
        winds = set([])
        alphaTarget = self.target.image[:, :, 3].copy()
        dsHeight = int(np.ceil(self.target.height / float(self.dictionary.max_height)))
        dsWidth = int(np.ceil(self.target.width / float(self.dictionary.max_width)))
        scaled = cv2.resize(alphaTarget, (dsWidth, dsHeight))
        indLkup = dict(zip(np.arange(len(self.dictionary.atoms)), np.zeros(len(self.dictionary.atoms))))
        self.tmpBuffer = np.zeros(np.shape(self.target.image))

        while count < self.kIterations:
            newDict = {}
            foundPoint = False
            maxDepth = 100
            maxDepthCounter = 0

            while (not foundPoint) and (maxDepthCounter < maxDepth):

                yo = self._roundnum(np.random.randint(self.target.height), self.snap_to_y)
                xo = self._roundnum(np.random.randint(self.target.width), self.snap_to_x)

                xo_ = np.ceil(xo / float(self.dictionary.max_width))
                yo_ = np.ceil(yo / float(self.dictionary.max_height))

                dsh, dsw = np.shape(scaled)
                if xo_ >= dsw:
                    xo_ = dsw - 1
                if yo_ >= dsh:
                    yo_ = dsh - 1

                if (scaled[int(yo_), int(xo_)] > 0.0):
                    foundPoint = True

                maxDepthCounter += 1

            if yo > self.target.height - 1:
                yo = self.target.height - 1
            if xo > self.target.width - 1:
                xo = self.target.width - 1

            newDict['yo'] = yo
            newDict['xo'] = xo
            newDict['coef'] = np.zeros((self.target.planes - 1, len(self.dictionary.atoms)))

            for k, atom in enumerate(self.dictionary.atoms):
                (xb, axb, yb, ayb) = self._indices(atom, xo, yo)
                for p in range(0, self.target.planes - 1):
                    newDict['coef'][p, k] = np.tensordot(atom.image[0:ayb, 0:axb, p], self.residual[yo:yb, xo:xb, p])

            potInds = np.argsort(np.sum(newDict['coef'], axis=0))[::-1]
            ind = None

            for candInd in potInds:
                if indLkup[candInd] < self.rep:
                    indLkup[candInd] += 1
                    ind = candInd
                    break

            if ind == None:
                ind = potInds[0]
                indLkup = dict(zip(np.arange(len(self.dictionary.atoms)), np.zeros(len(self.dictionary.atoms))))
                indLkup[potInds[0]] += 1

            atom = self.dictionary.atoms[ind]
            (xb, axb, yb, ayb) = self._indices(atom, xo, yo)
            c = newDict['coef'][:, ind]

            for p in range(0, self.target.planes - 1):
                self.residual[yo:yb, xo:xb, p] -= atom.image[0:ayb, 0:axb, p] * c[p]
                self.model[yo:yb, xo:xb, p] += atom.image[0:ayb, 0:axb, p] * c[p]
                self.tmpBuffer[yo:yb, xo:xb, p] += atom.image[0:ayb, 0:axb, p] * c[p]

            self.residual[yo:yb, xo:xb, 3] -= atom.image[0:ayb, 0:axb, 3]  # * self.target.image[yo:yb, xo:xb, 3]
            self.model[yo:yb, xo:xb, 3] += atom.image[0:ayb, 0:axb, 3]  # * self.target.image[yo:yb, xo:xb, 3]
            self.tmpBuffer[yo:yb, xo:xb, 3] += atom.image[0:ayb, 0:axb, 3]  # * self.target.image[yo:yb, xo:xb, 3]

            if count == 0:
                write_count += 1

            elif np.log2(count) % 1 == 0:
                lastpw2 = np.log2(count)
                nextpw2 = lastpw2 + 1
                pwdist = 2 ** nextpw2 - 2 ** lastpw2
                kdist = pwdist / 16
                winds = set(np.round(np.arange(2 ** lastpw2, 2 ** nextpw2, kdist)))
                write_count += 1

            elif count in winds:
                write_count += 1

            count += 1

        cv2.imwrite(self.output_path, self.model * 255)

