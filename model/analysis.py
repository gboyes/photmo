import cv2
import datetime
import sys
import os
import numpy as np
from multiprocessing import Process

class Analysis(object):
    def __init__(self, target, dictionary, params=None):

        self.timestamp = datetime.datetime.now()
        self.target = target
        self.dictionary = dictionary
        self.model = np.zeros(np.shape(self.target.image))
        self.residual = self.target.image.copy()
        self.outputDirectory = './tmp'
        self.kIterations = params.get('iterations') or 100
        self.snap_to_x = params.get('snap_to_x') or 120
        self.snap_to_y = params.get('snap_to_y') or 160
        self.rep = params.get('rep') or sys.maxsize
        self.random_walk = params.get('random_walk') or False

    def roundnum(self, num, mul):
        rem = num % mul
        if rem == 0:
            return num

        if num >= mul / 2:
            return (num - rem + mul)
        else:
            return num - rem

    # def writeIterationImage(self, kIteration, notificationHosts=[]):
    #
    #     # just the filename
    #     filename = "%s_%07d.png" % (self.timestamp.strftime("%Y-%m-%d_%H_%M_%S"), kIteration)
    #
    #     # the full path
    #     path = "%s/%s_%07d.png" % (self.outputDirectory, self.timestamp.strftime("%Y-%m-%d_%H_%M_%S"), kIteration)
    #
    #     # the remote path
    #     rpath = "%s/%s" % (self.outputIterationShared, filename)
    #
    #     q = self.tmpBuffer * 255
    #
    #     # local
    #     cv2.imwrite(path, q)
    #
    #     try:
    #         cv2.imwrite(rpath, q)
    #     except:
    #         print("Error writing iteration image to remote")
    #
    #     # local
    #     cv2.imwrite('./%s/%s' % (self.tmppath, filename), self.model * 255)
    #
    #     self.tmpBuffer = np.zeros(np.shape(self.target.image))

    def start(self):

        # self.tmppath = './tmp/' + self.timestamp.strftime("%Y-%m-%d_%H_%M_%S")
        # try:
        #     os.makedirs(self.tmppath)
        # except OSError:
        #     if not os.path.isdir(self.tmppath):
        #         raise

        count = 0
        writecount = 0
        winds = set([])
        modelParams = {}

        # downsample the target image alpha channel to optimize the search
        alphaTarget = self.target.image[:, :, 3].copy()
        dsHeight = int(np.ceil(self.target.height / float(self.dictionary.max_height)))
        dsWidth = int(np.ceil(self.target.width / float(self.dictionary.max_width)))

        scaled = cv2.resize(alphaTarget, (dsWidth, dsHeight))

        # counting
        indLkup = dict(zip(np.arange(len(self.dictionary.atoms)), np.zeros(len(self.dictionary.atoms))))
        self.tmpBuffer = np.zeros(np.shape(self.target.image))

        while count < self.kIterations:

            newDict = {}

            # make sure there's a non alphaed point
            foundPoint = False
            maxDepth = 100  # because we could be unlucky
            maxDepthCounter = 0

            while (not foundPoint) and (maxDepthCounter < maxDepth):

                yo = self.roundnum(np.random.randint(self.target.height), self.snap_to_y)
                xo = self.roundnum(np.random.randint(self.target.width), self.snap_to_x)

                xo_ = np.ceil(xo / float(self.dictionary.max_width))
                yo_ = np.ceil(yo / float(self.dictionary.max_height))

                dsh, dsw = np.shape(scaled)
                if xo_ >= dsw:
                    xo_ = dsw - 1
                if yo_ >= dsh:
                    yo_ = dsh - 1

                if (scaled[yo_, xo_] > 0.0):
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

                # bounds checking
                if yo + atom.height > self.target.height:
                    yb = self.target.height
                    ayb = atom.height - (yo + atom.height - self.target.height)
                else:
                    yb = yo + atom.height
                    ayb = atom.height

                if xo + atom.width > self.target.width:
                    xb = self.target.width
                    axb = atom.width - (xo + atom.width - self.target.width)
                else:
                    xb = xo + atom.width
                    axb = atom.width

                # compute the correlation over RGB
                for p in range(0, self.target.planes - 1):
                    newDict['coef'][p, k] = np.tensordot(atom.image[0:ayb, 0:axb, p], self.residual[yo:yb, xo:xb, p])

            potInds = np.argsort(np.sum(newDict['coef'], axis=0))[::-1]
            ind = None

            for candInd in potInds:
                if indLkup[candInd] < self.rep:
                    indLkup[candInd] += 1
                    ind = candInd
                    break

            # no viable ind, means that all of the candidate indices have been used minmaxRep times
            if ind == None:
                # by default, take the most correlated
                ind = potInds[0]
                indLkup = dict(zip(np.arange(len(self.dictionary.atoms)), np.zeros(len(self.dictionary.atoms))))
                indLkup[potInds[0]] += 1

            # quick and dirty, check the bounds again
            atom = self.dictionary.atoms[ind]
            if yo + atom.height > self.target.height:
                yb = self.target.height
                ayb = atom.height - (yo + atom.height - self.target.height)
            else:
                yb = yo + atom.height
                ayb = atom.height

            if xo + atom.width > self.target.width:
                xb = self.target.width
                axb = atom.width - (xo + atom.width - self.target.width)
            else:
                xb = xo + atom.width
                axb = atom.width

            c = newDict['coef'][:, ind]

            for p in range(0, self.target.planes - 1):
                self.residual[yo:yb, xo:xb, p] -= atom.image[0:ayb, 0:axb, p] * c[p]
                self.model[yo:yb, xo:xb, p] += atom.image[0:ayb, 0:axb, p] * c[p]
                self.tmpBuffer[yo:yb, xo:xb, p] += atom.image[0:ayb, 0:axb, p] * c[p]

            self.residual[yo:yb, xo:xb, 3] -= atom.image[0:ayb, 0:axb, 3]  # * self.target.image[yo:yb, xo:xb, 3]
            self.model[yo:yb, xo:xb, 3] += atom.image[0:ayb, 0:axb, 3]  # * self.target.image[yo:yb, xo:xb, 3]
            self.tmpBuffer[yo:yb, xo:xb, 3] += atom.image[0:ayb, 0:axb, 3]  # * self.target.image[yo:yb, xo:xb, 3]

            if count == 0:
                # self.writeIterationImage(writecount, iterSockets)
                writecount += 1


            elif np.log2(count) % 1 == 0:

                lastpw2 = np.log2(count)
                nextpw2 = lastpw2 + 1
                pwdist = 2 ** nextpw2 - 2 ** lastpw2
                kdist = pwdist / 16
                winds = set(np.round(np.arange(2 ** lastpw2, 2 ** nextpw2, kdist)))

                # self.writeIterationImage(writecount, iterSockets)
                writecount += 1

            elif count in winds:
                # self.writeIterationImage(writecount, iterSockets)
                writecount += 1

            count += 1

        # write the model

        filename = "%s_MODEL.png" % self.timestamp.strftime("%Y-%m-%d_%H_%M_%S")
        moviename = "%s_MODEL.mov" % self.timestamp.strftime("%Y-%m-%d_%H_%M_%S")

        localpath = "%s/%s" % (self.outputDirectory, filename)

        cv2.imwrite(localpath, self.model * 255)


        # # TOREMOVE
        # localmodel = open(localpath, 'rb')
        #
        # # spawn process for movie write
        # tmppath_ = self.tmppath
        # pattern_ = self.timestamp.strftime("%Y-%m-%d_%H_%M_%S")
        # sock = modelSocket
        # address = self.notificationVideo
        # p = Process(target=self._writemovie, args=(tmppath_, pattern_, remotePath, moviename, sock, address))
        # p.start()

    # def _writemovie(self, framespath, pattern, moviepath, moviename, socket, address):
    #     try:
    #         os.system(
    #             "ffmpeg -f image2 -r 24 -i %s/%s_%%7d.png -vcodec qtrle -pix_fmt argb -r 24 -b:v 64k -f mov -y %s -loglevel panic" % (
    #             framespath, pattern, moviepath))
    #     except:
    #         print("Error writing movie file")
    #
    #     for f in os.listdir(framespath):
    #         os.remove(framespath + "/%s" % f)
    #
    #     if socket:
    #         try:
    #             # block til done
    #             # subprocess.call("cp %s %s"%(localPath, self.outputRemote), shell=True)
    #             liblo.send(socket, "%s/%s" % (address, moviename))
    #         except IOError:
    #             print("Host doesn't exist or is down")
    #
    #     os.rmdir(framespath)
    #     print("video written")
