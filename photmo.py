#!/usr/bin/env python

import os
import sys
import json
import cv
import cv2
import numpy as np
import scipy.linalg as linalg
import datetime
import liblo
import time
import subprocess

from multiprocessing import Process
from scipy.signal import hann

class PhotmoTarget():
    
    '''An analysis target image'''
    
    def __init__(self, path_to_file):
        self.path = path_to_file
        
        i = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        
        #imread doesn't raise any exception if the path is bad 
        if i is None:
            print('No valid image')
            raise Exception
            
        self.image = i.astype(float) / 255.0
        self.height, self.width, self.planes = np.shape(self.image)
        
        #fake the alpha channel if it doesn't exist
        if self.planes < 4:
            print('Missing data, faking missing values')
            z = np.ones((self.height, self.width, 4))
            z[:, :, 0:self.planes] = self.image
            self.image  = z
            self.height, self.width, self.planes = np.shape(self.image)
            
        
    
class PhotmoAtom():
    
    '''An image in an analysis set'''
    
    def __init__(self, path_to_file, scalar=1, windowed=True):
        
        self.path = path_to_file
        
        i = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        if i is None:
            print('No valid image')
            raise Exception
        
        if scalar != 1:
            i = cv2.resize(i, (0, 0), fx=scalar, fy=scalar)
            self.image = i.astype(float) / 255.0
        else:
            self.image = i.astype(float) / 255.0
            
        self.height, self.width, self.planes = np.shape(self.image)
        
        #fake the alpha channel if it doesn't exist
        if self.planes < 4:
            print('Missing data, faking missing values')
            
            z = np.ones((self.height, self.width, 4))
            z[:, :, 0:self.planes] = self.image
            self.image  = z
            self.height, self.width, self.planes = np.shape(self.image)
        
        if windowed:
            win = np.vstack(hann(self.height)) * hann(self.width)
            for i in range(self.planes):
                self.image[:, :, i] *= win
        
        self.image[:, :, 0:3] *= 1./linalg.norm(self.image[:, :, 0:3])#normalize

class PhotmoDictionary():
    
    '''An analysis set'''
    
    def __init__(self, path_to_dir, params=None):
        if params:
            try:
                self.kImages = params['num_images']
                
            except KeyError:
                self.kImages = 100
                
            try:
                self.scalars = params['scalars']
            except KeyError:
                self.scalars = [1]
                
            try:
                self.windowed = params['windowed']
            except KeyError:
                self.windowed = True
                
        #set the default params         
        else:
            self.kImages = 100
            
        self.maxHeight = 1.0
        self.maxWidth = 1.0
            
        #sort the files on the basis of creation date, take the N most recent
        files = [os.path.join(path_to_dir, f) for f in os.listdir(path_to_dir)] # add path to each file
        files.sort(key=lambda x: os.path.getmtime(x))
        files = files[::-1]
        
        self.atoms = []
        count = 0
        for f in files:
            if count < self.kImages:
                for s in self.scalars:
                    try:
                        a = PhotmoAtom(f, scalar=s, windowed=self.windowed)
                        self.atoms.append(a)
                        
                        #store the maxes for later
                        if a.height > self.maxHeight:
                            self.maxHeight = a.height
                        if a.width  > self.maxWidth:
                            self.maxWidth = a.width
                            
                        count +=1
                        
                    except Exception:
                        print('Error making atom')
                        continue
            else:
                break
                
    
class PhotmoAnalysis():
    
    '''A class to manage the analysis of a target PhotmoTarget using a PhotmoDictionary'''
    
    def __init__(self, target, dictionary, params=None):
        
        #setup memory 
        self.timestamp = datetime.datetime.now()
        self.target = target
        self.dictionary = dictionary
        self.model = np.zeros(np.shape(self.target.image))
        self.residual = self.target.image.copy()
        
        print('Length of dictionary %d'%len(self.dictionary.atoms))
        
        if params:
            try:
                self.outputDirectory = params['output_path']
            except KeyError:
                print('Key error')
                self.outputDirectory = '/tmp'
                
            try:
                self.modelAddr = params['model_output']
            except KeyError:
                print('Key Error')
                
            try:
                self.iterAddr = params['iter_output']
            except KeyError:
                print('Key Error')
                
            try:
                self.kIterations = params['num_iter']
            except KeyError:
                self.kIterations = 100
                
            try:
                self.snapx = params['snapx']
            except KeyError:
                self.snapx = 120
            
            try:
                self.snapy = params['snapy']
            except KeyError:
                self.snapy = 160
                
            try:
                self.minmaxRep = params['minmaxRep']
            except KeyError:
                self.minmaxRep = sys.maxsize
                
            try:
                self.randomWalk = params['randomWalk']
            except KeyError:
                self.randomWalk = 0
        
        #default config
        else:
            self.kIterations = 100
            
        
    def roundnum(self, num, mul):
        rem = num % mul
        if rem == 0:
            return num
        
        if num >= mul/2:
             return (num - rem + mul)
        else:
             return num-rem    
    
    def writeIterationImage(self, kIteration, iSocket):
        
        filename = "%s_%07d.png"%(self.timestamp.strftime("%Y-%m-%d_%H_%M_%S"), kIteration)
        path = "%s/%s_%07d.png"%(self.outputDirectory, self.timestamp.strftime("%Y-%m-%d_%H_%M_%S"), kIteration)
        cv2.imwrite(path, self.tmpBuffer * 255)
        cv2.imwrite('./tmp/%s'%filename, self.model * 255)
        self.tmpBuffer = np.zeros(np.shape(self.target.image))
            
        if iSocket:
            liblo.send(iSocket, filename)
        
        #TODO: write current model into the a tmp directory, 
        
    
    def start(self):
        
        '''Start the analysis'''
        
        print('Analysis started.... ')
        
        iterSocket = None
        if self.iterAddr:
            iterSocket = liblo.Address(self.iterAddr['host'], self.iterAddr['port'] )
        
        modelSocket = None
        if self.modelAddr:
            modelSocket = liblo.Address(self.modelAddr['host'], self.modelAddr['port'] )
        
        count = 0
        writecount = 0
        winds = set([])
        modelParams = {}
        
        #downsample the target image alpha channel to optimize the search
        alphaTarget = self.target.image[:, :, 3].copy()
        dsHeight = int(np.ceil(self.target.height / float(self.dictionary.maxHeight)))
        dsWidth = int(np.ceil(self.target.width / float(self.dictionary.maxWidth)))
        
        scaled = cv2.resize(alphaTarget, (dsWidth, dsHeight))
        
        #counting
        indLkup = dict(zip(np.arange(len(self.dictionary.atoms)), np.zeros(len(self.dictionary.atoms))))
        self.tmpBuffer = np.zeros(np.shape(self.target.image))
        
        while count < self.kIterations:
            
            newDict = {}
            
            #make sure there's a non alphaed point
            foundPoint = False
            maxDepth = 100 #because we could be unlucky
            maxDepthCounter = 0
            
            while (not foundPoint) and (maxDepthCounter < maxDepth):
                
                yo = self.roundnum(np.random.randint(self.target.height), self.snapy)
                xo = self.roundnum(np.random.randint(self.target.width), self.snapx)
                
                xo_ = np.ceil(xo / float(self.dictionary.maxWidth))
                yo_ = np.ceil(yo / float(self.dictionary.maxHeight))
                
                dsh, dsw =  np.shape(scaled)
                if xo_ >= dsw:
                    xo_ = dsw - 1
                if yo_ >= dsh:
                    yo_ = dsh - 1
                
                if (scaled[yo_,xo_] > 0.0):
                    foundPoint = True
                
                maxDepthCounter += 1
                
            
            if yo > self.target.height-1:
                yo = self.target.height-1
            if xo > self.target.width-1:
                xo = self.target.width-1
            
            newDict['yo'] = yo
            newDict['xo'] = xo
            newDict['coef'] = np.zeros((self.target.planes-1, len(self.dictionary.atoms)))

            for k, atom in enumerate(self.dictionary.atoms):
                
                #bounds checking
                if yo+atom.height > self.target.height:
                    yb = self.target.height
                    ayb = atom.height - (yo+atom.height - self.target.height)
                else:
                    yb = yo+atom.height
                    ayb = atom.height
                    
                if xo+atom.width > self.target.width:
                    xb = self.target.width
                    axb = atom.width - (xo+atom.width - self.target.width)
                else:
                    xb = xo+atom.width
                    axb = atom.width  
                
                #compute the correlation over RGB
                for p in range(0, self.target.planes-1):
                    newDict['coef'][p, k] = np.tensordot(atom.image[0:ayb, 0:axb, p], self.residual[yo:yb, xo:xb, p])
            
            
            potInds = np.argsort(np.sum(newDict['coef'], axis=0))[::-1]
            ind = None
            
            for candInd in potInds:
                if indLkup[candInd] < self.minmaxRep:
                    indLkup[candInd] += 1
                    ind = candInd
                    break
            
            
            #no viable ind, means that all of the candidate indices have been used minmaxRep times
            if ind == None:
                
                #by default, take the most correlated
                ind = potInds[0]
                indLkup = dict(zip(np.arange(len(self.dictionary.atoms)), np.zeros(len(self.dictionary.atoms))))
                indLkup[potInds[0]] += 1
                
            
            #quick and dirty, check the bounds again
            atom = self.dictionary.atoms[ind]
            if yo+atom.height > self.target.height:
                yb = self.target.height
                ayb = atom.height - (yo+atom.height - self.target.height)
            else:
                yb = yo+atom.height
                ayb = atom.height
                    
            if xo+atom.width > self.target.width:
                xb = self.target.width
                axb = atom.width - (xo+atom.width - self.target.width)
            else:
                xb = xo+atom.width
                axb = atom.width  
                
            c = newDict['coef'][:, ind]
            
            for p in range(0, self.target.planes-1):
                    
                self.residual[yo:yb, xo:xb, p] -= atom.image[0:ayb, 0:axb, p] * c[p]
                self.model[yo:yb, xo:xb, p] +=  atom.image[0:ayb, 0:axb, p] * c[p]
                self.tmpBuffer[yo:yb, xo:xb, p] +=  atom.image[0:ayb, 0:axb, p] * c[p]
                
            
            self.residual[yo:yb, xo:xb, 3] -= atom.image[0:ayb, 0:axb, 3] #* self.target.image[yo:yb, xo:xb, 3]
            self.model[yo:yb, xo:xb, 3] +=  atom.image[0:ayb, 0:axb, 3] #* self.target.image[yo:yb, xo:xb, 3]
            self.tmpBuffer[yo:yb, xo:xb, 3] +=  atom.image[0:ayb, 0:axb, 3] #* self.target.image[yo:yb, xo:xb, 3]
             
           
            if count == 0:
                self.writeIterationImage(writecount, iterSocket)
                writecount += 1
            
           
            elif np.log2(count) % 1 == 0:
                
                lastpw2 = np.log2(count)
                nextpw2 = lastpw2 + 1
                pwdist = 2**nextpw2 - 2**lastpw2
                kdist = pwdist / 16
                winds = set(np.round(np.arange(2**lastpw2, 2**nextpw2, kdist)))
                
                self.writeIterationImage(writecount, iterSocket)
                writecount += 1
                    
            elif count in winds:
                self.writeIterationImage(writecount, iterSocket)
                writecount += 1
                
            count += 1
            
            
        #write the model
        filename = "%s_MODEL.gif"%self.timestamp.strftime("%Y-%m-%d_%H_%M_%S")
        #path = "%s/%s_MODEL.png"%(self.outputDirectory, self.timestamp.strftime("%Y-%m-%d_%H_%M_%S"))
        
        #cv2.imwrite(path, self.model * 255)
        
            
        
        gifpath = "%s/%s_MODEL.gif"%(self.outputDirectory, self.timestamp.strftime("%Y-%m-%d_%H_%M_%S"))
        
        #TODO: make subprocess and spawn ffmpeg to make gif, send a message somewhere to signal that the gif is complete
        os.system("convert ./tmp/*.png %s"%gifpath)
        #os.system("ffmpeg -f image2 -i ./tmp/%s_%%7d.png -pix_fmt bgra %s"%(self.timestamp.strftime("%Y-%m-%d_%H_%M_%S"), gifpath))
        for f in os.listdir("./tmp") :
            os.remove("./tmp/%s"%f)
            
        if modelSocket:
            liblo.send(modelSocket, filename)
    
    
    
class PhotmoListener():
    
    '''Manages the input and analysis queue'''
    
    def __init__(self, config):
        
        self.DICT_PATH = config["dictionaryPath"]
        self.serverPort = config["serverPort"]
        self.serverPath = config["serverPath"]
        self.dictParams = config["dictionaryParams"]
        self.analysisParams = config["analysisParams"]
        
        self.targetPath = None
        self.configureNetwork()
    
    
    def configureNetwork(self):
        
        '''Configure the OSC network'''
        
        print("(Re)Configuring network")
        
        # create OSC server
        try:
            self.oscServer = liblo.Server(self.serverPort)
                    
        except liblo.ServerError, err:
            print str(err)
            return
            
            
        #callback function for the sever
        def handle_target(path, args):
            '''Callback function for babble''' 
            self.targetPath = args[0]
            
        
        # register method taking an int to simulate the
        self.oscServer.add_method(self.serverPath, 's', handle_target)
        
    def randomWalk(self):
        
        self.dictParams["windowed"] = np.random.randint(2)
        
        k = len(self.dictParams["scalars"])
        
        #gamma = 0.75
        #gamma_  = 1.0 - gamma
        
        #self.dictParams["scalars"] = [(self.dictParams["scalars"][i%k] * gamma) + \
            #((np.random.randn() % 1) * gamma_) for i in range(0, np.random.randint(1, 3))]
        
        self.dictParams["scalars"] = [np.random.randint(10, 17) / 100.0 for i in range(0, np.random.randint(1, 3))]
        
        self.analysisParams["snapx"] = np.random.randint(1, np.ceil(480 * max(self.dictParams["scalars"])))
        self.analysisParams["snapy"] = np.random.randint(1, np.ceil(640 * max(self.dictParams["scalars"])))
        
        print(self.dictParams["scalars"])
    
    def listen(self):
        
        '''Listen for a notification to start the decomposition of an image'''
        
        while True:
    
            #This will set the targetPath
            self.oscServer.recv(1)
    
            if self.targetPath:
                
                #explicitly close the port to prevent intermediate queing
                self.oscServer.free()
                
                print(self.targetPath)
        
                try:
                    target = PhotmoTarget(self.targetPath)
                except Exception:
                    print("Couldn't make target")
                    self.targetPath = None
                    continue
        
                dictionary = PhotmoDictionary(self.DICT_PATH, params=self.dictParams)
                analysis = PhotmoAnalysis(target, dictionary, params=self.analysisParams)
                analysis.start()
        
                #reset the target path to None, in order to wait for next target, re-config the port
                self.configureNetwork()
                self.targetPath = None
                
                if analysis.randomWalk:
                    self.randomWalk()
        
    
            time.sleep(0.01)
    
def main(config):
    P = PhotmoListener(config)
    P.listen()


if __name__ == '__main__':
    
    path = sys.argv[1]
    config_file = open(path)
    config = json.load(config_file)
    main(config)

    
    

    
    


    

    
