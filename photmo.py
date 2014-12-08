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

from multiprocessing import Process
from scipy.signal import hann

class PhotmoTarget():
    
    def __init__(self, path_to_file):
        self.path = path_to_file
        
        i = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        
        #imread doesn't raise any exception if the path is bad 
        if i is None:
            print('No valid image')
            raise Exception
            
        self.image = i.astype(float) / 255.0
        self.height, self.width, self.planes = np.shape(self.image) 
    
class PhotmoAtom():
    
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
        
        if windowed:
            win = np.vstack(hann(self.height)) * hann(self.width)
            for i in range(self.planes):
                self.image[:, :, i] *= win
        
        self.image[:, :, 0:3] *= 1./linalg.norm(self.image[:, :, 0:3])#normalize

class PhotmoDictionary():
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
            
        self.atoms = []
        count = 0
        for f in os.listdir(path_to_dir):
            if count < self.kImages:
                print(path_to_dir + '/' + f)
                for s in self.scalars:
                    try:
                        a = PhotmoAtom(path_to_dir + '/' + f, scalar=s, windowed=self.windowed)
                        self.atoms.append(a)
                        count +=1
                    except Exception:
                        print('Error making atom')
                        continue
            else:
                break
                
    
class PhotmoAnalysis():
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
    
    def start(self):
        
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
        
        tmpBuffer = np.zeros(np.shape(self.target.image))
        
        while count < self.kIterations:
            
            newDict = {}
            
            #TODO: add config variable for these spacings
            yo = self.roundnum(np.random.randint(self.target.height), self.snapy)
            xo = self.roundnum(np.random.randint(self.target.width), self.snapx)
            
            if yo > self.target.height-1:
                yo = self.target.height-1
            if xo > self.target.width-1:
                xo = self.target.width-1
            
            newDict['yo'] = yo
            newDict['xo'] = xo
            newDict['coef'] = np.zeros((self.target.planes-1, len(self.dictionary.atoms)))
            
            for k, atom in enumerate(self.dictionary.atoms):
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
                
                for p in range(0, self.target.planes-1):
                    newDict['coef'][p, k] = np.tensordot(atom.image[0:ayb, 0:axb, p], self.residual[yo:yb, xo:xb, p])
            
            ind = np.argmax(np.sum(newDict['coef'], axis=0))
            
            #quick and dirty
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
                tmpBuffer[yo:yb, xo:xb, p] +=  atom.image[0:ayb, 0:axb, p] * c[p]
                
            
            self.residual[yo:yb, xo:xb, 3] -= atom.image[0:ayb, 0:axb, 3] * self.target.image[yo:yb, xo:xb, 3]
            self.model[yo:yb, xo:xb, 3] +=  atom.image[0:ayb, 0:axb, 3] * self.target.image[yo:yb, xo:xb, 3]
            tmpBuffer[yo:yb, xo:xb, 3] +=  atom.image[0:ayb, 0:axb, 3] * self.target.image[yo:yb, xo:xb, 3]
             
            #ugly hacks
            if np.log2(count) % 1 == 0:
                
                lastpw2 = np.log2(count)
                nextpw2 = lastpw2 + 1
                pwdist = 2**nextpw2 - 2**lastpw2
                kdist = pwdist / 16
                winds = set(np.round(np.arange(2**lastpw2, 2**nextpw2, kdist)))
                
                filename = "%s_%07d.png"%(self.timestamp.strftime("%Y-%m-%d_%H_%M"), writecount)
                path = "%s/%s_%07d.png"%(self.outputDirectory, self.timestamp.strftime("%Y-%m-%d_%H_%M"), writecount)
                cv2.imwrite(path, tmpBuffer * 255)
                tmpBuffer = np.zeros(np.shape(self.target.image))
            
                if iterSocket:
                    liblo.send(iterSocket, filename)
                    
                writecount += 1
                    
            elif count in winds:
                filename = "%s_%07d.png"%(self.timestamp.strftime("%Y-%m-%d_%H_%M"), writecount)
                path = "%s/%s_%07d.png"%(self.outputDirectory, self.timestamp.strftime("%Y-%m-%d_%H_%M"), writecount)
                cv2.imwrite(path, tmpBuffer * 255)
                tmpBuffer = np.zeros(np.shape(self.target.image))
            
                if iterSocket:
                    liblo.send(iterSocket, filename)
                    
                writecount += 1
            
            
            count += 1
            print(count)
            
            
        filename = "%s_MODEL.png"%self.timestamp.strftime("%Y-%m-%d_%H_%M")
        path = "%s/%s_MODEL.png"%(self.outputDirectory, self.timestamp.strftime("%Y-%m-%d_%H_%M"))
        
        cv2.imwrite(path, self.model * 255)
        if modelSocket:
            liblo.send(modelSocket, filename)
    
    
    
class PhotmoListener():
    
    def __init__(self, config):
        
        self.DICT_PATH = config["dictionaryPath"]
        self.serverPort = config["serverPort"]
        self.serverPath = config["serverPath"]
        self.dictParams = config["dictionaryParams"]
        self.analysisParams = config["analysisParams"]
        
        self.targetPath = None
        self.configureNetwork()
    
    
    def configureNetwork(self):
        
        '''Configure the network'''
        
        print("Configuring network")
        
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
    
    def listen(self):
        
        while True:
    
            #TODO: this will queue signals received, decide if that's the intended behaviour
            self.oscServer.recv(1)
    
            if self.targetPath:
                print(self.targetPath)
        
                try:
                    target = PhotmoTarget(self.targetPath)
                except Exception:
                    print("Couldn't make target")
                    self.targetPath = None
                    continue
        
                #need to notify two machines of iterations : 9002, 9004
                #send model : 9003 - orange box
          
                dictionary = PhotmoDictionary(self.DICT_PATH, params=self.dictParams)
                analysis = PhotmoAnalysis(target, dictionary, params=self.analysisParams)
                analysis.start()
        
             #reset the target path to None, in order to wait for next taeget
            self.targetPath = None
        
    
            time.sleep(0.01)
    
def main(config):
    P = PhotmoListener(config)
    P.listen()


if __name__ == '__main__':
    
    path = sys.argv[1]
    config_file = open(path)
    config = json.load(config_file)
    main(config)

    
    

    
    


    

    
