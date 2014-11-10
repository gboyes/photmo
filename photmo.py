#!/usr/bin/env python

import os
import cv
import cv2
import numpy as np
import scipy.linalg as linalg
import datetime
import liblo
import time
from multiprocessing import Process

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
    def __init__(self, path_to_file, scalar=1):
        
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
                        a = PhotmoAtom(path_to_dir + '/' + f, scalar=s)
                        self.atoms.append(a)
                        count +=1
                    except Exception:
                        print('Error making atom')
                        continue
            else:
                break
                
    
class PhotmoAnalysis():
    def __init__(self, target, dictionary, params=None):
        
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
        
        #mat = np.random.randn(self.target.height, self.target.width, self.target.planes) * 255
        count = 0
        writecount = 0
        winds = set([])
        modelParams = {}
        
        while count < self.kIterations:
            
            newDict = {}
            
            #random onsets
            yo = self.roundnum(np.random.randint(self.target.height), 32)
            xo = self.roundnum(np.random.randint(self.target.width), 24)
            
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
            #nilnil = np.zeros(np.shape(self.target.image))
            for p in range(0, self.target.planes-1):
                    
                self.residual[yo:yb, xo:xb, p] -= atom.image[0:ayb, 0:axb, p] * c[p]
                self.model[yo:yb, xo:xb, p] +=  atom.image[0:ayb, 0:axb, p] * c[p]
                #nilnil[yo:yb, xo:xb, p] += atom.image[0:ayb, 0:axb, p] * c[p]
            
            self.residual[yo:yb, xo:xb, 3] -= atom.image[0:ayb, 0:axb, 3]
            self.model[yo:yb, xo:xb, 3] +=  atom.image[0:ayb, 0:axb, 3]
            
            #TOTEST: file writing is a bottleneck, try spawning process for this, need queue
            #p = Process(target=cv2.imwrite, args=(path, self.model * 255))
            #p.start()
            
            #ugly hacks
            if np.log2(count) % 1 == 0:
                
                lastpw2 = np.log2(count)
                nextpw2 = lastpw2 + 1
                pwdist = 2**nextpw2 - 2**lastpw2
                kdist = pwdist / 7
                winds = set(np.round(np.arange(2**lastpw2, 2**nextpw2, kdist)))
                
                filename = "%s_%i.png"%(self.timestamp.strftime("%Y-%m-%d_%H_%M"), writecount)
                path = "%s/%s_%i.png"%(self.outputDirectory, self.timestamp.strftime("%Y-%m-%d_%H_%M"), writecount)
                cv2.imwrite(path, self.model* 255)
            
                if iterSocket:
                    liblo.send(iterSocket, filename)
                    #liblo.send(iterSocket, path)
                    
                writecount += 1
                    
            elif count in winds:
                filename = "%s_%i.png"%(self.timestamp.strftime("%Y-%m-%d_%H_%M"), writecount)
                path = "%s/%s_%i.png"%(self.outputDirectory, self.timestamp.strftime("%Y-%m-%d_%H_%M"), writecount)
                cv2.imwrite(path, self.model* 255)
            
                if iterSocket:
                    liblo.send(iterSocket, filename)
                    #liblo.send(iterSocket, path)
                    
                writecount += 1
            
            
            count += 1
            print(count)
            
            
        #mat = np.random.randn(self.target.height, self.target.width, self.target.planes) * 255
        filename = "%s_MODEL.png"%self.timestamp.strftime("%Y-%m-%d_%H_%M")
        path = "%s/%s_MODEL.png"%(self.outputDirectory, self.timestamp.strftime("%Y-%m-%d_%H_%M"))
        
        cv2.imwrite(path, self.model * 255)
        if modelSocket:
            liblo.send(modelSocket, filename)
            #liblo.send(modelSocket, path)
    
    

    

    
