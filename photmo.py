#!/usr/bin/env python

import os
import cv
import numpy as np
import scipy.linalg as linalg
import datetime
import liblo

class PhotmoTarget():
    def __init__(self, path_to_file):
        try:
            i = cv.LoadImageM(path_to_file)
        except IOError:
            print("No file named " + path_to_file)
            return None
        
        self.image = np.asarray(i).astype(float) / 255.0
        self.height, self.width, self.planes = np.shape(self.image)
    
class PhotmoDictionary():
    def __init__(self, path_to_dir, kfiles=None):
        if kfiles:
            self.kInFiles = kfiles
        else:
            self.kInFiles = 100
        
        count = 0
        while count < self.kInFiles:
            for f in os.listdir(path_to_dir):
                if f[0] == '.':
                    continue
                try:
                    i = cv.LoadImageM(path_to_dir + '/' + f)
                except IOError:
                    print("No file named " + path_to_dir)
                    
                count +=1
                
    
class PhotmoAnalysis():
    def __init__(self, target, dictionary, params=None):
        
        #make a timestamp at the beginning of the procedure, use for string formatting later #e.g.self.timstamp.strftime("%Y-%m-%d_%H_%M")
        self.timestamp = datetime.datetime.now()
        self.target = target
        self.dictionary = dictionary
        self.model = np.zeros(np.shape(self.target.image))
        self.residual = self.target.image.copy()
        
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
                
        
    def start(self):
        
        print('Analysis started.... ')
        
        iterSocket = None
        if self.iterAddr:
            iterSocket = liblo.Address(self.iterAddr['host'], self.iterAddr['port'] )
            print('made socket')
        
        modelSocket = None
        if self.modelAddr:
            modelSocket = liblo.Address(self.modelAddr['host'], self.modelAddr['port'] )
        
        count = 0
        while count < 100:
            mat = cv.fromarray(np.random.randn(self.target.height, self.target.width, self.target.planes) * 255)
            path = "%s/%s_%i.png"%(self.outputDirectory, self.timestamp.strftime("%Y-%m-%d_%H_%M"), count)
            cv.SaveImage(path, mat)
            
            if iterSocket:
                liblo.send(iterSocket, path)
            
            count += 1
            
        mat = cv.fromarray(np.random.randn(self.target.height, self.target.width, self.target.planes) * 255)
        path = "%s/%s_MODEL.png"%(self.outputDirectory, self.timestamp.strftime("%Y-%m-%d_%H_%M"))
        
        cv.SaveImage(path, mat)
        if modelSocket:
            liblo.send(modelSocket, path)
    
    

    

    
