#!/usr/bin/env python

import os
import cv
import numpy as np
import scipy.linalg as linalg
import datetime
import liblo
import time
from multiprocessing import Process

class PhotmoTarget():
    def __init__(self, path_to_file):
        try:
            i = cv.LoadImageM(path_to_file)
        except IOError:
            print("No file named " + path_to_file)
            return None
        
        self.image = np.asarray(i).astype(float) / 255.0
        self.height, self.width, self.planes = np.shape(self.image)
    
class PhotmoAtom():
    def __init__(self, path_to_file):
        try:
             i = cv.LoadImageM(path_to_file)
             
        except IOError:
            print("No valid file named " + path_to_file)
            return None
       

class PhotmoDictionary():
    def __init__(self, path_to_dir, params=None):
        if params:
            try:
                self.kImages = params['num_images']
                
            except KeyError:
                self.kImages = 100
                
        #set the default params         
        else:
            self.kImages = 100
            
        self.atoms = []
        count = 0
        for f in os.listdir(path_to_dir):
            if count < self.kImages:
                a = PhotmoAtom(path_to_dir + '/' + f)
                if a:
                    self.atoms.append(a)
                    count +=1
            else:
                break
                
    
class PhotmoAnalysis():
    def __init__(self, target, dictionary, params=None):
        
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
                
            try:
                self.kIterations = params['num_iter']
            except KeyError:
                self.kIterations = 100
        
        #default config
        else:
            self.kIterations = 100
            
        
        
    def start(self):
        
        print('Analysis started.... ')
        
        iterSocket = None
        if self.iterAddr:
            iterSocket = liblo.Address(self.iterAddr['host'], self.iterAddr['port'] )
        
        modelSocket = None
        if self.modelAddr:
            modelSocket = liblo.Address(self.modelAddr['host'], self.modelAddr['port'] )
        
        mat = cv.fromarray(np.random.randn(self.target.height, self.target.width, self.target.planes) * 255)
        count = 0
        while count < self.kIterations:
            
            path = "%s/%s_%i.png"%(self.outputDirectory, self.timestamp.strftime("%Y-%m-%d_%H_%M"), count)
            
            #TOTEST: file writing is a bottleneck, try spawning process for this
            p = Process(target=cv.SaveImage, args=(path, mat))
            p.start()
            
            if iterSocket:
                liblo.send(iterSocket, path, count)
            
            count += 1
            print(count)
            
            
        mat = cv.fromarray(np.random.randn(self.target.height, self.target.width, self.target.planes) * 255)
        path = "%s/%s_MODEL.png"%(self.outputDirectory, self.timestamp.strftime("%Y-%m-%d_%H_%M"))
        
        cv.SaveImage(path, mat)
        if modelSocket:
            liblo.send(modelSocket, path)
    
    

    

    
