'''
photmo
Copyright (C) 2014-2015 Graham Boyes

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
'''

#!/usr/bin/env python

import os
import sys
import json
import datetime
import time

import cv2
import numpy as np

class PhotmoListener():
    
    '''Manages the input and analysis queue'''
    
    def __init__(self, config):
        
        self.DICT_PATH = config["dictionaryPath"]
        
        #the server
        self.serverPort = config["serverPort"]
        self.serverPath = config["serverPath"]
        
        #busy notifications 
        self.busyHost = config["busy"]["host"]
        self.busyPort = config["busy"]["port"]
        self.busyPath = config["busy"]["path"]
        
        self.busyHost2 = config["busy2"]["host"]
        self.busyPort2 = config["busy2"]["port"]
        self.busyPath2 = config["busy2"]["path"]
        
        self.busyAddress = liblo.Address(self.busyHost, self.busyPort)
        self.busyAddress2 = liblo.Address(self.busyHost2, self.busyPort2)
        
        self.dictParams = config["dictionaryParams"]
        self.analysisParams = config["analysisParams"]
        
        self.targetPath = None
        self.configureNetwork()
    
    
    def configureNetwork(self):
        
        '''Configure the OSC network'''
        
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
        
    def randomWalk(self):
        
        self.dictParams["windowed"] = 1 #always windowed
        
        k = len(self.dictParams["scalars"])
        
        self.dictParams["scalars"] = [np.random.randint(50, 120) / 1000.0 for i in range(0, 2)]
        
        q = min(self.dictParams["scalars"])
        
        #change the snap configs
        self.analysisParams["snapx"] = int(q * 480 * 0.35)
        self.analysisParams["snapy"] = int(q * 640 * 0.35)
        
        print(self.dictParams["scalars"])
    
    def listen(self):
        
        '''Listen for a notification to start the decomposition of an image'''
     
        while True:
    
            #This will set the targetPath
            self.oscServer.recv(1)
    
            if self.targetPath:
                
                #explicitly close the port to prevent intermediate queing
                self.oscServer.free()
                try:
                    liblo.send(self.busyAddress, self.busyPath, 1)
                    liblo.send(self.busyAddress2, self.busyPath2, 1)
                except IOError:
                    print("Host is down or doesn't exist.")
                
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
                
                try:
                    liblo.send(self.busyAddress, self.busyPath, 0)
                    liblo.send(self.busyAddress2, self.busyPath2, 0)
                except IOError:
                    print("Host is down or doesn't exist.")
                    
                print('Ready')
                
            time.sleep(0.1)
    
def main(config):
    P = PhotmoListener(config)
    P.listen()


if __name__ == '__main__':
    
    path = sys.argv[1]
    config_file = open(path)
    config = json.load(config_file)
    main(config)

    
    

    
    


    

    
