#!/usr/bin/env python

import photmo
import liblo
import sys
import time

def handleSeverMessage(path, args):
    global TARG_PATH
    TARG_PATH = args[0]

DICT_PATH = './testData/portrait'
PORT = 9001

try:
    SERVER = liblo.Server(PORT)
except liblo.ServerError, err:
    print str(err)
    sys.exit()

TARG_PATH = None
SERVER.add_method('/target_path', 's', handleSeverMessage)

while True:
    #TODO: this will actually queue signals received, decide if that's the intended behaviour
    SERVER.recv(1)
    
    if TARG_PATH:
        print(TARG_PATH)
        
        target = photmo.PhotmoTarget(TARG_PATH)
        dictParams = {'num_images' : 3}
        
        dictionary = photmo.PhotmoDictionary(DICT_PATH, params=dictParams)
        
        params = {'num_iter' : 5,
                    'output_path' : './testData/output',
                    'model_output' : {'host' : '127.0.0.1', 'port' : 9002},
                    'iter_output' : {'host' : '127.0.0.1', 'port' : 9003}}
        
        analysis = photmo.PhotmoAnalysis(target, dictionary, params=params)
        analysis.start()
        
        #reset the target path to None, in order to wait for next taeget
        TARG_PATH = None
        
    
    time.sleep(0.01)