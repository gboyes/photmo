import os
import cv
import sys
import numpy as np
import scipy.linalg as linalg

def main(indir, outdir, newheight, newwidth):

    for f in os.listdir(indir):

        os.system('convert %s -resize %ix%i\! %s%ix%i.png'%(indir + '/' + f, newheight, newwidth, outdir + '/' + os.path.splitext(f)[0], newheight, newwidth ))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], eval(sys.argv[3]), eval(sys.argv[4]))
