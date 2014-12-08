import numpy as np
import scipy.linalg
import cv
import pydbm.atom

grgb =  [49, 203, 121]
rrgb =  [255, 153, 51]


k = (100, 100, 3)
w = pydbm.atom.GaborGen()

#g = np.abs(np.random.randn(k[0], k[1], k[2])) * 255.0
#r = np.abs(np.random.randn(k[0], k[1], k[2])) * 255.0

g = cv.LoadImageM('/Users/geb/Desktop/avgfemale/avgfemale/avgfemale.png')
g = np.asarray(g).astype(float)

r = cv.LoadImageM('/Users/geb/Desktop/avgmale.png')
r = np.asarray(r).astype(float)

k = np.shape(g)
'''
for i in range(3):
    g[:, :, i] *= grgb[i]
    r[:, :, i] *= rrgb[i] 
'''
mat = cv.fromarray(g)
cv.SaveImage("/Users/geb/Desktop/g.png", mat)
mat = cv.fromarray(r)
cv.SaveImage("/Users/geb/Desktop/r.png", mat)

g_ = g / 255.0
r_ = r / 255.0

co = []
t = np.zeros(k)
count = 0
while count < 100:
    for i in range(3):
        #for j in range(np.shape(g_)[1]):
        grain = r_[:, :, i] / scipy.linalg.norm(r_[:, :, i])
        #grain *= w.window(len(grain), alpha=0.5)
        #c = np.abs(np.inner(g_[:, :, i], grain))
        c = np.abs(np.tensordot(g_[:, :, i], grain))
        #c = g_[:, :, i] / grain
        t[:, :, i] += grain * c
        g_[:, :, i] -= grain * c
    count += 1

mat = cv.fromarray(t * 255.0)
cv.SaveImage("/Users/geb/Desktop/t.png", mat)
    
