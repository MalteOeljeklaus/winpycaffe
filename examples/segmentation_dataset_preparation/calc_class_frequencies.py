# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 13:36:52 2016

@author: oeljeklaus
"""

from PIL import Image
import numpy as np
import sys

execfile("D:/cityscapes/gitrepo/scripts/helpers/labels.py")

cnt = np.uint64(np.zeros(34))

f = open('D:/cityscapes/listings/gtFine_labelIds_train_val.txt','r')
inputs = f.read().splitlines()
#inputs = inputs[1:200]
f.close()
print '\n'
for in_ in inputs:
    sys.stdout.write('#')
    im = Image.open(in_) # load image
    im = np.array(im) # convert to nparray you need
    tmp = np.unique(im.flat, return_counts=True)
    cnt[tmp[0]] = cnt[tmp[0]] + tmp[1]

print '\ncityscapes fine gt done'

for i in range(0,34):
    print id2label[i].name + ":\t %(cnt)d" % {"cnt": cnt[i]}

#cityscapes-train-val-3475
#unlabeled:	 1080804
#ego vehicle:	 337332780
#rectification border:	 101143770
#out of roi:	 109928150
#static:	 99402231
#dynamic:	 22268576
#ground:	 94306630
#road:	 2381680967
#sidewalk:	 385659445
#parking:	 43216528
#rail track:	 11877624
#building:	 1461641548
#wall:	 42917813
#fence:	 55975907
#guard rail:	 586040
#bridge:	 18172370
#tunnel:	 3362825
#pole:	 81355164
#polegroup:	 578047
#traffic light:	 13285481
#traffic sign:	 36546566
#vegetation:	 1038651996
#terrain:	 71574562
#sky:	 252744993
#person:	 79239848
#rider:	 9438758
#car:	 446059503
#truck:	 17532539
#bus:	 16553410
#caravan:	 2546786
#trailer:	 1501661
#train:	 13895603
#motorcycle:	 6178567
#bicycle:	 29365708



#CamVid-600
#Bicyclist:	 995897
#Building:	 93151827
#Car:       15482502
#ColumnPole:4389382
#Fence:	 4767882
#Pedestrian:2642624
#Road:	 113261960
#Sidewalk:	 24672611
#SignSymbol:523898
#Sky:	      70376403
#Tree:	 39209804
#Ignore:	 45245210

#CamVid-701
#Bicyclist:	 2542720
#Building:	 110302864
#Car:	 16496593
#ColumnPole:	 4782984
#Fence:	 6921061
#Pedestrian:	 3097211
#Road:	 132226796
#Sidewalk:	 30757190
#SignSymbol:	 574244
#Sky:	 76801167
#Tree:	 50564119
#Ignore:	 49464251