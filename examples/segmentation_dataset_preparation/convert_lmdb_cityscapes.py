# -*- coding: utf-8 -*-
"""
Created on Thu Mar 03 11:03:16 2016

@author: oeljeklaus
"""

caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples/fcn
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import lmdb
from PIL import Image
import numpy as np

scale_factor=0.55 #  1126x563
#scale_factor=0.525 #  1075x537
output_dir = 'F:\\cityscapes_train_55'

execfile("D:/cityscapes/scripts/helpers/labels.py")



f = open('D:\\cityscapes\\listings\\leftImg8bit_train.txt','r')
inputs = f.read().splitlines()
ridx = np.arange(len(inputs))
np.random.shuffle(ridx)
inputs = [inputs[i] for i in ridx]
#inputs = inputs[1:10]
f.close()


# load first image to determine database size
im = Image.open(inputs[0]) # load image
im = im.resize((int(im.size[0]*scale_factor),int(im.size[1]*scale_factor)),Image.ANTIALIAS) # downsize for reduced memory usage
im = np.array(im) # convert to nparray you need
im = im[:,:,::-1]
im = im.transpose((2,0,1))
im_dat = caffe.io.array_to_datum(im)

in_db = lmdb.open(output_dir+'\cityscapes_55_train-lmdb', map_size=int(im_dat.ByteSize()*(len(inputs)+5))) # im_dat.ByteSize()=1901828
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs):
        if in_idx % round(len(inputs)/100) == 0:
            sys.stdout.write('#')
        # load image:
        # - as np.uint8 {0, ..., 255}
        # - in BGR (switch from RGB)
        # - in Channel x Height x Width order (switch from H x W x C)
        im = Image.open(in_) # load image
        im = im.resize((int(im.size[0]*scale_factor),int(im.size[1]*scale_factor)),Image.ANTIALIAS) # downsize for reduced memory usage
        im = np.array(im) # convert to nparray you need
        im = im[:,:,::-1]
        im = im.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(im)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
in_db.close()

print ' cityscapes train done'

f = open('D:\\cityscapes\\listings\\gtFine_labelIds_train.txt','r')
inputs = f.read().splitlines()
inputs = [inputs[i] for i in ridx]
#inputs = inputs[1:10]
f.close()

f = open('D:\\cityscapes\\listings\\scalar_labels_train.txt','r')
scalar_label = f.read().splitlines()
scalar_label = [scalar_label[i] for i in ridx]
#inputs = inputs[1:10]
f.close()

# load first image to determine database size
im = Image.open(inputs[0]) # load image
im = im.resize((int(im.size[0]*scale_factor),int(im.size[1]*scale_factor)),Image.NEAREST) # downsize for reduced memory usage
im = np.array(im) # convert to nparray you need
# convert to one dimensional ground truth labels
tmp = np.uint8(np.zeros((im.shape[0],im.shape[1],1)))
# - in Channel x Height x Width order (switch from H x W x C)
tmp = tmp.transpose((2,0,1))
im_dat = caffe.io.array_to_datum(tmp)

in_db = lmdb.open(output_dir+'\cityscapes_55_train-gt-lmdb', map_size=int(im_dat.ByteSize()*(len(inputs)+5)))
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs):
        if in_idx % round(len(inputs)/100) == 0:
            sys.stdout.write('#')
        im = Image.open(in_) # load image
        im = im.resize((int(im.size[0]*scale_factor),int(im.size[1]*scale_factor)),Image.NEAREST) # downsize for reduced memory usage
        im = np.uint8(np.array(im)) # convert to nparray you need
        # convert to one dimensional ground truth labels
        tmp = np.uint8(np.zeros((im.shape[0],im.shape[1],1)))
        tmp[:,:,0] = im
        # replace id with trainId
        for i in range(0,len(labels)):
            tmp[tmp[:,:,0]==i] = labels[i].trainId
        # - in Channel x Height x Width order (switch from H x W x C)
        tmp = tmp.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(tmp,int(scalar_label[in_idx]))
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
in_db.close()

np.save(output_dir+'\ridx_train.npy',ridx)

print ' cityscapes train gt done'


f = open('D:\\cityscapes\\listings\\leftImg8bit_val.txt','r')
inputs = f.read().splitlines()
ridx = np.arange(len(inputs))
np.random.shuffle(ridx)
inputs = [inputs[i] for i in ridx]
#inputs = inputs[1:10]
f.close()

# load first image to determine database size
im = Image.open(inputs[0]) # load image
im = im.resize((int(im.size[0]*scale_factor),int(im.size[1]*scale_factor)),Image.ANTIALIAS) # downsize for reduced memory usage
im = np.array(im) # convert to nparray you need
im = im[:,:,::-1]
im = im.transpose((2,0,1))
im_dat = caffe.io.array_to_datum(im)

in_db = lmdb.open(output_dir+'\cityscapes_55_val-lmdb', map_size=int(im_dat.ByteSize()*(len(inputs)+5)))
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs):
        if in_idx % round(len(inputs)/100) == 0:
            sys.stdout.write('#')
        # load image:
        # - as np.uint8 {0, ..., 255}
        # - in BGR (switch from RGB)
        # - in Channel x Height x Width order (switch from H x W x C)
        im = Image.open(in_) # load image
        im = im.resize((int(im.size[0]*scale_factor),int(im.size[1]*scale_factor)),Image.ANTIALIAS) # downsize for reduced memory usage
        im = np.array(im) # convert to nparray you need
        im = im[:,:,::-1]
        im = im.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(im)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
in_db.close()

print ' cityscapes val done'

f = open('D:\\cityscapes\\listings\\gtFine_labelIds_val.txt','r')
inputs = f.read().splitlines()
inputs = [inputs[i] for i in ridx]
#inputs = inputs[1:10]
f.close()

f = open('D:\\cityscapes\\listings\\scalar_labels_val.txt','r')
scalar_label = f.read().splitlines()
scalar_label = [scalar_label[i] for i in ridx]
#inputs = inputs[1:10]
f.close()


# load first image to determine database size
im = Image.open(inputs[0]) # load image
im = im.resize((int(im.size[0]*scale_factor),int(im.size[1]*scale_factor)),Image.NEAREST) # downsize for reduced memory usage
im = np.array(im) # convert to nparray you need
# convert to one dimensional ground truth labels
tmp = np.uint8(np.zeros((im.shape[0],im.shape[1],1)))
# - in Channel x Height x Width order (switch from H x W x C)
tmp = tmp.transpose((2,0,1))
im_dat = caffe.io.array_to_datum(tmp)

in_db = lmdb.open(output_dir+'\cityscapes_55_val-gt-lmdb', map_size=int(im_dat.ByteSize()*(len(inputs)+5)))
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs):
        if in_idx % round(len(inputs)/100) == 0:
            sys.stdout.write('#')
        im = Image.open(in_) # load image
        im = im.resize((int(im.size[0]*scale_factor),int(im.size[1]*scale_factor)),Image.NEAREST) # downsize for reduced memory usage
        im = np.uint8(np.array(im)) # convert to nparray you need
        # convert to one dimensional ground truth labels
        tmp = np.uint8(np.zeros((im.shape[0],im.shape[1],1)))
        tmp[:,:,0] = im
        # replace id with trainId
        for i in range(0,len(labels)):
            tmp[tmp[:,:,0]==i] = labels[i].trainId
        # - in Channel x Height x Width order (switch from H x W x C)
        tmp = tmp.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(tmp,int(scalar_label[in_idx]))
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
in_db.close()

np.save(output_dir+'\ridx_val.npy',ridx)

print ' cityscapes val gt done'
