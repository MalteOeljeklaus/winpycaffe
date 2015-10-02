

caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples/fcn
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import lmdb
from PIL import Image
import numpy as np

ridx = np.arange(101)
np.random.shuffle(ridx)

f = open('camseq01_list.txt','r')
inputs = f.read().splitlines()
inputs = [inputs[i] for i in ridx]
f.close()

in_db = lmdb.open('camseq01-lmdb', map_size=int(58720256))
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs):
        # load image:
        # - as np.uint8 {0, ..., 255}
        # - in BGR (switch from RGB)
        # - in Channel x Height x Width order (switch from H x W x C)
        im = Image.open(in_) # load image
        im = im.resize((int(im.size[0]*0.5),int(im.size[1]*0.5)),Image.ANTIALIAS) # downsize for reduced memory usage
        im = np.array(im) # convert to nparray you need
        im = im[:,:,::-1]
        im = im.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(im)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
in_db.close()

print 'camseq01 done'

# color code for ground truth images
label_colors = [(64,128,64),(192,0,128),(0,128,192),(0,128,64),(128,0,0),(64,0,128),(64,0,192),(192,128,64),(192,192,128),(64,64,128),(128,0,192),(192,0,64),(128,128,64),(192,0,192),(128,64,64),(64,192,128),(64,64,0),(128,64,128),(128,128,192),(0,0,192),(192,128,128),(128,128,128),(64,128,192),(0,0,64),(0,64,64),(192,64,128),(128,128,0),(192,128,192),(64,0,64),(192,192,0),(0,0,0),(64,192,0)]

f = open('camseq01_list_gt.txt','r')
inputs = f.read().splitlines()
inputs = [inputs[i] for i in ridx]
f.close()

in_db = lmdb.open('camseq01_gt-lmdb', map_size=int(20971520))
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs):
        im = Image.open(in_) # load image
        im = im.resize((int(im.size[0]*0.5),int(im.size[1]*0.5)),Image.NEAREST) # downsize for reduced memory usage
        im = np.array(im) # convert to nparray you need
        # convert to one dimensional ground truth labels
        tmp = np.uint8(np.zeros((im.shape[0],im.shape[1],1)))
        for i in range(0,len(label_colors)):
            tmp[:,:,0] = tmp[:,:,0] + i*np.prod(np.equal(im,label_colors[i]),2)

        # - in Channel x Height x Width order (switch from H x W x C)
        tmp = tmp.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(tmp)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
in_db.close()

print 'camseq01 gt done'
