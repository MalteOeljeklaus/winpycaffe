

caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples/fcn
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import lmdb
from PIL import Image
import numpy as np

ridx = np.arange(367)
np.random.shuffle(ridx)

f = open('camvid_list_train.txt','r')
inputs = f.read().splitlines()
inputs = [inputs[i] for i in ridx]
f.close()

in_db = lmdb.open('D:\camvid_train-lmdb', map_size=int(58720256*3.7))
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

print 'camvid train done'

# color code for ground truth images
label_colors = [(64,128,64),(192,0,128),(0,128,192),(0,128,64),(128,0,0),(64,0,128),(64,0,192),(192,128,64),(192,192,128),(64,64,128),(128,0,192),(192,0,64),(128,128,64),(192,0,192),(128,64,64),(64,192,128),(64,64,0),(128,64,128),(128,128,192),(0,0,192),(192,128,128),(128,128,128),(64,128,192),(0,0,64),(0,64,64),(192,64,128),(128,128,0),(192,128,192),(64,0,64),(192,192,0),(0,0,0),(64,192,0)]
label_class = [255,255,0,255,1,2,255,255,3,4,255,255,255,255,255,255,5,6,255,7,8,9,255,255,255,255,10,255,255,255,255,255]

f = open('camvid_list_train_gt.txt','r')
inputs = f.read().splitlines()
inputs = [inputs[i] for i in ridx]
f.close()

in_db = lmdb.open('D:\camvid_train-gt-lmdb', map_size=int(20971520*3.7))
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs):
        im = Image.open(in_) # load image
        im = im.resize((int(im.size[0]*0.5),int(im.size[1]*0.5)),Image.NEAREST) # downsize for reduced memory usage
        im = np.array(im) # convert to nparray you need
        # convert to one dimensional ground truth labels
        tmp = np.uint8(np.zeros((im.shape[0],im.shape[1],1)))
        for i in range(0,len(label_colors)):
            tmp[:,:,0] = tmp[:,:,0] + label_class[i]*np.prod(np.equal(im,label_colors[i]),2)

        # - in Channel x Height x Width order (switch from H x W x C)
        tmp = tmp.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(tmp)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
in_db.close()

print 'camvid train gt done'

ridx = np.arange(233)
np.random.shuffle(ridx)

f = open('camvid_list_test.txt','r')
inputs = f.read().splitlines()
inputs = [inputs[i] for i in ridx]
f.close()

in_db = lmdb.open('D:\camvid_test-lmdb', map_size=int(58720256*2.3))
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

print 'camvid test done'

f = open('camvid_list_test_gt.txt','r')
inputs = f.read().splitlines()
inputs = [inputs[i] for i in ridx]
f.close()

in_db = lmdb.open('D:\camvid_test-gt-lmdb', map_size=int(20971520*2.3))
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs):
        im = Image.open(in_) # load image
        im = im.resize((int(im.size[0]*0.5),int(im.size[1]*0.5)),Image.NEAREST) # downsize for reduced memory usage
        im = np.array(im) # convert to nparray you need
        # convert to one dimensional ground truth labels
        tmp = np.uint8(np.zeros((im.shape[0],im.shape[1],1)))
        for i in range(0,len(label_colors)):
            tmp[:,:,0] = tmp[:,:,0] + label_class[i]*np.prod(np.equal(im,label_colors[i]),2)

        # - in Channel x Height x Width order (switch from H x W x C)
        tmp = tmp.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(tmp)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
in_db.close()

print 'camvid test gt done'
