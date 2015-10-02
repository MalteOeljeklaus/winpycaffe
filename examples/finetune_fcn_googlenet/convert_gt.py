import numpy as np
from PIL import Image
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples/fcn
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import matplotlib.pyplot as plt

# color code for ground truth images
label_colors = [(64,128,64),(192,0,128),(0,128,192),(0,128,64),(128,0,0),(64,0,128),(64,0,192),(192,128,64),(192,192,128),(64,64,128),(128,0,192),(192,0,64),(128,128,64),(192,0,192),(128,64,64),(64,192,128),(64,64,0),(128,64,128),(128,128,192),(0,0,192),(192,128,128),(128,128,128),(64,128,192),(0,0,64),(0,64,64),(192,64,128),(128,128,0),(192,128,192),(64,0,64),(192,192,0),(0,0,0),(64,192,0)]
label_class = [11,11,0,11,1,2,11,11,3,4,11,11,11,11,11,11,5,6,11,7,8,9,11,11,11,11,10,11,11,11,11,11]
label_name = ['Bicyclist','Building','Car\t','Column_Pole','Fence','Pedestrian','Road','Sidewalk','SignSymbol','Sky\t','Tree', 'Global']

#thresh = np.array(range(0,256))*47.0/255.0  # check your outputs to specify the threshold range
thresh = np.array([0.5])  # check your outputs to specify the threshold range


inputs_gt = ['D:/LabeledApproved_full/Seq05VD_f03360_L.png']

#net = caffe.Net('D:/CIW15_Experimente/models/bvlc_googlenet/fcn_8stride/fcn-deploy_8stride_late.prototxt', 'D:/CIW15_Experimente/2_late_8s_IN/snapshot/fcn-googlenet-8stride_late_iter_42000.caffemodel', caffe.TEST)

# shape for input (data blob is N x C x H x W), set data

for (idx_, in_) in enumerate(inputs_gt):
    gt_in_ = inputs_gt[idx_]
    im = Image.open(gt_in_) # load image
    im = np.array(im) # convert to nparray you need
    # convert to one dimensional ground truth labels
    tmp = np.uint8(np.zeros((im.shape[0],im.shape[1])))
    prob_max = 0
    for i in range(0,len(label_colors)):
        tmp[:,:] = tmp[:,:] + label_class[i]*np.prod(np.equal(im,label_colors[i]),2)

plt.imshow(tmp)