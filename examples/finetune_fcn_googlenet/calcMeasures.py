import numpy as np
from PIL import Image
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples/fcn
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
#import matplotlib.pyplot as plt
from helper import evalExp, pxEval_maximizeFMeasure
#from sklearn.metrics import confusion_matrix

def calcClassMeasures(gt, prob, validArea, thresh):
#    print prob.max()
#    plt.subplot(2,1,1)
#    plt.imshow(gt)
#    plt.subplot(2,1,2)
#    plt.imshow(prob)
    return evalExp(gt, prob, thresh, validMap = None, validArea=validArea)

# color code for ground truth images
label_colors = [(64,128,64),(192,0,128),(0,128,192),(0,128,64),(128,0,0),(64,0,128),(64,0,192),(192,128,64),(192,192,128),(64,64,128),(128,0,192),(192,0,64),(128,128,64),(192,0,192),(128,64,64),(64,192,128),(64,64,0),(128,64,128),(128,128,192),(0,0,192),(192,128,128),(128,128,128),(64,128,192),(0,0,64),(0,64,64),(192,64,128),(128,128,0),(192,128,192),(64,0,64),(192,192,0),(0,0,0),(64,192,0)]
label_class = [255,255,0,255,1,2,255,255,3,4,255,255,255,255,255,255,5,6,255,7,8,9,255,255,255,255,10,255,255,255,255,255]
label_name = ['Bicyclist','Building','Car\t','Column_Pole','Fence','Pedestrian','Road','Sidewalk','SignSymbol','Sky\t','Tree', 'Global']

#thresh = np.array(range(0,256))*47.0/255.0  # check your outputs to specify the threshold range
thresh = np.array([0.5])  # check your outputs to specify the threshold range

f = open('camvid_list_test.txt','r')
inputs = f.read().splitlines()
f.close()

f = open('camvid_list_test_gt.txt','r')
inputs_gt = f.read().splitlines()
f.close()

#net = caffe.Net('D:/CIW15_Experimente/models/bvlc_googlenet/fcn_8stride/fcn-deploy_8stride_early.prototxt', 'D:/CIW15_Experimente/4_early_8S_16S/snapshot/fcn-googlenet-8stride_early_iter_14000.caffemodel', caffe.TEST)
#net = caffe.Net('D:/CIW15_Experimente/models/bvlc_googlenet/fcn_8stride/fcn-deploy_8stride_late.prototxt', 'D:/CIW15_Experimente/4_late_8S_16S/snapshot/fcn-googlenet-8stride_late_iter_14000.caffemodel', caffe.TEST)
#net = caffe.Net('D:/CIW15_Experimente/models/bvlc_googlenet/fcn_8stride/fcn-deploy_8stride_early.prototxt', 'D:/CIW15_Experimente/2_early_8s_IN/snapshot/fcn-googlenet-8stride_early_iter_42000.caffemodel', caffe.TEST)
net = caffe.Net('D:/CIW15_Experimente/models/bvlc_googlenet/fcn_8stride/fcn-deploy_8stride_late.prototxt', 'D:/CIW15_Experimente/2_late_8s_IN/snapshot/fcn-googlenet-8stride_late_iter_42000.caffemodel', caffe.TEST)

# shape for input (data blob is N x C x H x W), set data

totalFP = np.zeros( [12, len(thresh)] )
totalFN = np.zeros( [12, len(thresh)] )
totalPosNum = np.zeros([12,1])
totalNegNum = np.zeros([12,1])

for (idx_, in_) in enumerate(inputs):
#if 1==1:
#    idx_=0
#    in_=inputs[idx_]
    gt_in_ = inputs_gt[idx_]
    im = Image.open(in_) # load image
    im = np.array(im, dtype=np.float32)
    im = im[:,:,::-1]
    im -= np.array((104.00698793,116.66876762,122.67891434))
    im = im.transpose((2,0,1))
    net.blobs['data'].reshape(1, *im.shape)
    net.blobs['data'].data[...] = im
    assert isinstance(net, object)
    net.forward()
#    out = net.blobs['score'].data[0].argmax(axis=0)
    im = Image.open(gt_in_) # load image
    im = np.array(im) # convert to nparray you need
    # convert to one dimensional ground truth labels
    tmp = np.uint8(np.zeros((im.shape[0],im.shape[1])))
    prob_max = 0
    for i in range(0,len(label_colors)):
        tmp[:,:] = tmp[:,:] + label_class[i]*np.prod(np.equal(im,label_colors[i]),2)
#        if label_class[i]!=255:
#            prob_max = np.max([prob_max, net.blobs['score'].data[0][label_class[i],:,:].max()])
#    print prob_max
    for j in [i for i in range(0,len(label_class)) if label_class[i]!=255]:
        FN, FP, posNum, negNum = calcClassMeasures(np.prod(np.equal(im,label_colors[j]),2),(net.blobs['score'].data[0].argmax(axis=0)==label_class[j]).astype(np.float32),tmp!=255,thresh)
#        FN, FP, posNum, negNum = calcClassMeasures(np.prod(np.equal(im,label_colors[j]),2),net.blobs['score'].data[0][label_class[j],:,:],tmp!=255,thresh)
        totalFP[label_class[j],:] += FP
        totalFN[label_class[j],:] += FN
        totalPosNum[label_class[j]] += posNum
        totalNegNum[label_class[j]] += negNum
        
    # calculate measures globally
    FN, FP, posNum, negNum = calcClassMeasures(np.ones([im.shape[0],im.shape[1]]),(net.blobs['score'].data[0].argmax(axis=0)==tmp).astype(np.float32),tmp!=255,thresh)
    totalFP[11,:] += FP
    totalFN[11,:] += FN
    totalPosNum[11] += posNum
    totalNegNum[11] += negNum
#groundtruth = np.uint8(np.zeros((im.shape[0],im.shape[1])))
#for i in range(0,len(label_colors)):
#    groundtruth[:,:] = groundtruth[:,:] + label_class[i]*np.prod(np.equal(im,label_colors[i]),2)
#
#prediction=net.blobs['score'].data[0].argmax(axis=0)
#prediction = prediction.reshape([np.prod(prediction.shape),1])
#groundtruth = groundtruth.reshape([np.prod(groundtruth.shape),1])
#prediction = prediction[groundtruth!=255]
#groundtruth = groundtruth[groundtruth!=255]
#confmat = confusion_matrix(groundtruth,prediction)
#confmat = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]
# # accuracy as in Segmentation and Recognition Using Structure from Motion Point Clouds, ECCV 2008:
# # per-class accuracies (the normalized diagonal of the pixel-wise confusion matrix)
# is the same as REC_wp

for i in range(0,12):
    out = pxEval_maximizeFMeasure(totalPosNum[i,:], totalNegNum[i,:], totalFN[i,:], totalFP[i,:], thresh)
    print 'Klasse: ' + label_name[i] +'\t',
    for property in ['REC_wp', 'MaxF','iou']:
        print '%s: %4.2f \t' %(property, out[property]*100,),
    print ' '

