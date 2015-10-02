# -*- coding: utf-8 -*-
# use tools/extra/parse_log.py to obtain csv from log

import matplotlib.pyplot as plt
import numpy as np
#from numpy import genfromtxt

train = np.genfromtxt('D:\\Stick\\1_early_32s_IN\\logs\\caffe.WS24.SILAB.log.INFO.20150916-182940.1584.train', delimiter=',')
test = np.genfromtxt('D:\\Stick\\1_early_32s_IN\\logs\\caffe.WS24.SILAB.log.INFO.20150916-182940.1584.test', delimiter=',')

plt.subplot(1,2,1)
plt.plot(train[:,0],train[:,3],test[:,0],test[:,3])
plt.ylim( 0, 5)

print('32s: best test loss at iteration ' + np.str(test[test[1:,3].argmin(0),0]))



train = np.genfromtxt('D:\\Stick\\1_early_8s_IN\\logs\\caffe.WS21.SILAB.log.INFO.20150916-180254.2920.train', delimiter=',')
test = np.genfromtxt('D:\\Stick\\1_early_8s_IN\\logs\\caffe.WS21.SILAB.log.INFO.20150916-180254.2920.test', delimiter=',')

plt.subplot(1,2,2)
plt.plot(train[:,0],train[:,3],test[:,0],test[:,3])
plt.ylim( 0, 5)

print('8s: best test loss at iteration ' + np.str(test[test[1:,3].argmin(0),0]))



