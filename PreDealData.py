import numpy as np
import cv2 as cv
import scipy.misc as misc
import os
from read_MITSceneParsingData import create_image_lists

classes = ['background','aeroplane','bicycle','bird','boat',
           'bottle','bus','car','cat','chair','cow','diningtable',
           'dog','horse','motorbike','person','potted plant',
           'sheep','sofa','train','tv/monitor']

# RGB color for each class
colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            [64,128,0],[192,128,0],[64,0,128],[192,0,128],
            [64,128,128],[192,128,128],[0,64,0],[128,64,0],
            [0,192,0],[128,192,0],[0,64,128]]

cm2lbl = np.zeros(256**3) # 每个像素点有 0 ~ 255 的选择，RGB 三个通道
for i,cm in enumerate(colormap):
    cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i # 建立索引

def image2label(im):# 将3通道的label彩图变成单通道的图，图上每个像素点的值代表属于的class
    data = np.array(im, dtype='int32')
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]

    return np.array(cm2lbl[idx], dtype='int64') # 根据索引得到 label 矩阵

def transform():
    """
    transform the label data.3 channels to one channel
    :return:
    """
    files = os.listdir("/Users/huangxiao/imgData/VOCdevkit/VOC2012/SegmentationClass")
    for file in files[:10]:
        name = os.path.splitext(file)[0]
        #我是真他妈的操了，cv.imread默认通道是BGR,我说他妈的怎么有些图片没有映射成功。去你妈的opencv吧
        label_im = misc.imread('/Users/huangxiao/imgData/VOCdevkit/VOC2012/SegmentationClass/'+name+'.png')
        label = image2label(label_im)
        cv.imwrite('/Users/huangxiao/imgData/VOCtest/annotations/training/'+name+'.png', label)

transform()
create_image_lists("/Users/huangxiao/imgData/VOCtest/")