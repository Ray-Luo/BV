import os
import torch as t
from utils.config import opt
from data.dataset import Dataset, TestDataset
from torch.utils import data as data_
from tqdm import tqdm
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at
from utils.eval_tool import calc_detection_voc_prec_rec
import numpy as np

def eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
        gt_difficults=None,
        iou_thresh=0.5):

    prec, rec = calc_detection_voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        iou_thresh)

    return prec, rec

def eval(dataloader, faster_rcnn, test_num=10000):
    total = 0
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    # for target,predict in zip(gt_labels,pred_labels):
    #     print('target:',target,'predict',predict)
        # total += len(i)
    # print('labels:',len(total))

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults)

    return result

faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()
trainer.load('/home/kdd/Documents/simple-faster-rcnn-pytorch/checkpoints/fasterrcnn_11101036_0.5348237306784394')
opt.caffe_pretrain=True

# get dataloader
testset = TestDataset(opt)
test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
result = eval(test_dataloader,faster_rcnn)
prec, recall = result[0], result[1]
