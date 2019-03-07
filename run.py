import copy
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision import models, transforms, utils
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch
import argparse
from FeatureExtractor import FeatureExtractor


def nms(dets, scores, thresh):
    x1 = dets[:, 0, 0]
    y1 = dets[:, 0, 1]
    x2 = dets[:, 1, 0]
    y2 = dets[:, 1, 1]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='robust template matching using CNN')
    parser.add_argument('image_path')
    parser.add_argument('template_path')
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--use_cython', action='store_true')
    args = parser.parse_args()

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])
    
    raw_image = cv2.imread(args.image_path)[..., ::-1]
    image = image_transform(raw_image.copy()).unsqueeze(0)

    raw_template = cv2.imread(args.template_path)[..., ::-1]
    template = image_transform(raw_template.copy()).unsqueeze(0)

    vgg_feature = models.vgg13(pretrained=True).features
    FE = FeatureExtractor(vgg_feature, use_cuda=args.use_cuda, padding=True)
    boxes, centers, scores = FE(
        template, image, threshold=None, use_cython=args.use_cython)
    d_img = raw_image.astype(np.uint8).copy()
    nms_res = nms(np.array(boxes), np.array(scores), thresh=0.5)
    print("detected objects: {}".format(len(nms_res)))
    for i in nms_res:
        d_img = cv2.rectangle(d_img, boxes[i][0], boxes[i][1], (255, 0, 0), 3)
        d_img = cv2.circle(d_img, centers[i], int(
            (boxes[i][1][0] - boxes[i][0][0])*0.2), (0, 0, 255), 2)
        
    if cv2.imwrite("result.png", d_img[..., ::-1]):
        print("result.png was generated")
    
