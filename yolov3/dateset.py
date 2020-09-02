from os import path
from random import random, uniform, randint

import cv2
import numpy as np

from yolov3.const import defaul_anchors, defaul_input_shape, anchor_mask
from yolov3.utils import letterbox_image


class DataEnhancement(object):
    '''
    数据增强
    '''

    def __init__(self, train_input_shape, lightness=10., saturation=20.):
        self.train_input_shape = train_input_shape
        self.lightness = lightness
        self.saturation = saturation

    def rand(slef, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def random_horizontal_flip(self, image, bboxes):
        # 镜像翻转
        if random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]

        return image, bboxes

    def random_crop(self, image, bboxes):
        '''随机裁剪'''
        if random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + uniform(0, max_d_trans)))

            image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):
        if random() < 0.5:
            hlsImg = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2HLS)

            # 调整亮度
            hlsImg[:, :, 1] = (randint(1.0, self.lightness) / float(100)) * hlsImg[:, :, 1]
            hlsImg[:, :, 1][hlsImg[:, :, 1] > 1] = 1

            # 饱和度
            hlsImg[:, :, 2] = (randint(1.0, self.saturation) / float(100)) * hlsImg[:, :, 2]
            hlsImg[:, :, 2][hlsImg[:, :, 2] > 1] = 1

            image = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR) * 255
            image = image.astype(np.uint8)
        return image, bboxes

    def create_new_img(self, annotation):
        line = annotation.split()
        image_path = line[0]
        if not path.exists(image_path):
            raise KeyError("%s does not exist ... " % image_path)
        image = cv2.imread(image_path)
        bboxes = np.array([list(map(int, box.split(','))) for box in line[1:]])

        image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
        image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
        image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_paded, gt_boxes = letterbox_image(image, [self.train_input_shape, self.train_input_shape],
                                                gt_boxes=bboxes)

        return image_paded, gt_boxes


class Dataset(object):

    def __init__(self, batch_size, num_classes,
                 input_shape=None
                 ):
        '''
        :param bbox_util:  先验框
        :param batch_size: 批次
        :param image_size: (高,宽)
        :param num_classes: 识别种类(包含背景)
        '''
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.input_shape = input_shape or defaul_input_shape

    def generate(self, annotation_lines):
        obj_data_enhancement = DataEnhancement(self.input_shape)
        np.random.shuffle(annotation_lines)
        n = len(annotation_lines)
        i = 0
        while True:
            image_data = []
            box_data = []
            for b in range(self.batch_size):
                if i == n:
                    np.random.shuffle(annotation_lines)
                    i = 0
                i += 1
                image, box = obj_data_enhancement.create_new_img(annotation_lines[i])
                image_data.append(image)
                box_data.append(box)
            image_data = np.array(image_data)
            box_data = np.array(box_data)
            y_true = self.preprocess_true_boxes(box_data)
            yield [image_data, *y_true], np.zeros(self.batch_size)

    def preprocess_true_boxes(self, true_boxes):
        assert (true_boxes[..., 4] < self.num_classes).all(), '种类错误'

        num_layers = 3  # default setting
        true_boxes = np.array(true_boxes, dtype='float32')
        input_shape = np.array([self.input_shape, self.input_shape], dtype='int32')
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2  # 图像中心点
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]  # 图像宽高
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]  # 归一化操作
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

        m = true_boxes.shape[0]
        grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
        y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 4 + 1 + self.num_classes),
                           dtype='float32') for l in range(num_layers)]  # 4 表示 (x,y,w,h) 1 表示是否为背景

        anchors = np.expand_dims(defaul_anchors, 0)
        anchor_maxes = anchors / 2.
        anchor_mins = -anchor_maxes
        valid_mask = boxes_wh[..., 0] > 0  # 忽略宽高小于0的无效数据

        for b in range(m):
            wh = boxes_wh[b, valid_mask[b]]
            if len(wh) == 0: continue
            # Expand dim to apply broadcasting.
            wh = np.expand_dims(wh, -2)
            box_maxes = wh / 2.
            box_mins = -box_maxes

            iou = self.cal_iou(box_mins, box_maxes,
                               anchor_mins, anchor_maxes,
                               wh, anchors)

            best_anchor = np.argmax(iou, axis=-1)  # 选择最大iou的index

            for t, n in enumerate(best_anchor):
                for l in range(num_layers):
                    if n in anchor_mask[l]:
                        i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                        j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                        k = anchor_mask[l].index(n)
                        c = true_boxes[b, t, 4].astype('int32')
                        y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                        y_true[l][b, j, i, k, 4] = 1
                        y_true[l][b, j, i, k, 5 + c] = 1

        return y_true

    def cal_iou(self, box_mins, box_maxes,
                anchor_mins, anchor_maxes,
                wh, anchors):
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)
        return iou
