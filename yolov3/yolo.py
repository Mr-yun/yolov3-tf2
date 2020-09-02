from os import path

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model

from yolov3.const import defaul_anchors, anchor_mask
from yolov3.utils import get_files_value, get_colors, letterbox_image


class Yolo(object):
    def __init__(self, model_path, classes_path,
                 model_image_size=(416, 416), anchors=None,
                 iou_threshold=0.45):
        '''
        :param model_path:  yolo h5 文件存放路径
        :param classes_path: class 存放路径
        :param model_image_size:  模型图像大小
        :param anchors: 先验证框
        '''
        self.class_names = get_files_value(classes_path)
        self.anchors = anchors or defaul_anchors
        self.model = self._load_model(model_path)
        self.colors = get_colors(self.class_names)
        self.model_image_size = self._check_model_image_size(
            model_image_size)
        self.iou_threshold = iou_threshold

    def _check_model_image_size(self, model_image_size):
        if model_image_size != (None, None):
            assert model_image_size[0] % 32 == 0, '模型图像需要是32倍数'
            assert model_image_size[1] % 32 == 0, '模型图像需要是32倍数'
        return model_image_size

    def _load_model(self, model_path):
        '''
        加载模型,未支持tiny模型
        :return:
        '''
        if path.exists(model_path):
            assert model_path.endswith('.h5'), '模型文件需要以.h5结尾'
        else:
            raise Exception("模型输入文件路径不存在")

        model = load_model(model_path, compile=False)
        assert model.layers[-1].output_shape[-1] == len(
            self.anchors) / len(model.output) * (len(self.class_names) + 5)
        return model

    def detect_image(self, image, score_threshol=0.5, iou_threshold=None):
        iou_threshold = iou_threshold or self.iou_threshold
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxed_image = letterbox_image(image,
                                      self.model_image_size)  # 输入图像转成 识别大小
        boxed_image = boxed_image[np.newaxis, ...].astype(np.float32)
        out_boxes, out_scores, out_classes = self.eval_img(self.model(boxed_image),
                                                           image.shape[:2],
                                                           score_threshol, iou_threshold)  # 卷积识别
        image = self._draw_rectangle(image, out_boxes, out_scores, out_classes)  # 画识别方框
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    @staticmethod
    def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
        """转换识别结果
        例如:(batch_size,13,13,255) -> (batch_size,13,13,3,85)
        """
        num_anchors = len(anchors)
        anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

        grid_shape = K.shape(feats)[1:3]  # 特征层高和宽
        grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                        [1, grid_shape[1], 1, 1])
        grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                        [grid_shape[0], 1, 1, 1])
        grid = K.concatenate([grid_x, grid_y])
        # 生成 特征层网格点坐标
        # 如(13,13)特征层面,[[(0,0)..(0,12)]..[(12,0)..[12,12]]]

        grid = K.cast(grid, K.dtype(feats))
        feats = K.reshape(
            feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

        # 网格点坐标(特征层中心点)+识别结果(偏移量)
        box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
        box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))

        if calc_loss == True:
            return grid, feats, box_xy, box_wh
        else:
            box_confidence = K.sigmoid(feats[..., 4:5])
            box_class_probs = K.sigmoid(feats[..., 5:])  # todo:这里调用激活函数是起到什么作用
            return box_xy, box_wh, box_confidence, box_class_probs

    def correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        '''识别点在图像中,真实位置'''
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = K.cast(input_shape, K.dtype(box_yx))
        image_shape = K.cast(image_shape, K.dtype(box_yx))
        new_shape = K.round(image_shape * K.min(input_shape / image_shape))  # 等比例缩放后新的长宽
        offset = (input_shape - new_shape) / 2. / input_shape  # /2 是因为新图像,上下居中放置
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = K.concatenate([
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2]  # x_max
        ])

        boxes *= K.concatenate([image_shape, image_shape])
        return boxes

    def boxes_and_scores(self, feats, anchors, num_classes, input_shape, image_shape):
        '''预存结果转换为真实值'''
        box_xy, box_wh, box_confidence, box_class_probs = self.yolo_head(feats,
                                                                         anchors, num_classes, input_shape)
        boxes = self.correct_boxes(box_xy, box_wh, input_shape, image_shape)
        boxes = K.reshape(boxes, [-1, 4])
        box_scores = box_confidence * box_class_probs
        box_scores = K.reshape(box_scores, [-1, num_classes])
        return boxes, box_scores

    # @tf.function
    def eval_img(self, y_pred, image_shape,
                 score_threshold, iou_threshold, max_boxes=200):
        '''

        :param image_data:
        :param score_threshold:
        :param image_shape:
        :param max_boxes: todo 此次预设数量会约束识别出数量
        :return:
        '''

        image_shape = tf.constant(image_shape)

        num_classes = len(self.class_names)

        num_layers = len(y_pred)
        input_shape = K.shape(y_pred[0])[1:3] * 32
        boxes = []
        box_scores = []
        for l in range(num_layers):
            _boxes, _box_scores = self.boxes_and_scores(y_pred[l],
                                                        self.anchors[anchor_mask[l]],
                                                        num_classes, input_shape,
                                                        image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        boxes = K.concatenate(boxes, axis=0)
        box_scores = K.concatenate(box_scores, axis=0)

        mask = box_scores >= score_threshold
        max_boxes_tensor = K.constant(max_boxes, dtype='int32')
        boxes_ = []
        scores_ = []
        classes_ = []
        for c in range(num_classes):
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
            nms_index = tf.image.non_max_suppression(
                class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
            class_boxes = K.gather(class_boxes, nms_index)
            class_box_scores = K.gather(class_box_scores, nms_index)
            classes = K.ones_like(class_box_scores, 'int32') * c
            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        boxes_ = K.concatenate(boxes_, axis=0)
        scores_ = K.concatenate(scores_, axis=0)
        classes_ = K.concatenate(classes_, axis=0)
        return boxes_, scores_, classes_

    def _draw_rectangle(self, image, out_boxes, out_scores, out_classes, show_label=True):
        bbox_thick = int(0.6 * (sum(image.shape[:2])) / 600)

        for i, c in enumerate(out_classes):
            c = c.numpy()
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))

            c1, c2 = (left, top), (right, bottom)
            cv2.rectangle(image, c1, c2, self.colors[c], bbox_thick)
            if show_label:
                label = '{} {:.2f}'.format(predicted_class, score)
                cv2.putText(image, label, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
                # print(top, left, bottom, right, predicted_class, score)
        return image


class MultiboxLoss(object):
    def __init__(self, num_classes, input_shape=416, anchors=None,
                 ignore_thresh=.5, print_loss=True):
        self.anchors = anchors or defaul_anchors
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh
        self.print_loss = print_loss
        self.num_layers = len(self.anchors) // 3
        self.input_shape = input_shape

    def loss(self, args):
        y_pred = args[:self.num_layers]
        y_true = args[self.num_layers:]
        input_shape = K.cast(K.shape(y_pred[0])[1:3] * 32, K.dtype(y_true[0]))
        grid_shapes = [K.cast(K.shape(y_pred[l])[1:3], K.dtype(y_true[0])) for l in range(self.num_layers)]
        loss = 0
        m = K.shape(y_pred[0])[0]
        mf = K.cast(m, K.dtype(y_pred[0]))

        for l in range(self.num_layers):
            object_mask = y_true[l][..., 4:5]
            true_class_probs = y_true[l][..., 5:]

            grid, raw_pred, pred_xy, pred_wh = Yolo.yolo_head(y_pred[l],
                                                              self.anchors[anchor_mask[l]], self.num_classes,
                                                              input_shape,
                                                              calc_loss=True)
            pred_box = K.concatenate([pred_xy, pred_wh])

            raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
            raw_true_wh = K.log(y_true[l][..., 2:4] / self.anchors[anchor_mask[l]] * input_shape[::-1])
            raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))
            box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

            ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
            object_mask_bool = K.cast(object_mask, 'bool')

            def loop_body(b, ignore_mask):
                true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
                iou = self.box_iou(pred_box[b], true_box)
                best_iou = K.max(iou, axis=-1)
                ignore_mask = ignore_mask.write(b, K.cast(best_iou < self.ignore_thresh, K.dtype(true_box)))
                return b + 1, ignore_mask

            _, ignore_mask = tf.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
            ignore_mask = ignore_mask.stack()
            ignore_mask = K.expand_dims(ignore_mask, -1)

            xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                                                           from_logits=True)
            wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])

            confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                              (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                        from_logits=True) * ignore_mask  # 交叉熵

            class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:],
                                                             from_logits=True)

            xy_loss = tf.reduce_sum(xy_loss) / mf
            wh_loss = tf.reduce_sum(wh_loss) / mf
            confidence_loss = tf.reduce_sum(confidence_loss) / mf
            class_loss = tf.reduce_sum(class_loss) / mf
            loss = xy_loss + wh_loss + confidence_loss + class_loss

        loss = K.expand_dims(loss, axis=-1)
        return loss

    def box_iou(self, b1, b2):
        b1 = K.expand_dims(b1, -2)
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh / 2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half

        b2 = K.expand_dims(b2, 0)
        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh / 2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        intersect_mins = K.maximum(b1_mins, b2_mins)
        intersect_maxes = K.minimum(b1_maxes, b2_maxes)
        intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        iou = intersect_area / (b1_area + b2_area - intersect_area)

        return iou
