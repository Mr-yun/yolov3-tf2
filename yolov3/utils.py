import colorsys
import cv2

import numpy as np

def get_files_value(files_path):
    with open(files_path) as f:
        files_value = f.readlines()
    files_value = [c.strip() for c in files_value]
    return files_value

def get_colors(class_names):
    # 种类->标注颜色
    len_classes = len(class_names)
    hsv_tuples = [(x / len_classes, 1., 1.)
                  for x in range(len_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    np.random.seed(10101)
    np.random.shuffle(colors)
    np.random.seed(None)
    return colors

def letterbox_image(image, target_size, gt_boxes=None, max_boxes=20):
    '''输入图像等比例缩放成 模型图像大小'''
    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded
    else:
        max_boxes = max_boxes if len(gt_boxes) < max_boxes else len(gt_boxes)

        box_data = np.zeros((max_boxes, 5))
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        box_data[:len(gt_boxes)] = gt_boxes

        return image_paded, box_data