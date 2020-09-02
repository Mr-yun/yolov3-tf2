import cv2
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
)
from yolov3.yolo import Yolo

obj_yolo = Yolo('./model_data/yolo.h5',
                './model_data/coco.names',
                model_image_size=(512, 512))
img = cv2.imread('./docs/kite.jpg')
print(img.shape)
img = obj_yolo.detect_image(img)
cv2.imshow('a', img)
cv2.waitKey(0)
