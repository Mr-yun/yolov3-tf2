from os import path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# from ssd.utils import ModelCheckpoint
from yolov3.const import defaul_anchors
from yolov3.dateset import Dataset
from yolov3.nets import Darknet53
from yolov3.yolo import MultiboxLoss

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
)


def random_annotation(val_split):
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val
    return lines, num_train, num_val


def create_model(input_shape, num_classes,
                 weights_path='./yolov3/model_data/yolo_weights.h5'):
    num_anchors = len(defaul_anchors)
    # input_tensor = tf.keras.layers.Input(INPUT_SHAPE)
    h, w = input_shape[:2]

    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], \
                           num_anchors // 3, num_classes + 5)) for l in range(3)]

    model_body = Darknet53(tf.keras.layers.Input(shape=(None, None, 3)),
                           NUM_CLASSES).get_net()
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
    print('Load weights {}.'.format(weights_path))

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    num = (185, len(model_body.layers) - 3)[1]
    for i in range(num): model_body.layers[i].trainable = False

    model_loss = Lambda(MultiboxLoss(NUM_CLASSES).loss, output_shape=(1,), name='yolov3_loss')(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss) # print(model.summary())
    return model


def debug_part():
    obj_loss = MultiboxLoss(NUM_CLASSES)
    # while True:
    #     for image_data in gen.generate(lines[:num_train]):
    #         pred_result = model(image_data[0][0], training=True)
    #         y_true= [tf.convert_to_tensor(i) for i in image_data[0][1:]]
    #         obj_loss.loss(y_true,pred_result)

    #


if __name__ == "__main__":
    log_dir = ".\\logs\\"
    annotation_path = '2007_train.txt'

    NUM_CLASSES = 20
    INPUT_SHAPE = [416, 416, 3]

    # 0.1用于验证，0.9用于训练
    lines, num_train, num_val = random_annotation(0.1)

    model = create_model(INPUT_SHAPE, NUM_CLASSES)

    # 训练参数设置
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    if True:
        BATCH_SIZE = 32
        Lr = 1e-3
        Init_Epoch = 0
        Freeze_Epoch = 50
        gen = Dataset(BATCH_SIZE, NUM_CLASSES, input_shape=416)

        model.compile(optimizer=Adam(lr=Lr),
                      loss={'yolov3_loss': lambda y_true, y_pred: y_pred})
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(
            num_train, num_val, BATCH_SIZE))
        model.fit_generator(gen.generate(lines[:num_train]),
                            steps_per_epoch=max(1, num_train // BATCH_SIZE),
                            validation_data=gen.generate(lines[num_train:]),
                            validation_steps=max(1, num_val // BATCH_SIZE),
                            epochs=Freeze_Epoch,
                            initial_epoch=Init_Epoch,
                            callbacks=[logging, checkpoint])
        model.save_weights(path.join(log_dir, 'trained_weights_stage_1.h5'))
    print("end train yolov3")
