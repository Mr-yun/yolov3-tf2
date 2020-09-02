import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Concatenate, Add, LeakyReLU, UpSampling2D
from tensorflow.keras.models import Model


class Darknet53(object):
    def __init__(self, input_layer, num_class):
        self.input_layer = input_layer
        self.num_class = num_class

    def _conv2D_BN_Leaky(self, input_layer, filters_shape, downsample=False, activate=True, bn=True):
        # 卷积块
        # DarknetConv2D + BatchNormalization + LeakyReLU
        if downsample:
            input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
            padding = 'valid'
            strides = 2
        else:
            strides = 1
            padding = 'same'
        conv = Conv2D(filters=filters_shape[-1], kernel_size=filters_shape[0], strides=strides,
                      padding=padding,
                      use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                      bias_initializer=tf.constant_initializer(0.))(input_layer)  # todo use_bias???
        if bn: conv = BatchNormalization()(conv)
        if activate: conv = LeakyReLU(alpha=0.1)(conv)
        return conv

    def _resblock_body(self, input_conv, filters_shape, num_blocks):
        # 残差部分
        conv = self._conv2D_BN_Leaky(input_conv, filters_shape, downsample=True)
        # short_conv = conv
        for i in range(num_blocks):
            new_conv = self._conv2D_BN_Leaky(conv, filters_shape=(1, 1, filters_shape[-1], filters_shape[-2]))
            new_conv = self._conv2D_BN_Leaky(new_conv, filters_shape=(3, 3, filters_shape[-2], filters_shape[-1]))
            conv = Add()([conv, new_conv])

            # conv = short_conv + conv
        return conv

    def _upsample(self, input_layer):
        # 上采样
        return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')

    def _darknet_body(self, inputs):
        conv = self._conv2D_BN_Leaky(inputs, (3, 3, 3, 32))

        conv = self._resblock_body(conv, (3, 3, 32, 64), 1)
        conv = self._resblock_body(conv, (3, 3, 64, 128), 2)
        conv = self._resblock_body(conv, (3, 3, 128, 256), 8)
        route_1 = conv

        conv = self._resblock_body(conv, (3, 3, 256, 512), 8)
        route_2 = conv

        conv = self._resblock_body(conv, (3, 3, 512, 1024), 4)

        return route_1, route_2, conv

    def _conv_lbbox(self, conv):
        conv = self._conv2D_BN_Leaky(conv, (1, 1, 1024, 512))
        conv = self._conv2D_BN_Leaky(conv, (3, 3, 512, 1024))
        conv = self._conv2D_BN_Leaky(conv, (1, 1, 1024, 512))
        conv = self._conv2D_BN_Leaky(conv, (3, 3, 512, 1024))
        conv = self._conv2D_BN_Leaky(conv, (1, 1, 1024, 512))
        conv_lobj_branch = self._conv2D_BN_Leaky(conv, (3, 3, 512, 1024))
        conv_lbbox = self._conv2D_BN_Leaky(conv_lobj_branch, (1, 1, 1024, 3 * (self.num_class + 4 + 1)),
                                           activate=False, bn=False)  # 4: 中心点坐标和w,h. 1 背景
        return conv_lbbox, conv

    def _conv_mbbox(self, conv, route_2):
        conv = tf.concat([conv, route_2], axis=-1)

        conv = self._conv2D_BN_Leaky(conv, (1, 1, 768, 256))
        conv = self._conv2D_BN_Leaky(conv, (3, 3, 256, 512))
        conv = self._conv2D_BN_Leaky(conv, (1, 1, 512, 256))
        conv = self._conv2D_BN_Leaky(conv, (3, 3, 256, 512))
        conv = self._conv2D_BN_Leaky(conv, (1, 1, 512, 256))

        conv_mobj_branch = self._conv2D_BN_Leaky(conv, (3, 3, 256, 512))
        conv_mbbox = self._conv2D_BN_Leaky(conv_mobj_branch, (1, 1, 512, 3 * (self.num_class + 5)), activate=False,
                                           bn=False)
        return conv_mbbox, conv

    def _conv_sbbox(self, conv, route_1):
        conv = tf.concat([conv, route_1], axis=-1)

        conv = self._conv2D_BN_Leaky(conv, (1, 1, 384, 128))
        conv = self._conv2D_BN_Leaky(conv, (3, 3, 128, 256))
        conv = self._conv2D_BN_Leaky(conv, (1, 1, 256, 128))
        conv = self._conv2D_BN_Leaky(conv, (3, 3, 128, 256))
        conv = self._conv2D_BN_Leaky(conv, (1, 1, 256, 128))

        conv_sobj_branch = self._conv2D_BN_Leaky(conv, (3, 3, 128, 256))
        conv_sbbox = self._conv2D_BN_Leaky(conv_sobj_branch, (1, 1, 256, 3 * (self.num_class + 5)), activate=False,
                                           bn=False)
        return conv_sbbox

    def get_net(self):
        '''获取模型'''
        route_1, route_2, conv = self._darknet_body(self.input_layer)

        darknet = Model(self.input_layer, conv)

        lbbox, conv = self._conv_lbbox(conv)
        conv = self._conv2D_BN_Leaky(conv, (1, 1, 512, 256))
        if self.input_layer.shape[0] is not None:
            conv = self._upsample(conv)
        else:
            conv = UpSampling2D(2)(conv)

        mbbox, conv = self._conv_mbbox(conv, route_2)
        conv = self._conv2D_BN_Leaky(conv, (1, 1, 256, 128))
        if self.input_layer.shape[0] is not None:
            conv = self._upsample(conv)
        else:
            conv = UpSampling2D(2)(conv)

        sbbox = self._conv_sbbox(conv, route_1)
        return Model(self.input_layer, [lbbox, mbbox, sbbox])
