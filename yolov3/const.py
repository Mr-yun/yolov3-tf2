import numpy as np

defaul_anchors = np.array([[10., 13.], [16., 30.], [33., 23.],
                           [30., 61.], [62., 45.], [59., 119.],
                           [116., 90.], [156., 198.], [373., 326.]])

anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

defaul_input_shape = (416, 416)