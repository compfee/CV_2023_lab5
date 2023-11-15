# Numpy
import numpy as np

# Torch
import torch
import torch.nn.functional as F

num_classes = 8
h_crop = 256
w_crop = 256

# закодированные классы в бинарном виде и в one hot encoder
binary_encoded = [[0, 0, 0],[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
one_hot_encoded = [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]]

# бинарный вид в one hot
def bin2ohe(mask, num_class, binary_encoded, one_hot_encoded):
    mask = mask.permute(1, 2, 0)
    mask = mask.numpy()
    mask = mask.astype(np.int64)
    h, w = mask.shape[:-1]
    layout = np.zeros((h, w, num_class), dtype=np.int64)
    for i, label in enumerate(binary_encoded):
        layout[np.all(mask == label, axis=-1)] = one_hot_encoded[i]
    layout = layout.astype(np.float64)
    layout = torch.from_numpy(layout)
    layout = layout.permute(2, 0, 1)
    return layout

# one hot вид в бинарный
def ohe2bin(mask, one_hot_encoded, binary_encoded):
    mask = mask.permute(1, 2, 0)
    mask = mask.numpy()
    h, w = mask.shape[:-1]
    layout = np.zeros((h, w, 3), dtype=np.int64)
    for i, label in enumerate(one_hot_encoded):
        layout[np.all(mask == label, axis=-1)] = binary_encoded[i]

    layout = layout.astype(np.float64)
    layout = torch.from_numpy(layout)
    layout = layout.permute(2, 0, 1)
    return layout