import os
import numpy as np
import torch
from PIL import Image


def colorize_mask(mask):
    """
    colorize voc class labels with voc color palette
    """
    import json
    with open('color_palette.json', 'r') as fh:
        palette = json.load(fh)
    new_mask = Image.fromarray(
                        mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


class MaskToTensor:
    """
    class to convert PIL.Image of voc mask to torch.Tensor
    """
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class DeNormalize:
    """
    class to invert Dataset Normalization
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """
    Compute usampling weights for bilinear, channel-wise interpolation.
    Requires in_channels == out_channels
    """
    assert in_channels == out_channels, ("requires in_channels == out_channels"
                                         "in_channels was {} and out_channels"
                                         "was {}".format(in_channels,
                                                         out_channels))
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) \
           * (1 - abs(og[1] - center) / factor)
    weight = np.zeros(
                (in_channels, out_channels, kernel_size, kernel_size),
                dtype=np.float64)
    weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt
    return torch.from_numpy(weight).float()


def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes,
                                                              num_classes)
    return hist


def evaluate(predictions, gts, num_classes):
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
