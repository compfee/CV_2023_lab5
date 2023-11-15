import torch.nn as nn

class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #         inputs = F.softmax(inputs, dim=1)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        iou = (intersection + smooth) / (union + smooth)

        return iou

def IoU_by_class(inputs, target, smooth=1):
    iou_s = []
    for img in range(inputs.size()[0]):
        tmp = []
        for channel in range(inputs.size()[1]):
            flat_pred = inputs[channel].view(-1)
            flat_target = target[channel].view(-1)

            intersection = (flat_pred * flat_target).sum()
            total = (flat_pred + flat_target).sum()
            union = total - intersection

            iou = (intersection + smooth)/(union + smooth)
            tmp.append(iou.numpy())
        iou_s.append(tmp)

    return iou_s

def Pixel_accuracy(inputs, target):

  y_pred_argmax = inputs.argmax(dim=1)
  y_true_argmax = target.argmax(dim=1)

  correct_pixels = (y_pred_argmax == y_true_argmax).count_nonzero()
  uncorrect_pixels = (y_pred_argmax != y_true_argmax).count_nonzero()
  pixel_acc = (correct_pixels / (correct_pixels + uncorrect_pixels)).item()

  return pixel_acc

