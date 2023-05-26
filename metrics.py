import torch

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou + 1)

    true_positives = (output_ & target_).sum()
    false_positives = (output_ & ~target_).sum()
    false_negatives = (target_ & ~output_).sum()
    recall = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    true_negatives = (~output_ & ~target_).sum()
    specificity = true_negatives / (true_negatives + false_positives)

    return iou, dice, recall, precision, specificity