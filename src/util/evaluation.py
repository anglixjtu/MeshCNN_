import math


def get_patk(gt_label, label_list, K):
    """Calculate Precision@K

    Args:
        gt_label: the ground truth label for query
        label_list: labels for the retrieved ranked-list
        K: top K in the ranked-list for computing precision.
           Set to len(label_list) if compute P@N

    Returns:
        P@K score
    """
    patk = 0
    for i, pred_label in enumerate(label_list[:K]):
        if gt_label == pred_label:
            patk += 1
    patk /= K

    return patk


def get_map(gt_label, label_list, K):
    """Calculate mean average precision
        Args:
        gt_label: the ground truth label for query
        label_list: labels for the retrieved ranked-list

    Returns:
        AP score
    """
    # TODO: mAP and NDCG need to be divided by total number of
    # relevant models in ground truth set, not in the retrieval list itself
    map = 0
    counter = 0
    for k in range(len(label_list)):
        if gt_label == label_list[k]:  # at each relevant positions
            map += get_patk(gt_label, label_list, k+1)
            counter += 1
    '''if counter == 0:
        map = 0
    else:
        map /= counter'''

    map /= len(label_list)

    return map


def get_ndcg(gt_label, label_list, K):
    """Calculate Normalized Cumulative Gain (NDCG) at rank K

    Args:
        gt_label: the ground truth label for query
        label_list: labels for the retrieved ranked-list
        K: top K in the ranked-list for computing precision.
           Set to len(label_list) if compute NDCG@N

    Returns:
        NDCG@K

    """
    dcg = 0
    dcg_gt = 0
    if gt_label == label_list[0]:
        dcg += 1
    dcg_gt += 1

    for i, pred_label in enumerate(label_list[1:K]):  # iterate from rank 2
        if gt_label == pred_label:
            dcg += 1 / math.log(i+2, 2)
        dcg_gt += 1 / math.log(i+2, 2)
    dcg /= dcg_gt

    return dcg


class Evaluator(object):
    def __init__(self, metrics) -> None:
        self.metrics = dict()
        for metric in metrics:
            setattr(self, metric+'_acc', 0.)
            self.metrics[metric] = 0.
        self.counter = 0

    def update(self, gt, pred, K, len=1):
        self.counter += len
        for metric in self.metrics.keys():
            acc = getattr(self, metric+'_acc')
            acc += eval('get_'+metric)(gt, pred, K)
            setattr(self, metric+'_acc', acc)
            self.metrics[metric] = acc/self.counter
