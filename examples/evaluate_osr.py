import argparse
import pandas as pd
import os
import os.path as osp
import numpy as np
from pprint import pprint
from tqdm import tqdm

from SSB import get_osr_datasets
from torchvision import transforms
import torchvision
import torch
from torch.utils.data import DataLoader
import timm

"""
OoD evaluation utils from: https://github.com/deeplearning-wisc/react/blob/master/util/metrics.py
NOTE: '1' is considered ID/Known, and '0' is considered OoD/Unkown
"""

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def compute_oscr(x1, x2, pred, labels):

    """
    :param x1: open set score for each known class sample (B_k,)
    :param x2: open set score for each unknown class sample (B_u,)
    :param pred: predicted class for each known class sample (B_k,)
    :param labels: correct class for each known class sample (B_k,)
    :return: Open Set Classification Rate
    """

    correct = (pred == labels)
    m_x1 = np.zeros(len(x1))
    m_x1[pred == labels] = 1
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)

    # Cutoffs are of prediction values

    CCR = [0 for x in range(n + 2)]
    FPR = [0 for x in range(n + 2)]

    idx = predict.argsort()

    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n - 1):
        CC = s_k_target[k + 1:].sum()
        FP = s_u_target[k:].sum()

        # True	Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2))

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n + 1] = 1.0
    FPR[n + 1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)

    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n + 1):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0

        OSCR = OSCR + h * w

    return OSCR

def cal_ood_metrics(known, novel, method=None):

    """
    Note that the convention here is that ID samples should be labelled '1' and OoD samples should be labelled '0'
    Computes standard OoD-detection metrics: mtypes = ['FPR' (FPR @ TPR 95), 'AUROC', 'DTERR', 'AUIN', 'AUOUT']
    """

    tp, fp, fpr_at_tpr95 = get_curve(known, novel, method)
    results = dict()

    # FPR
    mtype = 'FPR'
    results[mtype] = fpr_at_tpr95

    # AUROC
    mtype = 'AUROC'
    tpr = np.concatenate([[1.], tp/tp[0], [0.]])
    fpr = np.concatenate([[1.], fp/fp[0], [0.]])
    results[mtype] = -np.trapz(1.-fpr, tpr)

    # DTERR
    mtype = 'DTERR'
    results[mtype] = ((tp[0] - tp + fp) / (tp[0] + fp[0])).min()

    # AUIN
    mtype = 'AUIN'
    denom = tp+fp
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp/denom, [0.]])
    results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])

    # AUOUT
    mtype = 'AUOUT'
    denom = tp[0]-tp+fp[0]-fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[0]-fp)/denom, [.5]])
    results[mtype] = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])

    return results

def get_curve(known, novel, method=None):
    tp, fp = dict(), dict()
    fpr_at_tpr95 = dict()

    known.sort()
    novel.sort()

    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known),np.min(novel)])

    all = np.concatenate((known, novel))
    all.sort()

    num_k = known.shape[0]
    num_n = novel.shape[0]

    if method == 'row':
        threshold = -0.5
    else:
        threshold = known[round(0.05 * num_k)]

    tp = -np.ones([num_k+num_n+1], dtype=int)
    fp = -np.ones([num_k+num_n+1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l]-1, -1, -1)
            break
        elif n == num_n:
            tp[l+1:] = np.arange(tp[l]-1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l+1] = tp[l]
                fp[l+1] = fp[l] - 1
            else:
                k += 1
                tp[l+1] = tp[l] - 1
                fp[l+1] = fp[l]

    j = num_k+num_n-1
    for l in range(num_k+num_n-1):
        if all[j] == all[j-1]:
            tp[j] = tp[j+1]
            fp[j] = fp[j+1]
        j -= 1

    fpr_at_tpr95 = np.sum(novel > threshold) / float(num_n)

    return tp, fp, fpr_at_tpr95

def get_imagenet_standard_transform(image_size=224, crop_pct=0.875,
                                    interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                                    mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):

    test_transform_standard = transforms.Compose([
        transforms.Resize(int(image_size / crop_pct), interpolation),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor(mean),
            std=torch.tensor(std))
    ])

    return test_transform_standard

@torch.no_grad()
def test(model, test_loader):

    """
    Get class predictions and Maximum Softmax Score for all instances in loader
    """

    model.eval()
    id_preds = []       # Store class preds
    osr_preds = []      # Stores OSR preds

    # First extract all features
    for batch_idx, (images, _, _) in enumerate(tqdm(test_loader)):

        images = images.cuda()

        # Get logits
        logits = model(images)
        sftmax = torch.nn.functional.softmax(logits, dim=-1)

        id_preds.extend(sftmax.argmax(dim=-1).cpu().numpy())
        osr_preds.extend(sftmax.max(dim=-1)[0].cpu().numpy())

    id_preds = np.array(id_preds)
    osr_preds = np.array(osr_preds)
    
    return id_preds, osr_preds

def main():

    args = parser.parse_args()
    print(args)
    
    # ------------------------
    # GET MODEL
    # ------------------------
    model = timm.create_model('deit3_base_patch16_224.fb_in1k', pretrained='True')
    model.eval()
    model.cuda()

    # ------------------------
    # DATASETS AND DATALOADERS
    # ------------------------
    transform = get_imagenet_standard_transform()
    datasets = get_osr_datasets(dataset_name='imagenet',
                                osr_split='Hard', 
                                train_transform=None, 
                                test_transform=transform, 
                                split_train_val=False,
                                eval_only=True)

    dataloaders = {}
    for k, v, in datasets.items():
        if v is not None:
            dataloaders[k] = DataLoader(v, batch_size=128, shuffle=False, sampler=None, num_workers=8)

    # ------------------------
    # GET PREDS AND LABELS
    # ------------------------
    # Get labels
    osr_labels = [1] * len(datasets['test_known']) + [0] * len(datasets['test_unknown'])                  # 1 if sample is ID else 0
    osr_labels = np.array(osr_labels)
    id_labels = np.array([x[1] for x in datasets['test_known'].samples])

    # Get preds
    id_preds, osr_preds_id_samples = test(model, dataloaders['test_known'])
    _, osr_preds_osr_samples = test(model, dataloaders['test_unknown'])

    results = cal_ood_metrics(osr_preds_id_samples, osr_preds_osr_samples)         # Compute OoD metrics
    results['OSCR'] = compute_oscr(
        osr_preds_id_samples,
        osr_preds_osr_samples,
        id_preds,
        id_labels
    )
    results['ACC'] = (id_labels == id_preds).mean()

    print(f"Current iid performance: {results['ACC'] * 100:.2f}%")
    pprint(results)

    output_path = os.path.join(args.output, "scores.txt")
    print("Writing scores to ", output_path)
    with open(output_path, mode="w") as f:
        for metric, perf in results.items():
            print(f'{metric}: {100 * perf:.2f}', file=f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--output', default='', type=str, metavar='PATH',
                        help='path to the output dir')
    main()