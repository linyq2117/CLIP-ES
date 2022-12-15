
import numpy as np
import os
from PIL import Image
import argparse

def print_iou(iou, dname='voc'):
    iou_dict = {}
    for i in range(len(iou)-1):
        iou_dict[i] = iou[i+1]
    print(iou_dict)

    return iou_dict

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist

def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }

def run_eval_cam(args, print_log=False, is_coco=False):
    preds = []
    labels = []
    n_images = 0
    for i, id in enumerate(eval_list):
        n_images += 1
        if args.cam_type == 'png':
            label_path = os.path.join(args.cam_out_dir, id + '.png')
            cls_labels = np.asarray(Image.open(label_path), dtype=np.uint8)
        else:
            cam_dict = np.load(os.path.join(args.cam_out_dir, id + '.npy'), allow_pickle=True).item()
            cams = cam_dict[args.cam_type]
            if 'bg' not in args.cam_type:
                if args.cam_eval_thres < 1:
                    cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
                else:
                    bg_score = np.power(1 - np.max(cams, axis=0, keepdims=True), args.cam_eval_thres)
                    cams = np.concatenate((bg_score, cams), axis=0)
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            cls_labels = np.argmax(cams, axis=0)
            cls_labels = keys[cls_labels].astype(np.uint8)
        preds.append(cls_labels)
        gt_file = os.path.join(args.gt_root, '%s.png' % id)
        gt = np.array(Image.open(gt_file)).astype(np.uint8)
        labels.append(gt)

    iou = scores(labels, preds, n_class=21 if not is_coco else 81)

    if print_log:
        print(iou)

    return iou["Mean IoU"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_out_dir", default="./cam_out", type=str)
    parser.add_argument("--cam_type", default="attn_highres", type=str)
    parser.add_argument("--split_file", default="/home/xxx/datasets/VOC2012/ImageSets/Segmentation/train.txt", type=str)
    parser.add_argument("--cam_eval_thres", default=2, type=float)
    parser.add_argument("--gt_root", default="/home/xxx/datasets/VOC2012/SegmentationClassAug", type=str)
    args = parser.parse_args()

    is_coco = 'coco' in args.cam_out_dir
    if 'voc' in args.cam_out_dir:
        eval_list = list(np.loadtxt(args.split_file, dtype=str))
    elif 'coco' in args.cam_out_dir:
        file_list = tuple(open(args.split_file, "r"))
        file_list = [id_.rstrip().split(" ") for id_ in file_list]
        eval_list = [x[0] for x in file_list]#[:2000]
    print('{} images to eval'.format(len(eval_list)))

    if 'bg' in args.cam_type or 'png' in args.cam_type:
        iou = run_eval_cam(args, True)
    else:
        if args.cam_eval_thres < 1:
            thres_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
        else:
            if 'attn' in args.cam_type:
                thres_list = [1, 2]
            else:
                thres_list =[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        max_iou = 0
        max_thres = 0
        for thres in thres_list:
            args.cam_eval_thres = thres
            iou = run_eval_cam(args, print_log=False, is_coco=is_coco)
            print(thres, iou)
            if iou > max_iou:
                max_iou = iou
                max_thres = thres

        args.cam_eval_thres = max_thres
        iou = run_eval_cam(args, print_log=True, is_coco=is_coco)
