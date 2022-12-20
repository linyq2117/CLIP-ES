import torch
import os
import numpy as np
import torch.nn.functional as F
import joblib
import multiprocessing
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import cv2
from PIL import Image
import argparse


class DenseCRF(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        C, H, W = probmap.shape

        U = utils.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)

        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q

def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

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

def crf(n_jobs, is_coco=False):
    """
    CRF post-processing on pre-computed logits
    """

    # Configuration
    torch.set_grad_enabled(False)
    print("# jobs:", n_jobs)

    # CRF post-processor
    postprocessor = DenseCRF(
        iter_max=10,
        pos_xy_std=1,
        pos_w=3,
        bi_xy_std=67,
        bi_rgb_std=3,
        bi_w=4,
    )

    # Process per sample
    def process(i):
        image_id = eval_list[i]
        image_path = os.path.join(args.image_root, image_id + '.jpg')
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        label_path = os.path.join(args.gt_root, image_id + '.png')
        gt_label = np.asarray(Image.open(label_path), dtype=np.int32)
        # Mean subtraction
        image -= mean_bgr
        # HWC -> CHW
        image = image.transpose(2, 0, 1)

        filename = os.path.join(args.cam_out_dir, image_id + ".npy")
        cam_dict = np.load(filename, allow_pickle=True).item()
        cams = cam_dict['attn_highres']
        bg_score = np.power(1 - np.max(cams, axis=0, keepdims=True), 1)
        cams = np.concatenate((bg_score, cams), axis=0)
        prob = cams

        image = image.astype(np.uint8).transpose(1, 2, 0)
        prob = postprocessor(image, prob)

        label = np.argmax(prob, axis=0)
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
        label = keys[label]
        if not args.eval_only:
            confidence = np.max(prob, axis=0)
            label[confidence < 0.95] = 255
            cv2.imwrite(os.path.join(args.pseudo_mask_save_path, image_id + '.png'), label.astype(np.uint8))

        return label.astype(np.uint8), gt_label.astype(np.uint8)

    # CRF in multi-process
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")(
           [joblib.delayed(process)(i) for i in range(len(eval_list))]
    )
    if args.eval_only:
        preds, gts = zip(*results)

        # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
        score = scores(gts, preds, n_class=21 if not is_coco else 81)
        print(score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_out_dir", default="./cam_out", type=str)
    parser.add_argument("--pseudo_mask_save_path", default="/home/xxx/code/code48/ablation/usss/voc/val_attn07_crf", type=str)
    parser.add_argument("--split_file", default="/home/xxx/datasets/VOC2012/ImageSets/Segmentation/train.txt",
                        type=str)
    parser.add_argument("--cam_eval_thres", default=2, type=float)
    parser.add_argument("--gt_root", default="/home/xxx/datasets/VOC2012/SegmentationClassAug", type=str)
    parser.add_argument("--image_root", default="/home/xxx/datasets/VOC2012/JPEGImages", type=str)
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    is_coco = 'coco' in args.cam_out_dir
    if 'voc' in args.cam_out_dir:
        eval_list = list(np.loadtxt(args.split_file, dtype=str))
    elif 'coco' in args.cam_out_dir:
        file_list = tuple(open(args.split_file, "r"))
        file_list = [id_.rstrip().split(" ") for id_ in file_list]
        eval_list = [x[0] for x in file_list]#[:2000]
    print('{} images to eval'.format(len(eval_list)))

    if not args.eval_only and not os.path.exists(args.pseudo_mask_save_path):
        os.makedirs(args.pseudo_mask_save_path)

    mean_bgr = (104.008, 116.669, 122.675)
    n_jobs =multiprocessing.cpu_count()
    crf(n_jobs, is_coco)
