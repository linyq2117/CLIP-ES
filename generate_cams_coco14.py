# -*- coding:UTF-8 -*-
from pytorch_grad_cam import GradCAM
import torch
import clip
from PIL import Image
import numpy as np
import cv2
import os

from tqdm import tqdm
from pytorch_grad_cam.utils.image import scale_cam_image
from utils import scoremap2bbox
from clip_text import class_names, new_class_names_coco, BACKGROUND_CATEGORY_COCO
import argparse
from torch import multiprocessing
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import warnings
warnings.filterwarnings("ignore")
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0

def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def split_dataset(dataset, all_label_list, n_splits):
    if n_splits == 1:
        return [dataset], [all_label_list]
    part = len(dataset) // n_splits
    dataset_list = []
    split_label_list = []
    for i in range(n_splits - 1):
        dataset_list.append(dataset[i*part:(i+1)*part])
        split_label_list.append(all_label_list[i*part:(i+1)*part])
    dataset_list.append(dataset[(i+1)*part:])
    split_label_list.append(all_label_list[(i+1)*part:])

    return dataset_list, split_label_list

def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).to(device) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights.t()


class ClipOutputTarget:
    def __init__(self, category):
        self.category = category
    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform_resize(h, w):
    return Compose([
        Resize((h,w), interpolation=BICUBIC),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def img_ms_and_flip(img_path, ori_height, ori_width, scales=[1.0], patch_size=16):
    all_imgs = []
    for scale in scales:
        preprocess = _transform_resize(int(np.ceil(scale * int(ori_height) / patch_size) * patch_size), int(np.ceil(scale * int(ori_width) / patch_size) * patch_size))
        image = preprocess(Image.open(img_path))
        image_ori = image
        image_flip = torch.flip(image, [-1])
        all_imgs.append(image_ori)
        all_imgs.append(image_flip)
    return all_imgs


def perform(process_id, dataset_list, args, model, bg_text_features, fg_text_features, cam, split_label_list):
    n_gpus = torch.cuda.device_count()
    device_id = "cuda:{}".format(process_id % n_gpus)
    databin = dataset_list[process_id]
    all_label_list = split_label_list[process_id]
    model = model.to(device_id)
    bg_text_features = bg_text_features.to(device_id)
    fg_text_features = fg_text_features.to(device_id)
    for im_idx, im in enumerate(tqdm(databin)):
        img_path = os.path.join(args.img_root, im)
        ori_image = Image.open(img_path)
        ori_height, ori_width = np.asarray(ori_image).shape[:2]
        label_id_list = all_label_list[im_idx]
        label_list = []
        for lid in label_id_list:
            label_list.append(new_class_names_coco[int(lid)])
        label_id_list = [int(lid) for lid in label_id_list]

        ms_imgs = img_ms_and_flip(img_path, ori_height, ori_width, scales=[1.0])
        ms_imgs = [ms_imgs[0]]
        cam_all_scales = []
        highres_cam_all_scales = []
        refined_cam_all_scales = []
        for image in ms_imgs:
            image = image.unsqueeze(0)
            h, w = image.shape[-2], image.shape[-1]
            image = image.to(device_id)
            image_features, attn_weight_list = model.encode_image(image, h, w)

            cam_to_save = []
            highres_cam_to_save = []
            refined_cam_to_save = []
            keys = []

            bg_features_temp = bg_text_features.to(device_id)  # [bg_id_for_each_image[im_idx]].to(device_id)
            fg_features_temp = fg_text_features[label_id_list].to(device_id)
            text_features_temp = torch.cat([fg_features_temp, bg_features_temp], dim=0)
            input_tensor = [image_features, text_features_temp.to(device_id), h, w]

            for idx, label in enumerate(label_list):
                keys.append(new_class_names_coco.index(label))
                targets = [ClipOutputTarget(label_list.index(label))]

                #torch.cuda.empty_cache()
                grayscale_cam, logits_per_image, attn_weight_last = cam(input_tensor=input_tensor,
                                                                                   targets=targets,
                                                                                   target_size=None)  # (ori_width, ori_height))

                grayscale_cam = grayscale_cam[0, :]

                grayscale_cam_highres = cv2.resize(grayscale_cam, (ori_width, ori_height))
                highres_cam_to_save.append(torch.tensor(grayscale_cam_highres))

                if idx == 0:
                    attn_weight_list.append(attn_weight_last)
                    attn_weight = [aw[:, 1:, 1:] for aw in attn_weight_list]  # (b, hxw, hxw)
                    attn_weight = torch.stack(attn_weight, dim=0)[-8:]
                    attn_weight = torch.mean(attn_weight, dim=0)
                    attn_weight = attn_weight[0].cpu().detach()
                attn_weight = attn_weight.float()

                box, cnt = scoremap2bbox(scoremap=grayscale_cam, threshold=0.7, multi_contour_eval=True)
                aff_mask = torch.zeros((grayscale_cam.shape[0],grayscale_cam.shape[1]))
                for i_ in range(cnt):
                    x0_, y0_, x1_, y1_ = box[i_]
                    aff_mask[y0_:y1_, x0_:x1_] = 1

                aff_mask = aff_mask.view(1,grayscale_cam.shape[0] * grayscale_cam.shape[1])
                aff_mat = attn_weight

                trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
                trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)

                for _ in range(2):
                    trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
                    trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
                trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2

                for _ in range(1):
                    trans_mat = torch.matmul(trans_mat, trans_mat)

                trans_mat = trans_mat * aff_mask

                cam_to_refine = torch.FloatTensor(grayscale_cam)
                cam_to_refine = cam_to_refine.view(-1,1)

                # (n,n) * (n,1)->(n,1)
                cam_refined = torch.matmul(trans_mat, cam_to_refine).reshape(h //16, w // 16)
                cam_refined = cam_refined.cpu().numpy().astype(np.float32)
                cam_refined_highres = scale_cam_image([cam_refined], (ori_width, ori_height))[0]
                refined_cam_to_save.append(torch.tensor(cam_refined_highres))

            keys = torch.tensor(keys)
            #cam_all_scales.append(torch.stack(cam_to_save,dim=0))
            highres_cam_all_scales.append(torch.stack(highres_cam_to_save,dim=0))
            refined_cam_all_scales.append(torch.stack(refined_cam_to_save,dim=0))


        #cam_all_scales = cam_all_scales[0]
        highres_cam_all_scales = highres_cam_all_scales[0]
        refined_cam_all_scales = refined_cam_all_scales[0]
        np.save(os.path.join(args.cam_out_dir, im.replace('jpg', 'npy')),
                {"keys": keys.numpy(),
                # "strided_cam": cam_per_scales.cpu().numpy(),
                #"highres": highres_cam_all_scales.cpu().numpy().astype(np.float16),
                "attn_highres": refined_cam_all_scales.cpu().numpy().astype(np.float16),
                })
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--img_root', type=str, default='/home/xxx/datasets/COCO2014/JPEGImages/train2014')
    parser.add_argument('--split_file', type=str, default='./coco14/train.txt')
    parser.add_argument('--cam_out_dir', type=str, default='./final/ablation/coco_test')
    parser.add_argument('--model', type=str, default='/home/xxx/pretrained_models/clip/ViT-B-16.pt')
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    file_list = tuple(open(args.split_file, "r"))
    file_list = [id_.rstrip().split(" ") for id_ in file_list]
    train_list = [x[0] + '.jpg' for x in file_list]  # [2000:]
    all_label_list = [x[1:] for x in file_list]  # [2000:]
    print(len(train_list))

    if not os.path.exists(args.cam_out_dir):
        os.makedirs(args.cam_out_dir)

    model, _ = clip.load(args.model, device=device)
    bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY_COCO, ['a clean origami {}.'], model)#['a rendering of a weird {}.'], model)
    fg_text_features = zeroshot_classifier(new_class_names_coco, ['a clean origami {}.'], model)#['a rendering of a weird {}.'], model)

    target_layers = [model.visual.transformer.resblocks[-1].ln_1]
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    dataset_list, split_label_list = split_dataset(train_list, all_label_list, n_splits=args.num_workers)
    if args.num_workers == 1:
        perform(0, dataset_list, args, model, bg_text_features, fg_text_features, cam, split_label_list)
    else:
        multiprocessing.spawn(perform, nprocs=args.num_workers,
                              args=(dataset_list, args, model, bg_text_features, fg_text_features, cam, split_label_list))
