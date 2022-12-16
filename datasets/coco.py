
import numpy as np
import torch
from torch.utils.data import Dataset
import os.path
import imageio
from misc import imutils
import random

import datasets.transforms as tf
from PIL import Image, ImagePalette
from .utils import colormap

SLURM_JOB_ID=os.getenv('SLURM_JOB_ID')
SLURM_JOB_USER=os.getenv('SLURM_JOB_USER')
#dir_name="voc_aug/"
#tmpFolder="/ssd/{}/{}/{}".format(SLURM_JOB_USER,SLURM_JOB_ID,dir_name)
tmpFolder='/home/10102009/coco/'

IMG_FOLDER_NAME ="images" #"JPEGImages"
ANNOT_FOLDER_NAME ="coco_seg_anno"  #"Annotations"
IGNORE = 255

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


cmap = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 0.0, 1.0), (0.0, 0.25, 0.0), (0.0, 0.25, 0.5),
                   (0.0, 0.25, 1.0),
                   (0.0, 0.5, 0.0), (0.0, 0.5, 0.25), (0.0, 0.5, 0.5), (0.0, 0.75, 0.0), (0.0, 0.75, 0.25),
                   (0.0, 0.75, 0.5),
                   (0.0, 0.75, 0.75), (0.0, 0.75, 1.0), (0.0, 1.0, 0.25), (0.25, 0.0, 0.0), (0.25, 0.0, 0.25),
                   (0.25, 0.0, 0.5),
                   (0.25, 0.0, 1.0), (0.25, 0.25, 0.0), (0.25, 0.25, 0.5), (0.25, 0.25, 1.0), (0.25, 0.5, 0.0),
                   (0.25, 0.5, 0.25),
                   (0.25, 0.5, 1.0), (0.25, 0.75, 0.0), (0.25, 0.75, 0.25), (0.25, 0.75, 0.5), (0.25, 1.0, 0.0),
                   (0.25, 1.0, 0.75),
                   (0.5, 0.0, 0.0), (0.5, 0.0, 0.25), (0.5, 0.0, 0.5), (0.5, 0.0, 0.75), (0.5, 0.25, 0.0),
                   (0.5, 0.25, 1.0),
                   (0.5, 0.5, 0.0), (0.5, 0.5, 0.25), (0.5, 0.5, 0.5), (0.5, 0.5, 0.75), (0.5, 0.75, 0.0),
                   (0.5, 0.75, 0.5),
                   (0.5, 0.75, 0.75), (0.5, 1.0, 0.0), (0.5, 1.0, 0.25), (0.5, 1.0, 0.5), (0.5, 1.0, 1.0),
                   (0.75, 0.0, 0.0),
                   (0.75, 0.0, 0.25), (0.75, 0.0, 1.0), (0.75, 0.25, 0.0), (0.75, 0.25, 1.0), (0.75, 0.5, 0.0),
                   (0.75, 0.5, 0.25),
                   (0.75, 0.5, 0.5), (0.75, 0.5, 0.75), (0.75, 0.5, 1.0), (0.75, 0.75, 0.0), (0.75, 0.75, 0.25),
                   (0.75, 0.75, 0.5),
                   (0.75, 0.75, 0.75), (0.75, 0.75, 1.0), (0.75, 1.0, 0.0), (0.75, 1.0, 0.25), (0.75, 1.0, 0.5),
                   (0.75, 1.0, 0.75),
                   (0.75, 1.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.5), (1.0, 0.25, 0.25), (1.0, 0.25, 0.5),
                   (1.0, 0.25, 0.75),
                   (1.0, 0.25, 1.0), (1.0, 0.5, 0.0), (1.0, 0.5, 0.25), (1.0, 0.5, 0.5), (1.0, 0.5, 0.75),
                   (1.0, 0.5, 1.0), (1.0, 0.75, 0.25), (1.0, 0.75, 0.5), (1.0, 0.75, 0.75)]
palette_new = []

for p in cmap:
    for pp in p:
        palette_new.append(int(pp * 255))



cls_labels_dict = np.load('datasets/coco14/cls_labels_coco.npy', allow_pickle=True).item()

def decode_int_filename(int_filename):
    # s = str(int(int_filename))

    s = str(int_filename).split('\n')[0]
    if len(s) != 12:
        s = '%012d' % int(s)
    return s


def load_image_label_list_from_npy(img_name_list):
    # print(img_name_list)
    # print(img_name_list[0])
    # print(cls_labels_dict[int(img_name_list[0])])
    return np.array([cls_labels_dict[int(img_name)] for img_name in img_name_list])

def get_img_path(img_name, voc12_root,sub_path):
    if not isinstance(img_name, str):
        img_name = decode_int_filename(img_name)
    return os.path.join(voc12_root, IMG_FOLDER_NAME, sub_path, 'COCO_{}_{}.jpg'.format(sub_path,img_name))

def get_anno_path(img_name, voc12_root):
    if not isinstance(img_name, str):
        img_name = decode_int_filename(img_name)
    return os.path.join(voc12_root, ANNOT_FOLDER_NAME, img_name + '.png')

def load_img_name_list(dataset_path):

    img_name_list = np.loadtxt(dataset_path, dtype=np.int32)
    img_name_list = img_name_list[::-1]

    return img_name_list

class COCO14ImageDataset(Dataset):

    CLASSES = [
        'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
        'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'ambiguous'
    ]

    NUM_CLASS = 81


    def __init__(self):
        super().__init__()
        self._init_palette()

    def _init_palette(self):
        """self.cmap = colormap()
        self.palette = ImagePalette.ImagePalette()
        for rgb in self.cmap:
            self.palette.getcolor(rgb)"""
        self.palette = palette_new

    def get_palette(self):
        return self.palette

    def denorm(self, image):

        if image.dim() == 3:
            assert image.dim() == 3, "Expected image [CxHxW]"
            assert image.size(0) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip(image, MEAN, STD):
                t.mul_(s).add_(m)
        elif image.dim() == 4:
            # batch mode
            assert image.size(1) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip((0, 1, 2), MEAN, STD):
                image[:, t, :, :].mul_(s).add_(m)

        return image

class TorchvisionNormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img




class COCO14Segmentation(COCO14ImageDataset):

    def __init__(self, cfg, split, test_mode,scales=(1.0,), coco14_root=os.path.expanduser('./data/coco')):

        self.split = split
        self.cfg = cfg
        self.scales = scales
        self.test_mode = test_mode
        self.coco14_root = tmpFolder #coco14_root

        if self.split == 'train':
            self.img_name_list_path="datasets/coco14/train14_new.txt"
            self.sub_path="train2014"
        elif self.split == 'train_voc':
            self.img_name_list_path="datasets/coco14/val14_new.txt"
            self.sub_path = "val2014"
        elif self.split == 'val':
            self.img_name_list_path="datasets/coco14/val14_new.txt"
            self.sub_path = "val2014"
        elif self.split == 'test':
            self.img_name_list_path="datasets/coco14/test14.txt"
            self.sub_path = "test2014"
        else:
            raise RuntimeError('Unknown dataset split.')

        super(COCO14Segmentation, self).__init__() #cfg, self.img_name_list_path, self.coco14_root, self.sub_path

        self.images = []
        self.masks = []
        # print(os.getcwd())
        # datasets_root='/home/lenovo/PycharmProjects/SSSS/1-stage-wseg-master/voc_aug/'

        with open(self.img_name_list_path, "r") as lines:
            for line in lines:
                _image = line.strip("\n") #.split(' ')
                _image = get_img_path(_image,self.coco14_root,self.sub_path)
                assert os.path.isfile(_image), '%s not found' % _image
                self.images.append(_image)

                if self.split != 'test':
                    _mask = get_anno_path(line.strip("\n"),self.coco14_root)
                    assert os.path.isfile(_mask), '%s not found' % _mask
                    self.masks.append(_mask)

        if self.split != 'test':
            assert (len(self.images) == len(self.masks))
            if self.split == 'train':
                assert len(self.images) == 82081 #82783-702
            elif self.split == 'val':
                assert len(self.images) == 40137     #40504-367


        #assert os.path.isfile(self.img_name_list_path), "%s not found" % self.img_name_list_path
        self.transform = tf.Compose([tf.MaskRandResizedCrop(self.cfg.DATASET), \
                                     tf.MaskHFlip(), \
                                     tf.MaskColourJitter(p=1.0), \
                                     tf.MaskNormalise(MEAN, STD), \
                                     tf.MaskToTensor()])


        #self.label_list = load_image_label_list_from_npy(self.img_name_list)

    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        #name = self.img_name_list[idx]
        #name_str = decode_int_filename(name)

        # image = Image.open(get_img_path(name_str, self.coco14_root,self.sub_path)).convert('RGB')
        # mask = Image.open(get_anno_path(name_str, self.coco14_root))
        image = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx])

        unique_labels = np.unique(mask)

        # ambigious
        if unique_labels[-1] == 255:
            unique_labels = unique_labels[:-1]

        # ignoring BG
        labels = torch.zeros(self.NUM_CLASS - 1)
        if unique_labels[0] == 0:
            unique_labels = unique_labels[1:]
        unique_labels -= 1

        assert unique_labels.size > 0, 'No labels found in %s' % self.masks[idx]
        for i in unique_labels:
            labels[i] = 1
        # labels[unique_labels] = 1

        image, mask = self.transform(image, mask)

        #return {'name': name_str, 'img': img}
        return image, labels, os.path.basename(self.images[idx])


    @property
    def pred_offset(self):
        return 0


