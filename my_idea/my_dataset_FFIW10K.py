import os

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from train_utils import convert_to_coco_api, convert_coco_poly_mask


class FFIW(Dataset):
    def __init__(self,root,dataset='Train',transform=None):
        super(FFIW, self).__init__()

        self.image_dir = os.path.join(root,f"{dataset}")
        self.anno_path = os.path.join(root,f"{dataset}_poly.json")
        self.transforms = transform
        self.coco = COCO(self.anno_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        if dataset=='Train':
            self.mode = 'train'
        else:
            self.mode = 'val'


    def parse_target(self,img_id,image_related_targets,w,h):
        image_related_targets = [obj for obj in image_related_targets if obj['iscrowd']==0]

        boxes = [obj["bbox"] for obj in image_related_targets]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # boxes[:, 2:] += boxes[:, :2]
        # boxes[:, 0::2].clamp_(min=0, max=w)
        # boxes[:, 1::2].clamp_(min=0, max=h)

        target = {}
        classes = [obj["category_id"] for obj in image_related_targets]
        classes = torch.tensor(classes, dtype=torch.int64)

        # area = torch.tensor([obj["area"] for obj in image_related_targets])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in image_related_targets])
        # segmentations = [obj["segmentation"] for obj in image_related_targets]
        # masks = convert_coco_poly_mask(segmentations, h, w)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        # masks = masks[keep]
        # area = area[keep]
        iscrowd = iscrowd[keep]

        target["boxes"] = boxes
        target["labels"] = classes
        # target["masks"] = masks
        target["image_id"] = torch.tensor([img_id])

        # for conversion to coco api
        # target["area"] = area
        target["iscrowd"] = iscrowd

        return target

    def __getitem__(self, idx):
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        image_related_targets = self.coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.image_dir, path)).convert('RGB')


        w,h = img.size
        target = self.parse_target(img_id,image_related_targets,w,h)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img,target

    def __len__(self):
        return len(self.ids)

    def get_height_and_width(self, index):
        coco = self.coco
        img_id = self.ids[index]

        img_info = coco.loadImgs(img_id)[0]
        w = img_info["width"]
        h = img_info["height"]
        return h, w

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))