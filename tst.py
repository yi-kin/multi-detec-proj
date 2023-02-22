import json
import os

import torch
from PIL import Image
from pycocotools.coco import COCO

from train_utils import convert_coco_poly_mask

# index = 1
# coco = COCO("D:\deeplearning\deep-learning-for-image-processing-master\pytorch_object_detection\mask_rcnn\data\openfrenic\Train_poly.json")
# ids = list(sorted(coco.imgs.keys()))
# img_id = ids[index]
# ann_ids = coco.getAnnIds(imgIds=img_id)
#
# coco_target = coco.loadAnns(ann_ids)
# path = coco.loadImgs(img_id)[0]['file_name']
#
#
#
#
#
# boxes = [obj["bbox"] for obj in coco_target]
# boxes = torch.as_tensor(boxes,dtype=torch.float32).reshape(-1,4)
# boxes[:, 2:] += boxes[:, :2]
# boxes[:, 0::2].clamp_(min=0, max=w)
# boxes[:, 1::2].clamp_(min=0, max=h)
#
# target={}
# classes = [obj["category_id"] for obj in coco_target]
# classes = torch.tensor(classes, dtype=torch.int64)
#
# area = torch.tensor([obj["area"] for obj in coco_target])
# iscrowd = torch.tensor([obj["iscrowd"] for obj in coco_target])
# segmentations = [obj["segmentation"] for obj in coco_target]
# masks = convert_coco_poly_mask(segmentations, w, h)
#
# keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
# boxes = boxes[keep]
# classes = classes[keep]
# masks = masks[keep]
# area = area[keep]
# iscrowd = iscrowd[keep]
#
# target["boxes"] = boxes
# target["labels"] = classes
# target["masks"] = masks
# target["image_id"] = torch.tensor([img_id])
#
# # for conversion to coco api
# target["area"] = area
# target["iscrowd"] = iscrowd
#
# print("e")

if __name__ == '__main__':
    x=[]
    w=[]
    a = os.listdir("data/openfrenic/Images/Train")
    for i in range(len(a)):
        b = a[i].split('.')[0]
        x.append(b)
    with open('data/openfrenic/Train_poly.json','r') as f:
        json_content = json.load(f)
    q=0
    for i in range(len(json_content['images'])):
        m = json_content['images'][q]['file_name'].split('/')[-1].split('.')[0]
        n = json_content['images'][q]['id']
        if (m not in x) or (i%100 != 0):
            del json_content['images'][q]
        else:
            q+=1
            w.append(n)
    p=0
    for i in range(len(json_content['annotations'])):
        m = json_content['annotations'][p]['image_id']
        if m not in w:
            del  json_content['annotations'][p]
        else:
            p=p+1
    # for i in range(len(json_content['images'])):

    json_str = json.dumps(json_content)
    with open('./data/openfrenic/test_Train.json', 'w') as json_file:
        json_file.write(json_str)







