"""
该脚本用于调用训练好的模型权重去计算验证集/测试集的COCO指标
以及每个类别的mAP(IoU=0.5)
"""

import os
import json

import torch
from tqdm import tqdm
import numpy as np
import PIL
import transforms
from backbone import resnet50_fpn_backbone
from my_dataset_openfrenic import OpenForensics
from network_files import MaskRCNN
#from my_dataset_coco import CocoDetection
#from my_dataset_voc import VOCInstances
from train_utils import EvalCOCOMetric
#Sbi导入的文件
from model import Detector
from preprocess import extract_frames
from datasets import *
from PIL import Image
from torchvision.transforms import Resize
from sklearn.metrics import confusion_matrix, roc_auc_score

from network_files import boxes as box_ops

def summarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 12, [""] * 12
    stats[0], print_list[0] = _summarize(1)
    stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
    stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
    stats[3], print_list[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
    stats[4], print_list[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[5], print_list[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
    stats[6], print_list[6] = _summarize(0, maxDets=self.params.maxDets[0])
    stats[7], print_list[7] = _summarize(0, maxDets=self.params.maxDets[1])
    stats[8], print_list[8] = _summarize(0, maxDets=self.params.maxDets[2])
    stats[9], print_list[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
    stats[10], print_list[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[11], print_list[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])

    print_info = "\n".join(print_list)

    if not self.eval:
        raise Exception('Please run accumulate() first')

    return stats, print_info

def del_tensor_ele(arr,index):
    arr1 = arr[0:index]

    return arr1


def save_info(coco_evaluator,
              category_index: dict,
              save_name: str = "record_mAP.txt"):
    iou_type = coco_evaluator.params.iouType
    print(f"IoU metric: {iou_type}")
    # calculate COCO info for all classes
    coco_stats, print_coco = summarize(coco_evaluator)

    # calculate voc info for every classes(IoU=0.5)
    classes = [v for v in category_index.values() if v != "N/A"]
    voc_map_info_list = []
    for i in range(len(classes)):
        stats, _ = summarize(coco_evaluator, catId=i)
        voc_map_info_list.append(" {:15}: {}".format(classes[i], stats[1]))

    print_voc = "\n".join(voc_map_info_list)
    print(print_voc)

    # 将验证结果保存至txt文件中
    with open(save_name, "w") as f:
        record_lines = ["COCO results:",
                        print_coco,
                        "",
                        "mAP(IoU=0.5) for each category:",
                        print_voc]
        f.write("\n".join(record_lines))


def main(parser_data):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    data_transform = {
        "val": transforms.Compose([transforms.ToTensor()])
    }

    # read class_indict
    label_json_path = parser_data.label_json_path
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        category_index = json.load(f)

    data_root = parser_data.data_path

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = parser_data.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    # load validation data set
    #val_dataset = CocoDetection(data_root, "Val", data_transform["val"])
    #val_dataset = OpenForensics(data_root,dataset='Val',transform=data_transform["val"])
    val_dataset = OpenForensics(data_root,dataset='Test-Challenge',transform=data_transform["val"])
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    # val_dataset = VOCInstances(data_root, year="2012", txt_name="val.txt", transforms=data_transform["val"])
    val_dataset_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     pin_memory=True,
                                                     num_workers=nw,
                                                     collate_fn=val_dataset.collate_fn)

    # create model
    backbone = resnet50_fpn_backbone()
    model = MaskRCNN(backbone, num_classes=args.num_classes + 1)

    # 载入你自己训练好的模型权重
    weights_path = parser_data.weights_path
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    # print(model)

    model.to(device)

    # evaluate on the val dataset
    cpu_device = torch.device("cpu")

    # det_metric = EvalCOCOMetric(val_dataset.coco, "bbox", "det_results.json")
    # seg_metric = EvalCOCOMetric(val_dataset.coco, "segm", "seg_results.json")
    model.eval()

#SBImodel
    model1 = Detector()
    model1 = model1.to(device)
    cnn_sd = torch.load('./fine_tune_fu.tar', map_location="cpu")["model"]
   # cnn_sd = torch.load('./FFraw.tar', map_location="cpu")["model"]

    model1.load_state_dict(cnn_sd)
    model1.eval()
    #sbi
    target_list = []
    output_list = []

    with torch.no_grad():
        for image, targets in tqdm(val_dataset_loader, desc="validation..."):
            # 将图片传入指定设备device
            image = list(img.to(device) for img in image)

            # inference
            outputs = model(image)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

            # 正负样本匹配
            # dtype = outputs[0].dtype
            # device = outputs[0].device

            gt_boxes = [t["boxes"] for t in targets]
            gt_labels = [t["labels"] for t in targets]
            outputs_ = [t["boxes"] for t in outputs]
            gt_boxes_after = []
            gt_labels_after = []
            outputs_after = []
            for outputs_in_image, gt_boxes_in_image, gt_labels_in_image in zip(outputs_, gt_boxes, gt_labels):
                if (len(gt_labels_in_image) >= len(outputs_in_image)):
                    match_quality_matrix = box_ops.box_iou(outputs_in_image, gt_boxes_in_image)
                    _, indices = match_quality_matrix.max(dim=1)
                    gt_boxes_in_image = gt_boxes_in_image[indices]
                    gt_labels_in_image = gt_labels_in_image[indices]
                    gt_boxes_after.append(gt_boxes_in_image)
                    gt_labels_after.append(gt_labels_in_image)
                    outputs_after.append(outputs_in_image)
                else:
                    match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, outputs_in_image)
                    _, indices = match_quality_matrix.max(dim=1)
                    outputs_in_image = outputs_in_image[indices]
                    gt_boxes_after.append(gt_boxes_in_image)
                    gt_labels_after.append(gt_labels_in_image)
                    outputs_after.append(outputs_in_image)


            #这是为了让targets和outputs数量相同
            # for i in range(2):
            #     num_targets = len(targets[i]['labels'])
            #     num_outputs = len(outputs[i]['labels'])
            #     if(num_outputs>num_targets):
            #         for j in range(num_outputs-num_targets):
            #             outputs[i]['boxes'] = del_tensor_ele(outputs[i]['boxes'],-1)
            #             outputs[i]['labels'] = del_tensor_ele(outputs[i]['labels'], -1)
            #             # del outputs[i]['labels'][-1]
            #     elif(num_outputs<num_targets):
            #         for j in range(num_targets-num_outputs):
            #             targets[i]['boxes'] = del_tensor_ele(targets[i]['boxes'], -1)
            #             targets[i]['labels'] = del_tensor_ele(targets[i]['labels'], -1)
            #             # del targets[i]['boxes'][-1]
            #             # del targets[i]['labels'][-1]

            face_imgs = []
            #将图片中的人脸剪裁出来，准备放入SBI模型中
            for i in range(len(outputs_after)):
                coordinates = outputs_after[i]
                coordinates = coordinates.round().long()
                for j in range(len(coordinates)):
                    # coordinates = coordinates.round().long()
                    #coordinates_size = (coordinates[j][3]-coordinates[j][1],coordinates[j][2]-coordinates[j][0])
                    face_img = image[i][:,coordinates[j][1]:coordinates[j][3],coordinates[j][0]:coordinates[j][2]]
                    face_img = face_img.unsqueeze(dim=0)
                    face_img =  torch.nn.functional.interpolate(face_img, size=380, mode='bilinear', align_corners=False)
                    face_imgs.append(face_img)
            if(len(face_imgs)>0):
                face_input = torch.cat(face_imgs,dim=0)
                pred = model1(face_input).softmax(1)[:, 1]
                pred = pred.to('cpu').numpy()
                output_list.extend(pred)
            else:
                continue
            # 将每个人脸的labels放进列表
            for i in range(len(gt_labels_after)):
                # temp_list=targets[i]['labels']
                temp_list = gt_labels_after[i]
                for j in range(len(temp_list)):
                    target_list.append(temp_list[j])




            # det_metric.update(targets, outputs)
            # seg_metric.update(targets, outputs)
    target_list = [*map(lambda x: x - 1, target_list)]
    auc = roc_auc_score(target_list, output_list)
    print(f'openfor | AUC: {auc:.4f}')
    # det_metric.synchronize_results()
    # seg_metric.synchronize_results()
    # det_metric.evaluate()
    # seg_metric.evaluate()
    #
    # save_info(det_metric.coco_evaluator, category_index, "det_record_mAP.txt")
    # save_info(seg_metric.coco_evaluator, category_index, "seg_record_mAP.txt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 使用设备类型
    parser.add_argument('--device', default='cuda', help='device')

    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-classes', type=int, default=2, help='number of classes')

    # 数据集的根目录
    parser.add_argument('--data-path', default='../data/openfrenic', help='dataset root')

    # 训练好的权重文件
    parser.add_argument('--weights-path', default='../multi_train/model_11.pth', type=str, help='training weights')

    # batch size(set to 1, don't change)
    parser.add_argument('--batch-size', default=2, type=int, metavar='N',
                        help='batch size when validation.')
    # 类别索引和类别名称对应关系
    parser.add_argument('--label-json-path', type=str, default="./test_open.json")

    args = parser.parse_args()

    main(args)
