"""
此文件是用于验证xception模型在openforeniscs和FFIW10K上的准确率
注意，这个xception是在Imagenet上进行预训练的，没有进行任何迁移
这个文件可以进行迁移学习后的模型准确度验证，如xception在FFIW和Openfor上面进行迁移后在测试集上进行验证(主要)
"""

import os
import json

import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import PIL

import network_files
import transforms
from backbone import resnet50_fpn_backbone
from my_dataset_openfrenic import OpenForensics
from my_dataset_FFIW10K import FFIW

from network_files import MaskRCNN
# from my_dataset_coco import CocoDetection
# from my_dataset_voc import VOCInstances
# from train_utils import EvalCOCOMetric
#Sbi导入的文件
from xception import xception
# from PIL import Image
# from torchvision.transforms import Resize
from sklearn.metrics import confusion_matrix, roc_auc_score

from network_files import boxes as box_ops




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

    # data_root = parser_data.data_path
    data_root2 = "../data/openfrenic"

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = parser_data.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    # load validation data set

    # FFIW_Test_dataset = FFIW(data_root,dataset='Test',transform=data_transform["val"])
    open_val_dataset = OpenForensics(data_root2, dataset='Val', transform=data_transform["val"])
    open_test_dataset = OpenForensics(data_root2, dataset='Test-Dev', transform=data_transform["val"])
    open_challenge_dataset = OpenForensics(data_root2, dataset='Test-Challenge', transform=data_transform["val"])


    # FFIW_Test_dataset_loader = torch.utils.data.DataLoader(FFIW_Test_dataset,
    #                                                  batch_size=batch_size,
    #                                                  shuffle=False,
    #                                                  pin_memory=True,
    #                                                  num_workers=nw,
    #                                                  collate_fn=FFIW_Test_dataset.collate_fn)
    open_val_dataset_loader = torch.utils.data.DataLoader(open_val_dataset,
                                                           batch_size=batch_size,
                                                           shuffle=False,
                                                           pin_memory=True,
                                                           num_workers=nw,
                                                           collate_fn=open_val_dataset.collate_fn)
    open_test_dataset_loader = torch.utils.data.DataLoader(open_test_dataset,
                                                          batch_size=batch_size,
                                                          shuffle=False,
                                                          pin_memory=True,
                                                          num_workers=nw,
                                                          collate_fn=open_test_dataset.collate_fn)
    open_challenge_dataset_loader = torch.utils.data.DataLoader(open_challenge_dataset,
                                                          batch_size=batch_size,
                                                          shuffle=False,
                                                          pin_memory=True,
                                                          num_workers=nw,
                                                          collate_fn=open_challenge_dataset.collate_fn)

    # create model
    backbone = resnet50_fpn_backbone()
    model = MaskRCNN(backbone, num_classes=args.num_classes + 1)
    # 载入你自己训练好的模型权重
    weights_path = parser_data.weights_path
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)
    # evaluate on the val dataset
    cpu_device = torch.device("cpu")
    model.eval()

#Xception-model
    model1 = xception()
    model1.net.fc = nn.Linear(model1.net.fc.in_features, 2)
    # nn.init.xavier_uniform_(model1.net.fc.weight)
    # cnn_sd = torch.load('FFIW：3epoch-0.8308557893871815.pth', map_location="cpu")
    # cnn_sd = torch.load('Openfor：1epoch-0.9668582683719695.pth', map_location="cpu")
    cnn_sd = torch.load('pre_trained75.tar', map_location="cpu")["model"]
    model1.load_state_dict(cnn_sd)

    model1 = model1.to(device)
    model1.eval()
    #sbi
    target_list = []
    output_list = []

    with torch.no_grad():
        # for image, targets in tqdm(FFIW_Test_dataset_loader, desc="validation..."):
        #     # 将图片传入指定设备device
        #     image = list(img.to(device) for img in image)
        #
        #     # inference
        #     outputs = model(image)
        #
        #     outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        #
        #     gt_boxes = [t["boxes"] for t in targets]
        #     gt_labels = [t["labels"] for t in targets]
        #     outputs_ = [t["boxes"] for t in outputs]
        #     gt_boxes_after = []
        #     gt_labels_after = []
        #     outputs_after = []
        #     for outputs_in_image,gt_boxes_in_image, gt_labels_in_image in zip(outputs_,gt_boxes, gt_labels):
        #         if(len(gt_labels_in_image) >= len(outputs_in_image)):
        #             match_quality_matrix = box_ops.box_iou(outputs_in_image,gt_boxes_in_image)
        #             _, indices = match_quality_matrix.max(dim=1)
        #             gt_boxes_in_image = gt_boxes_in_image[indices]
        #             gt_labels_in_image = gt_labels_in_image[indices]
        #             gt_boxes_after.append(gt_boxes_in_image)
        #             gt_labels_after.append(gt_labels_in_image)
        #             outputs_after.append(outputs_in_image)
        #         else:
        #             match_quality_matrix = box_ops.box_iou(gt_boxes_in_image,outputs_in_image)
        #             _, indices = match_quality_matrix.max(dim=1)
        #             outputs_in_image = outputs_in_image[indices]
        #             gt_boxes_after.append(gt_boxes_in_image)
        #             gt_labels_after.append(gt_labels_in_image)
        #             outputs_after.append(outputs_in_image)
        #
        #
        #     face_imgs = []
        #     #将图片中的人脸剪裁出来，准备放入SBI模型中
        #     for i in range(len(outputs_after)):
        #         #coordinates = outputs[i]['boxes']
        #         coordinates = outputs_after[i]
        #         coordinates = coordinates.round().long()
        #         for j in range(len(coordinates)):
        #             # coordinates = coordinates.round().long()
        #             #coordinates_size = (coordinates[j][3]-coordinates[j][1],coordinates[j][2]-coordinates[j][0])
        #             face_img = image[i][:,coordinates[j][1]:coordinates[j][3],coordinates[j][0]:coordinates[j][2]]
        #             face_img = face_img.unsqueeze(dim=0)
        #             face_img =  torch.nn.functional.interpolate(face_img, size=380, mode='bilinear', align_corners=False)
        #             face_imgs.append(face_img)
        #
        #
        #         if len(face_imgs)>0:
        #             face_imgs = torch.cat(face_imgs, dim=0)  # 5,3,380,380
        #             pred,_ = model1(face_imgs)
        #             pred = pred.softmax(1)[:, 1].to('cpu').numpy()
        #
        #             output_list.extend(pred)
        #             face_imgs = []
        #         else:
        #             continue
        #
        #
        #     #将每个人脸的labels放进列表
        #     for i in range(len(gt_labels_after)):
        #         #temp_list=targets[i]['labels']
        #         temp_list=gt_labels_after[i]
        #         for j in range(len(temp_list)):
        #             target_list.append(temp_list[j])
        #
        #
        # target_list = [*map(lambda x: x - 1, target_list)]
        # auc = roc_auc_score(target_list, output_list)
        # print(f'FFIW | xception-AUC: {auc:.4f}')
    #############################################  val
        target_list = []
        output_list = []
        for image, targets in tqdm(open_val_dataset_loader, desc="validation..."):
            # 将图片传入指定设备device
            image = list(img.to(device) for img in image)

            # inference
            outputs = model(image)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]


            gt_boxes = [t["boxes"] for t in targets]
            gt_labels = [t["labels"] for t in targets]
            outputs_ = [t["boxes"] for t in outputs]
            gt_boxes_after = []
            gt_labels_after = []
            outputs_after = []
            for outputs_in_image,gt_boxes_in_image, gt_labels_in_image in zip(outputs_,gt_boxes, gt_labels):
                if(len(gt_labels_in_image) >= len(outputs_in_image)):
                    match_quality_matrix = box_ops.box_iou(outputs_in_image,gt_boxes_in_image)
                    _, indices = match_quality_matrix.max(dim=1)
                    gt_boxes_in_image = gt_boxes_in_image[indices]
                    gt_labels_in_image = gt_labels_in_image[indices]
                    gt_boxes_after.append(gt_boxes_in_image)
                    gt_labels_after.append(gt_labels_in_image)
                    outputs_after.append(outputs_in_image)
                else:
                    match_quality_matrix = box_ops.box_iou(gt_boxes_in_image,outputs_in_image)
                    _, indices = match_quality_matrix.max(dim=1)
                    outputs_in_image = outputs_in_image[indices]
                    gt_boxes_after.append(gt_boxes_in_image)
                    gt_labels_after.append(gt_labels_in_image)
                    outputs_after.append(outputs_in_image)


            face_imgs = []
            #将图片中的人脸剪裁出来，准备放入SBI模型中
            for i in range(len(outputs_after)):
                #coordinates = outputs[i]['boxes']
                coordinates = outputs_after[i]
                coordinates = coordinates.round().long()
                for j in range(len(coordinates)):
                    # coordinates = coordinates.round().long()
                    #coordinates_size = (coordinates[j][3]-coordinates[j][1],coordinates[j][2]-coordinates[j][0])
                    face_img = image[i][:,coordinates[j][1]:coordinates[j][3],coordinates[j][0]:coordinates[j][2]]
                    face_img = face_img.unsqueeze(dim=0)
                    face_img =  torch.nn.functional.interpolate(face_img, size=380, mode='bilinear', align_corners=False)
                    face_imgs.append(face_img)


                if len(face_imgs)>0:
                    face_imgs = torch.cat(face_imgs, dim=0)  # 5,3,380,380
                    pred,_ = model1(face_imgs)
                    pred = pred.softmax(1)[:, 1].to('cpu').numpy()

                    output_list.extend(pred)
                    face_imgs = []
                else:
                    continue


            #将每个人脸的labels放进列表
            for i in range(len(gt_labels_after)):
                #temp_list=targets[i]['labels']
                temp_list=gt_labels_after[i]
                for j in range(len(temp_list)):
                    target_list.append(temp_list[j])


        target_list = [*map(lambda x: x - 1, target_list)]
        auc = roc_auc_score(target_list, output_list)
        print(f'openfor | xception-Val- AUC: {auc:.4f}')
        #
        # #############################################  test
        target_list = []
        output_list = []
        for image, targets in tqdm(open_test_dataset_loader, desc="validation..."):
            # 将图片传入指定设备device
            image = list(img.to(device) for img in image)

            # inference
            outputs = model(image)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]


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


            face_imgs = []
            # 将图片中的人脸剪裁出来，准备放入SBI模型中
            for i in range(len(outputs_after)):
                # coordinates = outputs[i]['boxes']
                coordinates = outputs_after[i]
                coordinates = coordinates.round().long()
                for j in range(len(coordinates)):
                    # coordinates = coordinates.round().long()
                    # coordinates_size = (coordinates[j][3]-coordinates[j][1],coordinates[j][2]-coordinates[j][0])
                    face_img = image[i][:, coordinates[j][1]:coordinates[j][3], coordinates[j][0]:coordinates[j][2]]
                    face_img = face_img.unsqueeze(dim=0)
                    face_img = torch.nn.functional.interpolate(face_img, size=380, mode='bilinear', align_corners=False)
                    face_imgs.append(face_img)

                if len(face_imgs) > 0:
                    face_imgs = torch.cat(face_imgs, dim=0)  # 5,3,380,380
                    pred, _ = model1(face_imgs)
                    pred = pred.softmax(1)[:, 1].to('cpu').numpy()

                    output_list.extend(pred)
                    face_imgs = []
                else:
                    continue

            # 将每个人脸的labels放进列表
            for i in range(len(gt_labels_after)):
                # temp_list=targets[i]['labels']
                temp_list = gt_labels_after[i]
                for j in range(len(temp_list)):
                    target_list.append(temp_list[j])

        target_list = [*map(lambda x: x - 1, target_list)]
        auc = roc_auc_score(target_list, output_list)
        print(f'openfor | xception-test- AUC: {auc:.4f}')
        # #############################################  cha
        target_list = []
        output_list = []
        for image, targets in tqdm(open_challenge_dataset_loader, desc="validation..."):
            # 将图片传入指定设备device
            image = list(img.to(device) for img in image)

            # inference
            outputs = model(image)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]


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

            face_imgs = []
            # 将图片中的人脸剪裁出来，准备放入SBI模型中
            for i in range(len(outputs_after)):
                # coordinates = outputs[i]['boxes']
                coordinates = outputs_after[i]
                coordinates = coordinates.round().long()
                for j in range(len(coordinates)):

                    face_img = image[i][:, coordinates[j][1]:coordinates[j][3], coordinates[j][0]:coordinates[j][2]]
                    face_img = face_img.unsqueeze(dim=0)
                    face_img = torch.nn.functional.interpolate(face_img, size=380, mode='bilinear', align_corners=False)
                    face_imgs.append(face_img)

                if len(face_imgs) > 0:
                    face_imgs = torch.cat(face_imgs, dim=0)  # 5,3,380,380
                    pred, _ = model1(face_imgs)
                    pred = pred.softmax(1)[:, 1].to('cpu').numpy()

                    output_list.extend(pred)
                    face_imgs = []
                else:
                    continue

            # 将每个人脸的labels放进列表
            for i in range(len(gt_labels_after)):
                # temp_list=targets[i]['labels']
                temp_list = gt_labels_after[i]
                for j in range(len(temp_list)):
                    target_list.append(temp_list[j])

        target_list = [*map(lambda x: x - 1, target_list)]
        auc = roc_auc_score(target_list, output_list)
        print(f'openfor | xception-Challenge- AUC: {auc:.4f}')
        # torch.save(model1.state_dict(), f'xception-FFIW.pth')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 使用设备类型
    parser.add_argument('--device', default='cuda', help='device')

    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-classes', type=int, default=2, help='number of classes')

    # 数据集的根目录
    parser.add_argument('--data-path', default='../data/FFIW10K', help='dataset root')

    # 训练好的权重文件
    parser.add_argument('--weights-path', default='../multi_train/model_11.pth', type=str, help='training weights')

    # batch size(set to 1, don't change)
    parser.add_argument('--batch-size', default=2, type=int, metavar='N',
                        help='batch size when validation.')
    # 类别索引和类别名称对应关系
    parser.add_argument('--label-json-path', type=str, default="./test_open.json")

    args = parser.parse_args()

    main(args)
