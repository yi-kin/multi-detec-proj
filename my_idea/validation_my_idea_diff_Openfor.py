"""
此脚本是用来训练第一个想法diff的，在训练集上训练后再测试集验证，用于Openfor数据集
"""

import os
import json

import torch
from torch import optim
from tqdm import tqdm
import numpy as np
import PIL

import network_files
import transforms
from backbone import resnet50_fpn_backbone
from my_dataset_openfrenic import OpenForensics
from my_dataset_FFIW10K import FFIW

from network_files import MaskRCNN
#from my_dataset_coco import CocoDetection
#from my_dataset_voc import VOCInstances
#from train_utils import EvalCOCOMetric
#Sbi导入的文件
from xception import xception
#from PIL import Image
from torchvision.transforms import Resize
from sklearn.metrics import confusion_matrix, roc_auc_score

from network_files import boxes as box_ops

import torch.nn as nn
import torch.nn.functional as F
class ClassifierD(nn.Module):
    def __init__(self, xception, output_dim=2):
        super(ClassifierD, self).__init__()
        #########################
        self.xcep = xception
        #############################
        self.linear1 = nn.Linear(4096, 128)
        self.linear2 = nn.Linear(128, output_dim)

        for name,param in self.xcep.net.named_parameters():
            if name not in ["fc.weight", "fc.bias", "bn4.weight", "bn4.bias", 'conv4.conv1.weight', 'conv4.pointwise.weight']:
                param.requires_grad_(False)


    def forward(self, x,y):

        x,mx = self.xcep(x)
        diff = mx-y
        x = torch.cat((mx, diff), dim=1)

        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x


def main(parser_data):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }


    # read class_indict
    label_json_path = parser_data.label_json_path
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        category_index = json.load(f)



    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = parser_data.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)


    #加载数据集
    data_root = parser_data.data_path
    # train_dataset = OpenForensics(data_root, "Train", data_transform["train"])
    # val_dataset = OpenForensics(data_root,dataset='Test-Dev',transform=data_transform["val"])
    # test_dataset = OpenForensics(data_root,dataset='Test-Challenge',transform=data_transform["val"])
    #FFIW
    train_dataset = OpenForensics(data_root, "Train", data_transform["train"])
    val_dataset = OpenForensics(data_root, "Val", data_transform["train"])
    test_dataset = OpenForensics(data_root, dataset='Test-Dev', transform=data_transform["val"])
    test_Challenge_dataset = OpenForensics(data_root, dataset='Test-Challenge', transform=data_transform["val"])


    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    pin_memory=True,
                                                    num_workers=nw,
                                                    collate_fn=train_dataset.collate_fn)

    val_dataset_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     pin_memory=True,
                                                     num_workers=nw,
                                                     collate_fn=val_dataset.collate_fn)
    test_dataset_loader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     pin_memory=True,
                                                     num_workers=nw,
                                                     collate_fn=test_dataset.collate_fn)
    test_challenge_dataset_loader = torch.utils.data.DataLoader(test_Challenge_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      num_workers=nw,
                                                      collate_fn=test_Challenge_dataset.collate_fn)


    # create model（目标检测的模型）################1111#####################
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
    ###############################################1111##########################



#Xception--model############################
    model1 = xception()
    model1.net.fc = nn.Linear(model1.net.fc.in_features, 2)
    nn.init.xavier_uniform_(model1.net.fc.weight)
    #这里是加载在FFIW或Openfor上面迁移训练后的模型参数

    cnn_sd = torch.load('Openfor：1epoch-0.9668582683719695.pth', map_location="cpu")
    model1.load_state_dict(cnn_sd)

    model1.net.num_classes = 2
    model1 = model1.to(device)
    model1.train()

    n_epoch = args.epoch
#############################################
    model_cls = ClassifierD(model1)

    model_cls.train()
    ############resume

    model_cls.to(device)
    optimizer = optim.Adam(model_cls.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=10e-5)
    criterion = torch.nn.CrossEntropyLoss(size_average=True)


########################################
    for epoch in range(n_epoch):
        output_list = []
        target_list = []
        train_loss = 0.

        model1.train()
        model_cls.train()

        print(f'---epoch:{epoch}----')
        for image, targets in tqdm(train_data_loader, desc="train..."):
            # img=data['img'].to(device, non_blocking=True).float()
            # target=data['label'].to(device, non_blocking=True).long()
            #改动
            image = list(img.to(device) for img in image)
            outputs = model(image)
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

           ########################这是将多余检测框去掉或是去掉多余的标注，方便后面计算AUC#######################################3
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
            ##########################################################################

            face_imgs = []
            # 将图片中的人脸剪裁出来，准备放入SBI模型中
            for i in range(len(outputs_after)):

                face_input = []
                coordinates = outputs_after[i]
                coordinates = coordinates.round().long()
                targets_batch = gt_labels_after[i].to("cpu").numpy()
                if len(targets_batch)<1:
                    continue
                targets_batch = [*map(lambda x: x - 1, targets_batch)]
                targets_batch = torch.tensor(targets_batch).to(device)


                for j in range(len(coordinates)):

                    face_img = image[i][:, coordinates[j][1]:coordinates[j][3], coordinates[j][0]:coordinates[j][2]]
                    face_img = face_img.unsqueeze(dim=0)
                    face_img = torch.nn.functional.interpolate(face_img, size=380, mode='bilinear', align_corners=False)
                    face_imgs.append(face_img)
                    if j == 0:
                        _,stander = model1(face_img)    #tensor(1,2048)
                if (len(face_imgs) > 1 and len(face_imgs)<10):

                    face_imgs = torch.cat(face_imgs,dim=0)
                    optimizer.zero_grad()
                    out_cls = model_cls(face_imgs, stander)
                    loss_1 = criterion(out_cls, targets_batch)
                    loss_1.backward()
                    optimizer.step()

                    train_loss += loss_1

                    out_cls = out_cls.softmax(1)[:, 1].detach().to('cpu').numpy()
                    output_list.extend(out_cls)
                    face_imgs = []

                    for m in range(len(targets_batch)):
                        target_list.append(gt_labels_after[i][m])
                else:
                    face_imgs = []
                    continue



        target_list = [*map(lambda x: x - 1, target_list)]
        auc = roc_auc_score(target_list, output_list)
        print(f'openfor | train-AUC: {auc:.4f}')
        # print(f'FFIW | train-diff-AUC: {auc:.4f}')

        #########################################################Val
        target_list = []
        output_list = []
        model_cls.eval()
        model1.eval()

        with torch.no_grad():
            for image, targets in tqdm(val_dataset_loader, desc="validation..."):
                # 将图片传入指定设备device

                image = list(img.to(device) for img in image)
                outputs = model(image)
                outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

                ########################这是将多余检测框去掉或是去掉多余的标注，方便后面计算AUC#######################################3
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
                ##########################################################################

                face_imgs = []
                # 将图片中的人脸剪裁出来，准备放入SBI模型中
                for i in range(len(outputs_after)):

                    face_input = []
                    coordinates = outputs_after[i]
                    coordinates = coordinates.round().long()
                    targets_batch = gt_labels_after[i].to("cpu").numpy()
                    if len(targets_batch) < 1:
                        continue
                    targets_batch = [*map(lambda x: x - 1, targets_batch)]
                    targets_batch = torch.tensor(targets_batch).to(device)
                    input_cls_list = []


                    for j in range(len(coordinates)):
                        face_img = image[i][:, coordinates[j][1]:coordinates[j][3], coordinates[j][0]:coordinates[j][2]]
                        face_img = face_img.unsqueeze(dim=0)
                        face_img = torch.nn.functional.interpolate(face_img, size=380, mode='bilinear',
                                                                   align_corners=False)
                        face_imgs.append(face_img)
                        if j == 0:
                            _, stander = model1(face_img)  # tensor(1,2048)

                    if (len(face_imgs) > 1 and len(face_imgs)<10):

                        face_imgs = torch.cat(face_imgs, dim=0)
                        out_cls = model_cls(face_imgs, stander)
                        out_cls = out_cls.softmax(1)[:, 1].detach().to('cpu').numpy()
                        output_list.extend(out_cls)
                        face_imgs = []

                        for m in range(len(targets_batch)):
                            target_list.append(gt_labels_after[i][m])

                    else:
                        face_imgs = []
                        continue

            target_list = [*map(lambda x: x - 1, target_list)]
            auc = roc_auc_score(target_list, output_list)
            print(f'openfor | Val-diff-AUC: {auc:.4f}')


            target_list = []
            output_list = []
            #####################################################################################
            for image, targets in tqdm(test_dataset_loader, desc="validation..."):
                # 将图片传入指定设备device

                image = list(img.to(device) for img in image)
                outputs = model(image)
                outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

                ########################这是将多余检测框去掉或是去掉多余的标注，方便后面计算AUC#######################################3
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
                ##########################################################################

                face_imgs = []
                # 将图片中的人脸剪裁出来，准备放入SBI模型中
                for i in range(len(outputs_after)):
                    # coordinates = outputs[i]['boxes']
                    face_input = []
                    coordinates = outputs_after[i]
                    coordinates = coordinates.round().long()
                    targets_batch = gt_labels_after[i].to("cpu").numpy()
                    if len(targets_batch) < 1:
                        continue
                    targets_batch = [*map(lambda x: x - 1, targets_batch)]
                    targets_batch = torch.tensor(targets_batch).to(device)
                    input_cls_list = []


                    for j in range(len(coordinates)):
                        face_img = image[i][:, coordinates[j][1]:coordinates[j][3], coordinates[j][0]:coordinates[j][2]]
                        face_img = face_img.unsqueeze(dim=0)
                        face_img = torch.nn.functional.interpolate(face_img, size=380, mode='bilinear',
                                                                   align_corners=False)
                        face_imgs.append(face_img)
                        if j == 0:
                            _, stander = model1(face_img)  # tensor(1,2048)

                    if (len(face_imgs) > 1 and len(face_imgs)<10):

                        face_imgs = torch.cat(face_imgs, dim=0)
                        out_cls = model_cls(face_imgs, stander)
                        out_cls = out_cls.softmax(1)[:, 1].detach().to('cpu').numpy()
                        output_list.extend(out_cls)
                        face_imgs = []

                        for m in range(len(targets_batch)):
                            target_list.append(gt_labels_after[i][m])

                    else:
                        face_imgs = []
                        continue

                # 将每个人脸的labels放进列表
                # for i in range(len(gt_labels_after)):
                #     # temp_list=targets[i]['labels']
                #     temp_list = gt_labels_after[i]
                #     for j in range(len(temp_list)):
                #         target_list.append(temp_list[j])

            target_list = [*map(lambda x: x - 1, target_list)]
            auc = roc_auc_score(target_list, output_list)
            print(f'openfor | Test-diff-AUC: {auc:.4f}')

            target_list = []
            output_list = []
            #####################################################################################
            for image, targets in tqdm(test_challenge_dataset_loader, desc="validation..."):
                # 将图片传入指定设备device

                image = list(img.to(device) for img in image)
                outputs = model(image)
                outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

                ########################这是将多余检测框去掉或是去掉多余的标注，方便后面计算AUC#######################################3
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
                ##########################################################################

                face_imgs = []
                # 将图片中的人脸剪裁出来，准备放入SBI模型中
                for i in range(len(outputs_after)):
                    # coordinates = outputs[i]['boxes']
                    face_input = []
                    coordinates = outputs_after[i]
                    coordinates = coordinates.round().long()
                    targets_batch = gt_labels_after[i].to("cpu").numpy()
                    if len(targets_batch) < 1:
                        continue
                    targets_batch = [*map(lambda x: x - 1, targets_batch)]
                    targets_batch = torch.tensor(targets_batch).to(device)
                    input_cls_list = []

                    for j in range(len(coordinates)):
                        face_img = image[i][:, coordinates[j][1]:coordinates[j][3], coordinates[j][0]:coordinates[j][2]]
                        face_img = face_img.unsqueeze(dim=0)
                        face_img = torch.nn.functional.interpolate(face_img, size=380, mode='bilinear',
                                                                   align_corners=False)
                        face_imgs.append(face_img)
                        if j == 0:
                            _, stander = model1(face_img)  # tensor(1,2048)

                    if (len(face_imgs) > 1 and len(face_imgs) < 10):

                        face_imgs = torch.cat(face_imgs, dim=0)
                        out_cls = model_cls(face_imgs, stander)
                        out_cls = out_cls.softmax(1)[:, 1].detach().to('cpu').numpy()
                        output_list.extend(out_cls)
                        face_imgs = []

                        for m in range(len(targets_batch)):
                            target_list.append(gt_labels_after[i][m])

                    else:
                        face_imgs = []
                        continue

            target_list = [*map(lambda x: x - 1, target_list)]
            auc = roc_auc_score(target_list, output_list)
            print(f'openfor | Test-Chanllenge-diff-AUC: {auc:.4f}')

        torch.save(model_cls.state_dict(),'./outputs/Openfor_CLS_diff_{}epoch-(test-auc:{}).pth'.format(epoch,auc))
        torch.save(model1.state_dict(),'./outputs/Openfor_xcep_diff_{}epoch-(test-auc:{}).pth'.format(epoch,auc))



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

    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--resume', type=bool, default=False)


    args = parser.parse_args()

    main(args)
