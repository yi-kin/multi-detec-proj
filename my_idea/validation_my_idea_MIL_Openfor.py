"""
该脚本用于第二个想法MIL的训练及验证，此脚本用于FFIW数据集
"""

import os
import json

import torch
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
from xception_MIL import xception
#from PIL import Image
from torchvision.transforms import Resize
from sklearn.metrics import confusion_matrix, roc_auc_score

from network_files import boxes as box_ops

import torch.nn as nn
import torch.nn.functional as F
from weight_loss import CrossEntropyLoss as CE
import torch.optim as optim




class AttentionLayer(nn.Module):
    def __init__(self, dim=512):
        super(AttentionLayer, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, 1)

    def forward(self, features, W_1, b_1,W_2 ,b_2,flag): #feature:(4,2048)
        if flag == 1:
            #下面这个是共享全连接层
            out_c = F.linear(features, W_1, b_1)
            out_c = F.relu(out_c)
            out_c = F.linear(out_c,W_2,b_2)

            out = out_c - out_c.max()
            out = out.exp()
            out = out.sum(1, keepdim=True)
            alpha = out / out.sum(0)

            alpha01 = features.size(0) * alpha.expand_as(features) #alpha0:(4,2048)
            context = torch.mul(features, alpha01)
        else:
            context = features
            alpha = torch.zeros(features.size(0), 1)

        return context, out_c, torch.squeeze(alpha)

class MIL_xcep(nn.Module):
    def __init__(self,xcep):
        super(MIL_xcep, self).__init__()
        self.xception = xcep
        self.att_layer = AttentionLayer(2048)
        self.linear1 = nn.Linear(2048, 128)
        self.linear2 = nn.Linear(128, 2)
        params = {}
        for name,param in self.xception.named_parameters():
            if name not in  ["fc.weight", "fc.bias", "bn4.weight", "bn4.bias", 'conv4.conv1.weight', 'conv4.pointwise.weight']:
                param.requires_grad_(False)




    def forward(self, x, flag=1):
        _,mx = self.xception(x)

        out, out_c, alpha = self.att_layer(mx, self.linear1.weight, self.linear1.bias,self.linear2.weight,self.linear2.bias, flag)
        # m = out
        out = out.mean(0, keepdim=True)
        

        y = self.linear1(out)
        y = F.relu(y)
        y = self.linear2(y)

        return y,out_c,alpha



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
    train_dataset = OpenForensics(data_root, "Train", data_transform["train"])
    val_dataset = OpenForensics(data_root,dataset='Val',transform=data_transform["val"])
    test_dataset = OpenForensics(data_root,dataset='Test-Dev',transform=data_transform["val"])
    test_challenge_dataset = OpenForensics(data_root,dataset='Test-Challenge',transform=data_transform["val"])


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
    test_challenge_dataset_loader = torch.utils.data.DataLoader(test_challenge_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      num_workers=nw,
                                                      collate_fn=test_challenge_dataset.collate_fn)

    # create model（目标检测的模型）################1111#####################
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
    model.eval()
    ###############################################1111##########################



#Xception--model############################
    model1 = xception()  #这已经改成了二分类了
    #model1.load_state_dict(torch.load('xception2.pth'))

    model1.net.fc = nn.Linear(model1.net.fc.in_features, 2)
    nn.init.xavier_uniform_(model1.net.fc.weight)

    # cnn_sd = torch.load('pre_trained75.tar', map_location="cpu")["model"]
    cnn_sd = torch.load('Openfor：1epoch-0.9668582683719695.pth', map_location="cpu")
    model1.load_state_dict(cnn_sd)

    model1.net.num_classes = 2
    model1 = model1.to(device)

    model1.train()


    n_epoch = args.epoch
#############################################
    model_cls = MIL_xcep(model1)

    model_cls.train()
    ############resume

    model_cls.to(device)



    optimizer = optim.Adam(model_cls.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=args.reg)
    criterion = torch.nn.CrossEntropyLoss(size_average=True)
    weight_criterion = CE(aggregate='mean')

########################################
    for epoch in range(n_epoch):
        model1.train()
        model_cls.train()

        output_list = []
        target_list = []

        train_loss = 0.
        count = 0
        no_count = 0
        print(f'--------------epoch:{epoch}------------------')
        for image, targets in tqdm(train_data_loader, desc="train..."):

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
                bag_label = targets_batch.max()-1
                targets_batch = [*map(lambda x: x - 1, targets_batch)]

                bag_label = torch.tensor(bag_label).to(device).unsqueeze(dim=0)
                targets_batch = torch.tensor(targets_batch).to(device)
                input_cls_list = []


                for j in range(len(coordinates)):

                    face_img = image[i][:, coordinates[j][1]:coordinates[j][3], coordinates[j][0]:coordinates[j][2]]
                    face_img = face_img.unsqueeze(dim=0)
                    face_img = torch.nn.functional.interpolate(face_img, size=380, mode='bilinear', align_corners=False)
                    face_imgs.append(face_img)

                if (len(face_imgs) > 1 and len(face_imgs)<10):
                    count+=1
                    face_imgs = torch.cat(face_imgs,dim=0)

                    optimizer.zero_grad()
                    bag_pre,instance_pre,alpha = model_cls(face_imgs)

                    loss_1 = criterion(bag_pre, bag_label)

                    loss_2 = weight_criterion(instance_pre, targets_batch, weights=alpha)
                    loss = loss_1 + 2.0 * loss_2

                    # backward pass
                    loss.backward()
                    # step
                    optimizer.step()

                    out_cls = instance_pre.softmax(1)[:, 1].detach().to('cpu').numpy()
                    output_list.extend(out_cls)
                    face_imgs = []

                    for m in range(len(targets_batch)):
                        target_list.append(gt_labels_after[i][m])
                else:
                    face_imgs = []
                    no_count+=1
                    continue


        target_list = [*map(lambda x: x - 1, target_list)]
        auc = roc_auc_score(target_list, output_list)

        print("count=",count)
        print("no-count",no_count)
        print(f'openfor | train-MIL-AUC: {auc:.4f}')

        #sbi
        target_list = []
        output_list = []
        count = 0
        no_count = 0
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

                    bag_label = targets_batch.max() - 1
                    bag_label = torch.tensor(bag_label).to(device).unsqueeze(dim=0)
                    targets_batch = [*map(lambda x: x - 1, targets_batch)]

                    targets_batch = torch.tensor(targets_batch).to(device)
                    input_cls_list = []

                    for j in range(len(coordinates)):

                        face_img = image[i][:, coordinates[j][1]:coordinates[j][3], coordinates[j][0]:coordinates[j][2]]
                        face_img = face_img.unsqueeze(dim=0)
                        face_img = torch.nn.functional.interpolate(face_img, size=380, mode='bilinear',
                                                                   align_corners=False)
                        face_imgs.append(face_img)

                    if (len(face_imgs) > 1 and len(face_imgs)<10):

                        face_imgs = torch.cat(face_imgs, dim=0)
                        bag_pre,instance_pre,alpha = model_cls(face_imgs)
                        out_cls = instance_pre.softmax(1)[:, 1].detach().to('cpu').numpy()
                        output_list.extend(out_cls)
                        face_imgs = []

                        for m in range(len(targets_batch)):
                            target_list.append(gt_labels_after[i][m])
                        count+=1

                    else:
                        no_count+=1
                        face_imgs = []
                        continue


            target_list = [*map(lambda x: x - 1, target_list)]
            auc = roc_auc_score(target_list, output_list)
            print(f'openfor | Val-MIL-AUC: {auc:.4f}')
            print("count=", count)
            print("no-count", no_count)


            target_list = []
            output_list = []
            count=0
            no_count=0
            #####################################################################################
            for image, targets in tqdm(test_dataset_loader, desc="test..."):
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
                    bag_label = targets_batch.max() - 1
                    bag_label = torch.tensor(bag_label).to(device).unsqueeze(dim=0)
                    targets_batch = [*map(lambda x: x - 1, targets_batch)]
                    targets_batch = torch.tensor(targets_batch).to(device)
                    input_cls_list = []

                    for j in range(len(coordinates)):
                        face_img = image[i][:, coordinates[j][1]:coordinates[j][3], coordinates[j][0]:coordinates[j][2]]
                        face_img = face_img.unsqueeze(dim=0)
                        face_img = torch.nn.functional.interpolate(face_img, size=380, mode='bilinear',
                                                                   align_corners=False)
                        face_imgs.append(face_img)

                    if (len(face_imgs) > 1 and len(face_imgs)<10):

                        face_imgs = torch.cat(face_imgs, dim=0)
                        bag_pre,instance_pre,alpha = model_cls(face_imgs)
                        out_cls = instance_pre.softmax(1)[:, 1].detach().to('cpu').numpy()
                        output_list.extend(out_cls)
                        face_imgs = []

                        for m in range(len(targets_batch)):
                            target_list.append(gt_labels_after[i][m])
                        count+=1
                    else:
                        no_count+=1
                        face_imgs = []
                        continue


            target_list = [*map(lambda x: x - 1, target_list)]
            auc = roc_auc_score(target_list, output_list)
            print(f'openfor | Test-MIL-AUC: {auc:.4f}')
            print("count=", count)
            print("no-count", no_count)

            target_list = []
            output_list = []
            count = 0
            no_count = 0
            #####################################################################################
            for image, targets in tqdm(test_challenge_dataset_loader, desc="test..."):
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
                    bag_label = targets_batch.max() - 1
                    bag_label = torch.tensor(bag_label).to(device).unsqueeze(dim=0)
                    targets_batch = [*map(lambda x: x - 1, targets_batch)]
                    targets_batch = torch.tensor(targets_batch).to(device)
                    input_cls_list = []

                    for j in range(len(coordinates)):
                        face_img = image[i][:, coordinates[j][1]:coordinates[j][3], coordinates[j][0]:coordinates[j][2]]
                        face_img = face_img.unsqueeze(dim=0)
                        face_img = torch.nn.functional.interpolate(face_img, size=380, mode='bilinear',
                                                                   align_corners=False)
                        face_imgs.append(face_img)

                    if (len(face_imgs) > 1 and len(face_imgs) < 10):

                        face_imgs = torch.cat(face_imgs, dim=0)
                        bag_pre, instance_pre, alpha = model_cls(face_imgs)
                        out_cls = instance_pre.softmax(1)[:, 1].detach().to('cpu').numpy()
                        output_list.extend(out_cls)
                        face_imgs = []
                        for m in range(len(targets_batch)):
                            target_list.append(gt_labels_after[i][m])
                        count += 1
                    else:
                        no_count += 1
                        face_imgs = []
                        continue

            target_list = [*map(lambda x: x - 1, target_list)]
            auc = roc_auc_score(target_list, output_list)
            print(f'openfor | Test-challenge-MIL-AUC: {auc:.4f}')
            print("count=", count)
            print("no-count", no_count)

        torch.save(model_cls.state_dict(),'./outputs/Openfor_CLS_MIL_{}epoch-(test-auc:{}).pth'.format(epoch,auc))
        torch.save(model1.state_dict(),'./outputs/Openfor_xcep_MIL_{}epoch-(test-auc:{}).pth'.format(epoch,auc))



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
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                        help='weight decay')


    args = parser.parse_args()

    main(args)
