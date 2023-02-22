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

import network_files
import transforms
from backbone import resnet50_fpn_backbone
from my_dataset_openfrenic import OpenForensics
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
    def __init__(self, xception,input_dim=3, output_dim=2):
        super(ClassifierD, self).__init__()
        #########################
        self.xcep = xception
        #############################
        self.linear1 = nn.Linear(4096, 128)
        #self.batchnorm = nn.BatchNorm1d(128)
        #self.batchnorm = nn.BatchNorm1d(128)
        self.linear2 = nn.Linear(128, output_dim)

    def forward(self, x,y):
        # x = x.view(-1,x.size(0))
        x,mx = self.xcep(x)
        diff = mx-y
        x = torch.cat((diff, mx), dim=1)

        x = self.linear1(x)
        #x = self.batchnorm(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

class ClsD(nn.Module):
    def __init__(self,xception):
        # Instantiate the model
        super(ClsD, self).__init__()
        self.input_dim = 3
        self.output_dim = 2
        self.net = ClassifierD(xception,self.input_dim, self.output_dim)

        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        for name,param in self.net.xcep.named_parameters():
            if name not in ["bn4.weight","bn4.bias","conv4.weight","conv4.bias"]:
                param.requires_grad_(False)
            param.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001)
    def forward(self,x,y):
        outputs = []
        for i in range(len(x)):
            output = self.net(x[i].unsqueeze(dim=0),y)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0)
        return outputs
    def train_step(self,x,target,y):
        # outputs = []
        outputs = self.forward(x,y)
        # for i in range(len(x)):
        #
        #
        #     output = self.forward(x[i].unsqueeze(dim=0),y)
        #     #output = output.squeeze(dim=0)
        #         #target_loss = target[i].reshape(-1,1)
        #     outputs.append(output)
        #     target_loss = target[i].reshape(-1,)
        #     loss += self.criterion(output, target_loss)
        # outputs = torch.cat(outputs,dim=0)
        loss = self.criterion(outputs, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        ###############

        return outputs,loss


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
                                                     collate_fn=val_dataset.collate_fn)


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

    cnn_sd = torch.load('pre_trained75.tar', map_location="cpu")["model"]
    model1.load_state_dict(cnn_sd)
    model1.net.num_classes = 2
    model1 = model1.to(device)
    # cnn_sd = torch.load('./xception.pth', map_location="cpu")
    # model1.load_state_dict(cnn_sd)
    model1.train()

    # model2 = xception()
    # #model2.load_state_dict(torch.load('xception2.pth'))
    #
    # model2.net.fc = nn.Linear(model2.net.fc.in_features, 2)
    # nn.init.xavier_uniform_(model2.net.fc.weight)
    # cnn_sd = torch.load('pre_trained75.tar', map_location="cpu")["model"]
    # model1.load_state_dict(cnn_sd)
    # model2.net.num_classes = 2
    #
    # model2.to(device)
    # model2.eval()
    n_epoch = args.epoch
#############################################
    model_cls = ClsD(model1)

    model_cls.train()
    ############resume

    model_cls.to(device)


########################################
    for epoch in range(n_epoch):
        output_list = []
        target_list = []
        train_loss = 0.
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
                # coordinates = outputs[i]['boxes']
                face_input = []
                coordinates = outputs_after[i]
                coordinates = coordinates.round().long()
                targets_batch = gt_labels_after[i].to("cpu").numpy()
                targets_batch = [*map(lambda x: x - 1, targets_batch)]
                targets_batch = torch.tensor(targets_batch).to(device)
                input_cls_list = []

                # if len(coordinates) > 1:
                #stander = torch.randn(2048,)
                for j in range(len(coordinates)):
                    # coordinates = coordinates.round().long()
                    # coordinates_size = (coordinates[j][3]-coordinates[j][1],coordinates[j][2]-coordinates[j][0])
                    face_img = image[i][:, coordinates[j][1]:coordinates[j][3], coordinates[j][0]:coordinates[j][2]]
                    face_img = face_img.unsqueeze(dim=0)
                    face_img = torch.nn.functional.interpolate(face_img, size=380, mode='bilinear', align_corners=False)
                    face_imgs.append(face_img)
                    if j == 0:
                        _,stander = model1(face_img)    #tensor(1,2048)
                if (len(face_imgs) > 0):

                    face_imgs = torch.cat(face_imgs,dim=0)
                    out_cls, loss = model_cls.train_step(face_imgs, targets_batch.long(), stander)
                    train_loss += loss
                    out_cls = out_cls.softmax(1)[:, 1].detach().to('cpu').numpy()
                    output_list.extend(out_cls)
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
        print(f'openfor | train-AUC: {auc:.4f}')

        #sbi
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
                    # coordinates = outputs[i]['boxes']
                    face_input = []
                    coordinates = outputs_after[i]
                    coordinates = coordinates.round().long()
                    targets_batch = gt_labels_after[i].to("cpu").numpy()
                    targets_batch = [*map(lambda x: x - 1, targets_batch)]
                    targets_batch = torch.tensor(targets_batch).to(device)
                    input_cls_list = []

                    #if len(coordinates) > 1:
                        # stander = torch.randn(2048,)
                    for j in range(len(coordinates)):
                        # coordinates = coordinates.round().long()
                        # coordinates_size = (coordinates[j][3]-coordinates[j][1],coordinates[j][2]-coordinates[j][0])
                        face_img = image[i][:, coordinates[j][1]:coordinates[j][3], coordinates[j][0]:coordinates[j][2]]
                        face_img = face_img.unsqueeze(dim=0)
                        face_img = torch.nn.functional.interpolate(face_img, size=380, mode='bilinear',
                                                                   align_corners=False)
                        face_imgs.append(face_img)
                        if j == 0:
                            _, stander = model1(face_img)  # tensor(1,2048)

                    if (len(face_imgs) > 0):

                        face_imgs = torch.cat(face_imgs, dim=0)
                        out_cls = model_cls(face_imgs, stander)
                        out_cls = out_cls.softmax(1)[:, 1].detach().to('cpu').numpy()
                        output_list.extend(out_cls)
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
            print(f'openfor | Val-AUC: {auc:.4f}')


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
                    targets_batch = [*map(lambda x: x - 1, targets_batch)]
                    targets_batch = torch.tensor(targets_batch).to(device)
                    input_cls_list = []

                    #if len(coordinates) > 1:
                        # stander = torch.randn(2048,)
                    for j in range(len(coordinates)):
                        # coordinates = coordinates.round().long()
                        # coordinates_size = (coordinates[j][3]-coordinates[j][1],coordinates[j][2]-coordinates[j][0])
                        face_img = image[i][:, coordinates[j][1]:coordinates[j][3], coordinates[j][0]:coordinates[j][2]]
                        face_img = face_img.unsqueeze(dim=0)
                        face_img = torch.nn.functional.interpolate(face_img, size=380, mode='bilinear',
                                                                   align_corners=False)
                        face_imgs.append(face_img)
                        if j == 0:
                            _, stander = model1(face_img)  # tensor(1,2048)

                    if (len(face_imgs) > 0):

                        face_imgs = torch.cat(face_imgs, dim=0)
                        out_cls = model_cls(face_imgs, stander)
                        out_cls = out_cls.softmax(1)[:, 1].detach().to('cpu').numpy()
                        output_list.extend(out_cls)
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
            print(f'openfor | Test-AUC: {auc:.4f}')

        torch.save(model_cls.state_dict(),'./outputs/{}new_epoch.pth'.format(epoch))
        torch.save(model1.state_dict(),'./outputs/{}new_Xception_epoch.pth'.format(epoch))



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

    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--resume', type=bool, default=False)


    args = parser.parse_args()

    main(args)
