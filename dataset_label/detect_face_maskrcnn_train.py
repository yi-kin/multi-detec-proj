import argparse
import os
import json

import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import PIL
import cv2
import network_files
from torchvision import transforms
from backbone import resnet50_fpn_backbone
from draw_box_utils import draw_objs
from network_files import MaskRCNN
from network_files import boxes as box_ops
from mask_detect import get_mask_detect
from Pytorch_Retinaface_master.layers.functions.prior_box import PriorBox
from Pytorch_Retinaface_master.models.retinaface import RetinaFace
from Pytorch_Retinaface_master.utils.box_utils import decode, decode_landm
from Pytorch_Retinaface_master.utils.timer import Timer
from Pytorch_Retinaface_master.data import cfg_mnet, cfg_re50
from Pytorch_Retinaface_master.utils.nms.py_cpu_nms import py_cpu_nms
import torch.backends.cudnn as cudnn
parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--trained_model', default='../Pytorch_Retinaface_master/weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
parser.add_argument('--dataset_folder', default='./data/widerface/val/images/', type=str, help='dataset path')
parser.add_argument('--confidence_threshold', default=0.85, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.7, type=float, help='visualization_threshold')
args = parser.parse_args()

def get_face_detect(count,each_pic_path,save_name=None,img=None):
    if each_pic_path==None:
        img_raw = img
    else:
        img_raw = cv2.imread(each_pic_path, cv2.IMREAD_COLOR)
    img = np.float32(img_raw)

    # testing scale
    target_size = 1600
    max_size = 2150
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    resize = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(resize * im_size_max) > max_size:
        resize = float(max_size) / float(im_size_max)
    if args.origin_size:
        resize = 1

    if resize != 1:
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    _t['forward_pass'].tic()
    loc, conf, landms = net(img)  # forward pass
    _t['forward_pass'].toc()
    _t['misc'].tic()
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1]
    # order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    # dets = dets[:args.keep_top_k, :]
    # landms = landms[:args.keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)
    _t['misc'].toc()

    # --------------------------------------------------------------------
    print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(count + 1, 8000,
                                                                                 _t['forward_pass'].average_time,
                                                                                 _t['misc'].average_time))

    # save image
    if save_name is not None:
        for b in dets:
            if b[4] < args.vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))


        # save image
        # if not os.path.exists("./results/"):
        #     os.makedirs("./results/")
        # name = "./results/" + str(count) + ".jpg"

        cv2.imwrite(save_name, img_raw)
    boxes_ret = []
    for det in dets:
        box_ret = [det[0],det[1],det[2],det[3]]
        boxes_ret.append(box_ret)

    return boxes_ret


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

torch.set_grad_enabled(False)

cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50
# net and model
net = RetinaFace(cfg=cfg, phase = 'test')
net = load_model(net, args.trained_model, args.cpu)
net.eval()
print('Finished loading model!')
print(net)
cudnn.benchmark = True
device = torch.device("cpu" if args.cpu else "cuda")
net = net.to(device)

_t = {'forward_pass': Timer(), 'misc': Timer()}



def begin():


    pic_count = 0
    face_label_count = 0

    mask_img_box_detect_save_path = "E:/yifangbin/dataset/FFIW10K/FFIW10K_test/target/train_mask_box_detect_save"
    face_img_box_detect_save_path = "E:/yifangbin/dataset/FFIW10K/FFIW10K_test/target/train_face_box_detect_save"
    pic_path = "E:/yifangbin/dataset/FFIW10K/FFIW10K_test/target/train_pic_save"
    mask_path = "E:/yifangbin/dataset/FFIW10K/FFIW10K_test/target/tarin_mask_pic_save"

    my_dict = {}
    annotations = []
    images = []
    categories = []

    dict2 = {}
    dict2["id"] = 1
    dict2["name"] = "Real"
    categories.append(dict2)
    dict2= {}
    dict2["id"] = 2
    dict2["name"] = "Fake"
    categories.append(dict2)
    with torch.no_grad():
        for pic_name,mask_name in zip(os.listdir(pic_path),os.listdir(mask_path)):

            each_pic_path = os.path.join(pic_path, pic_name)
            each_pic_mask = os.path.join(mask_path,mask_name)

            save_path_face = face_img_box_detect_save_path + f'/{pic_count}.jpg'
            # boxes_face = get_face_detect(each_pic_path,pic_count,save_path_face)
            boxes_face = get_face_detect(pic_count,each_pic_path,save_path_face,img=None)

            save_path_mask = mask_img_box_detect_save_path + f'/{pic_count}.jpg'
            boxes_mask = get_mask_detect(each_pic_mask,save_path_mask)

            #当没有检测到东西时，跳过这张图片
            if len(boxes_face)<1 or boxes_mask==None:
                # pic_count+=1
                continue

            #检测到人脸，开始标注
            dict1 = {}
            dict1["id"] = pic_count
            dict1["file_name"] = pic_name
            images.append(dict1)

            ##########################计算IOU
            boxes_face = torch.Tensor(boxes_face)
            boxes_mask = torch.Tensor(boxes_mask)
            match_quality_matrix = box_ops.box_iou(boxes_face, boxes_mask)
            IOU, indices = match_quality_matrix.max(dim=1)
            NEW_IOU = [2 if x > 0.76 else 1 for x in IOU]

            boxes_face = boxes_face.numpy().tolist()
            boxes_mask = boxes_mask.numpy().tolist()
            for i in range(len(boxes_face)):
                boxes_face[i] = [int(x) for x in boxes_face[i]]
            for box, label in zip(boxes_face, NEW_IOU):
                dict = {}
                dict["id"] = face_label_count
                dict["image_id"] = pic_count
                dict["iscrowd"] = 0
                dict["category_id"] = label
                dict["bbox"] = box

                face_label_count+=1
                annotations.append(dict)


            print("face boxses:",pic_count," ",boxes_face)
            print("mask boxes:",pic_count," ",boxes_mask)
            print("---------------------------")
            pic_count+=1

        my_dict["categories"] = categories
        my_dict["images"] = images
        my_dict["annotations"] = annotations
        json_str1 = json.dumps(my_dict, indent=4)


        with open('Train_poly2.json', "a") as f:
            f.write(json_str1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retinaface')
    parser.add_argument('-m', '--trained_model', default='../Pytorch_Retinaface_master/weights/Resnet50_Final.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
    parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str,
                        help='Dir to save txt results')
    parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
    parser.add_argument('--dataset_folder', default='./data/widerface/val/images/', type=str, help='dataset path')
    parser.add_argument('--confidence_threshold', default=0.85, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
    parser.add_argument('--vis_thres', default=0.7, type=float, help='visualization_threshold')
    args = parser.parse_args()

    begin()