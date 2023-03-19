import cv2
# import dlib
import numpy as np
import torch
from PIL import Image
from xception import xception
import torch.nn as nn
import torchvision.transforms as transforms
# 加载Xception模型，不包括最后一层全连接层，以便输出特征向量
model = xception()
model.net.fc = nn.Linear(model.net.fc.in_features, 2)
nn.init.xavier_uniform_(model.net.fc.weight)
#这里是加载在FFIW或Openfor上面迁移训练后的模型参数

cnn_sd = torch.load('pre_trained75.tar', map_location="cpu")["model"]
model.load_state_dict(cnn_sd)
# 定义一个函数，用于检测和裁剪人脸，并转换为224x224的大小，以适应Xception模型的输入要求
# def preprocess_face(image):
#   # 使用dlib的人脸检测器，返回一个矩形列表，每个矩形代表一个人脸区域
#   detector = dlib.get_frontal_face_detector()
#   rects = detector(image, 1)
#   # 如果没有检测到人脸，返回空值
#   if len(rects) == 0:
#     return None
#   # 否则，取第一个矩形作为人脸区域，并裁剪出来
#   rect = rects[0]
#   x1 = rect.left()
#   y1 = rect.top()
#   x2 = rect.right()
#   y2 = rect.bottom()
#   face = image[y1:y2, x1:x2]
#   # 将人脸缩放为224x224，并转换为浮点数类型
#   face = cv2.resize(face, (224, 224))
#   face = face.astype(np.float32)
#   # 返回预处理后的人脸图像
#   return face

# 定义一个函数，用于提取人脸图像的特征向量，并将其展平为一维数组
def extract_feature(face):
  # 将人脸图像增加一个维度，以适应Xception模型的输入要求（batch_size, height, width, channels）
  face = np.expand_dims(face, axis=0)
  face = torch.tensor(face)
  # face = face.unsqueeze(dim=0)
  # 使用Xception模型对人脸图像进行特征提取，得到一个四维数组（batch_size, height, width, features）
  _,feature = model(face)
  # 将四维数组展平为一维数组，并返回特征向量
  feature = feature.flatten()
  return feature




# 定义一个函数，用于根据两个特征向量绘制一个纹理图片
def draw_texture(feature_1, feature_2):
    # 定义一些参数，可以根据需要修改
    texture_size = (512, 512)  # 纹理图片的大小（高度，宽度）
    texture_color = (255, 255, 255)  # 纹理图片的颜色（BGR格式）
    stripe_width = 10  # 条纹的宽度（像素）
    stripe_color_1 = (0, 0, 255)  # 第一张人脸对应的条纹颜色（BGR格式）
    stripe_color_2 = (0, 255, 0)  # 第二张人脸对应的条纹颜色（BGR格式）

    # 创建一个空白的纹理图片，并填充为指定的颜色
    texture = np.zeros(texture_size + (3,), dtype=np.uint8)
    texture[:] = texture_color

    # 计算两个特征向量之间的欧氏距离，并归一化到[0,1]区间
    feature_1 = feature_1.detach().numpy()
    feature_2 = feature_2.detach().numpy()

    distance = np.linalg.norm(feature_1 - feature_2)
    # distance = min(max(distance / np.sqrt(len(feature_1)), 0), 1)

    # 根据距离计算条纹之间的间隔（像素），越相似则间隔越小，越不相似则间隔越大
    stripe_gap = int(distance * texture_size[1])

    # 在纹理图片上绘制条纹，交替使用两种颜色，并保持一定的宽度和间隔
    for i in range(0, texture_size[0], stripe_width + stripe_gap):
        cv2.rectangle(texture, (i, 0), (i + stripe_width - 1, texture_size[1] - 1), stripe_color_1, -1)
        cv2.rectangle(texture, (i + stripe_width + stripe_gap - 1, 0), (i + 2 * stripe_width + stripe_gap - 2,
                                                                        texture_size[1] - 1), stripe_color_2, -1)

        # 返回绘制好的纹理图片
    return texture


# 定义两个变量，分别存储两张输入图片的路径（根据实际情况修改）
image_path_1 = 'face2.jpg'
image_path_2 = 'face3.jpg'

# 分别读取两张图片，并转换为灰度图像（如果是彩色图像）
# image_1 = cv2.imread(image_path_1)
# image_2 = cv2.imread(image_path_2)
# if image_1.shape[2] ==3:
#     image_1= cv2.cvtColor(image_1,cv2.COLOR_BGR2GRAY)
# if image_2.shape[2] ==3:
#     image_2= cv2.cvtColor(image_2,cv2.COLOR_BGR2GRAY)


# 分别对两张图片进行预处理，得到两个人脸图像（如果有多个人脸，则只取第一个）
img1 = Image.open(image_path_1).convert('RGB')
transform = transforms.PILToTensor() # 定义一个转化器
image_1 = transform(img1) # 将图片转化为Tensor

img2 = Image.open(image_path_2).convert('RGB')
transform = transforms.PILToTensor() # 定义一个转化器
image_2 = transform(img2) #

image_1 = image_1.unsqueeze(dim=0).float()
image_2 = image_2.unsqueeze(dim=0).float()

face_1 = torch.nn.functional.interpolate(image_1, size=224, mode='bilinear', align_corners=False)
face_2 = torch.nn.functional.interpolate(image_2, size=224, mode='bilinear', align_corners=False)
face_1 = face_1.squeeze(dim=0)
face_2 = face_2.squeeze(dim=0)
# face_1 = cv2.resize(image_1, (224, 224))
# face_1 = face_1.astype(np.float32)
#
# face_2 = cv2.resize(image_2, (224, 224))
# face_2 = face_2.astype(np.float32)

feature_1 = extract_feature(face_1)
feature_2 = extract_feature(face_2)

# 调用上面定义好的函数，传入两个特征向量作为参数，得到一个纹理图片
texture = draw_texture(feature_1, feature_2)

# 显示并保存纹理图片（根据实际情况修改保存路径）
# cv2.imshow('Texture', texture)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite('texture.jpg', texture)