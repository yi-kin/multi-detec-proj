import cv2
import os
import numpy as np
import random
from dataset_label.detect_face_maskrcnn import get_face_detect
from dataset_label.mask_detect import get_mask_detect


# 定义第一个视频每帧之间的时间间隔
def get_interval(video_path, num_frames):
    # 读取视频时长
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()

    # 计算每帧之间的时间间隔
    interval = duration / num_frames

    return interval

# 定义第二个视频随机抽取一帧
def get_random_frame(face_video_path,mask_video_path):
    # 读取视频帧数
    cap = cv2.VideoCapture(face_video_path)
    cap1 = cv2.VideoCapture(mask_video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))


    # 随机抽取一帧
    # save_random_path = "D:/yifangbin/dataset/FFIW10K/FFIW10K-v1-release-test/FFIW10K-v1-release-test/target/44"
    random_frame_index = random.randint(0,frame_count-1)
    # random_frame_path = os.path.join(save_random_path, f"frame_{random_frame_index}.jpg")

    # 读取随机帧并保存

    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_index)
    cap1.set(cv2.CAP_PROP_POS_FRAMES, random_frame_index)
    ret, face_frame = cap.read()
    ret, mask_frame = cap1.read()

    cv2.imwrite("face.jpg", face_frame)
    cv2.imwrite("mask.jpg", mask_frame)

    cap.release()
    cap1.release()


    return face_frame,mask_frame


def begin_exract(video_path,num_frames,count0,save_path,video_name):
    # 计算每帧之间的时间间隔
    interval = get_interval(video_path, num_frames)

    # 抽取每个视频中的15帧
    cap = cv2.VideoCapture(video_path)
    count = 0
    index = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        temp = int(index * interval * cap.get(cv2.CAP_PROP_FPS))
        if count == temp:
            index += 1
            # 保存帧
            frame_path = os.path.join(save_path, f"{video_name}_{count0}.jpg")
            count0+=1
            judge = cv2.imwrite(frame_path, frame)
            if index >= num_frames+1:
                break
    cap.release()

# 遍历第一个视频数据集
#train_img
# video_dir_1 = "D:/yifangbin/dataset/FFIW10K/FFIW10K-v1_2/FFIW10K-v1-release/target/train"
# save_path = "D:/yifangbin/dataset/FFIW10K/FFIW10K_test/target/train_pic_save"
#train_mask_img
# video_dir_1 = "D:/yifangbin/dataset/FFIW10K/FFIW10K-v1_2/FFIW10K-v1-release/target_mask/train"
# save_path = "D:/yifangbin/dataset/FFIW10K/FFIW10K_test/target/tarin_mask_pic_save"
#val_img
# video_dir_1 = "D:/yifangbin/dataset/FFIW10K/FFIW10K-v1_2/FFIW10K-v1-release/target/val"
# save_path = "D:/yifangbin/dataset/FFIW10K/FFIW10K_test/target/val_pic_save"
#val_mask_img
video_dir_1 = "D:/yifangbin/dataset/FFIW10K/FFIW10K-v1-release-test/FFIW10K-v1-release-test/target/test"
video_dir_2 = "D:/yifangbin/dataset/FFIW10K/FFIW10K-v1-release-test/FFIW10K-v1-release-test/target_mask/test"

face_extract_save_path = "D:/yifangbin/dataset/FFIW10K/FFIW10K_test/target/test_face_extract"
face_mask_extract_save_path = "D:/yifangbin/dataset/FFIW10K/FFIW10K_test/target/test_face_mask_extract"

test_vider_random_frame = "D:/yifangbin/dataset/FFIW10K/FFIW10K_test/target/test_vedio_random_frame"
test_mask_vedio_random_frame = "D:/yifangbin/dataset/FFIW10K/FFIW10K_test/target/test_mask_vedio_random_frame"
#统计
face_statistics = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
face_mask_statistics = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# 定义每个视频需要抽取的帧数
num_frames = 5
video_count = 0
count0 = 0
#防止检测不到人脸设置的循环推出
random_count = 0

for face_video_name,mask_video_name in zip(os.listdir(video_dir_1),os.listdir(video_dir_2)):
    face_video_path = os.path.join(video_dir_1, face_video_name)
    mask_video_path = os.path.join(video_dir_2, mask_video_name)

    #统计每个视频里面的人脸数(每个视频随机抽取一帧后),决定每个视频抽几帧
    face_num = 0
    mask_num = 0
    random_count = 0
    while face_num<1:
        face_frame_random,mask_frame_random = get_random_frame(face_video_path,mask_video_path)
        boxes_face = get_face_detect(video_count, save_name=None, img=face_frame_random,each_pic_path=None)
        boxes_mask = get_mask_detect(mask_frame_random, pic_name=None,save_path=None)
        if random_count>10:
            face_num = 0
            mask_num = 0
            break
        if boxes_mask == None or len(boxes_face)<1:
            random_count+=1
            continue
        face_num = len(boxes_face)
        mask_num = len(boxes_mask)
    face_statistics[face_num]+=1
    face_mask_statistics[mask_num]+=1
    print(f'--------------{video_count}:face-num{face_num}    mask-num{mask_num}')

    if face_num>1 and mask_num>1:
        begin_exract(face_video_path,30,count0,face_extract_save_path,face_video_name)
        begin_exract(mask_video_path,30,count0,face_mask_extract_save_path,mask_video_name)
    elif face_num>1 and mask_num <2 :
        begin_exract(face_video_path, 15, count0, face_extract_save_path, face_video_name)
        begin_exract(mask_video_path, 15, count0, face_mask_extract_save_path, mask_video_name)
    elif face_num  <= 1:
        begin_exract(face_video_path, 5, count0, face_extract_save_path, face_video_name)
        begin_exract(mask_video_path, 5, count0, face_mask_extract_save_path, mask_video_name)

    video_count+=1


print(face_statistics)





