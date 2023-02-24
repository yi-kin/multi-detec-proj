import cv2
import os
import numpy as np

# 定义每个视频需要抽取的帧数
num_frames = 5
count0 = 0
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
def get_random_frame(video_path):
    # 读取视频帧数
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # 随机抽取一帧
    random_frame_index = np.random.randint(frame_count)
    random_frame_path = os.path.join(video_path, f"frame_{random_frame_index}.jpg")

    # 读取随机帧并保存
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_index)
    ret, frame = cap.read()
    cv2.imwrite(random_frame_path, frame)
    cap.release()

    return random_frame_path

# 遍历第一个视频数据集
# video_dir_1 = "D:/yifangbin/dataset/FFIW10K/FFIW10K_test/target/train_video"
# save_path = "D:/yifangbin/dataset/FFIW10K/FFIW10K_test/target/train_pic_save"

video_dir_1 = "D:/yifangbin/dataset/FFIW10K/FFIW10K_test/target/train_mask_vedio"
save_path = "D:/yifangbin/dataset/FFIW10K/FFIW10K_test/target/tarin_mask_pic_save"

for video_name in os.listdir(video_dir_1):
    video_path = os.path.join(video_dir_1, video_name)

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

# # 遍历第二个视频数据集
# video_dir_2 = "video_dataset_2"
# for video_name in os.listdir(video_dir_2):
#     video_path = os.path.join(video_dir_2, video_name)
#
#     # 随机抽取一帧并保存
#     random_frame_path = get_random_frame(video_path)
#     print(f"Randomly selected frame from {video_path}: {random_frame_path}")