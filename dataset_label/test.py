import torch
import json
from tqdm import tqdm
from my_dataset_openfrenic import OpenForensics
from network_files import boxes as box_ops

# my_dict = {}
# annotations = []
# a = [[412.8107, 85.46322, 620.52966, 391.48483], [660.9271, 269.1384, 839.69543, 548.11847], [258.52515, 48.649708, 371.46866, 221.78111]]
# for detection in a:
#     # 遍历每个参数并将其转换为整数
#     for i in range(len(detection)):
#         detection[i] = int(detection[i])
# b = [[400, 77, 621, 396],[660,269,839,548]]
# a=torch.Tensor(a)
# b = torch.Tensor(b)
# match_quality_matrix = box_ops.box_iou(a, b)
# IOU, indices = match_quality_matrix.max(dim=1)
# NEW_IOU =  [2 if x>0.8 else 1 for x in IOU]
# for box,label in zip(a,NEW_IOU):
#     dict = {}
#     dict["bbox"] = box
#     dict["category_id"] = label
#     annotations.append(dict)
# my_dict["annotations"] = annotations
# print(indices)



# my_dict = {}
# my_por = []
# my_fre = []
# for i in range(6):
#     dict = {}
#     dict["cate"] = str([[2.0,3.0],[4.0,5.0]])
#     dict["id"] = str([[1.0,2.0]])
#     dict["jame"] = "3"+f'{i}'
#     my_por.append(dict)
# for i in  range(6):
#     dict = {}
#     dict["cate"] = "1" + f'{i}'
#     dict["id"] = "2" + f'{i}'
#     dict["jame"] = "3" + f'{i}'
#     my_fre.append(dict)
#
# # json_str1 = json.dumps(my_fre, indent=4)
# # json_str2 = json.dumps(my_por, indent=4)
# my_dict["my_fre"] = my_fre
# my_dict["my_por"] = my_por
# json_str1 = json.dumps(my_dict, indent=4)
#
# with open("test.json","a") as f:
#     f.write(json_str1)


def train():
    data_root = "D:/yifangbin/dataset/FFIW10K/FFIW10K_test/target"
    train_dataset = OpenForensics(data_root, "Train")
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=2,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=2,
                                                        collate_fn=train_dataset.collate_fn)
    for image, targets in tqdm(train_data_loader, desc="train..."):
        print("img")
        print("-----")
if __name__ == '__main__':
    train()



