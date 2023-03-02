import cv2
import os
import numpy as np
import math



def get_center(rect):
    x, y, w, h = rect
    return (x + w // 2, y + h // 2)


pic_path = "D:/yifangbin/dataset/FFIW10K/FFIW10K_test/target/tarin_mask_pic_save"

def get_mask_detect(img,pic_name,save_path):
    # pic_path = "D:/yifangbin/dataset/FFIW10K/FFIW10K_test/target/tarin_mask_pic_save"
    # save_path = "mask_result"
    if pic_name == None:
        img_raw = img
    else:
        each_pic_path = os.path.join(pic_path, pic_name)
        img_raw = cv2.imread(each_pic_path, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

    # 对图像进行二值化处理
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # 进行形态学操作，填充空洞
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 轮廓检测
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    # 绘制方形边界框
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box1 = x
        box2 = y
        box3 = w+x
        box4 = h+y
        area = int((box3 - box1)*(box4-box2))
        box = [box1,box2,box3,box4]
        if area>400:
            boxes.append(box)
    if len(boxes)>1:
        max_idx = 0
        max_area = 0
        for i, rect in enumerate(boxes):
            area = rect[2] * rect[3]
            if area > max_area:
                max_idx = i
                max_area = area

        # 计算面积最大检测框的中心点坐标
        max_center = get_center(boxes[max_idx])

        # 遍历所有检测框，计算每个检测框的中心点坐标，并计算到面积最大检测框中心点的距离
        new_rects = []
        new_rects.append(boxes[max_idx])
        for rect in boxes:
            center = get_center(rect)
            dist = math.sqrt((center[0] - max_center[0]) ** 2 + (center[1] - max_center[1]) ** 2)
            if dist >= 200:
                new_rects.append(rect)
        for last_box in new_rects:
            cv2.rectangle(img_raw, (last_box[0], last_box[1]), (last_box[2],last_box[3]), (0, 255, 0), 2)
        if save_path is not None:
            cv2.imwrite(save_path, img_raw)
        return new_rects

    elif len(boxes)<1:
        return None

    else:

        cv2.rectangle(img_raw, (boxes[0][0], boxes[0][1]), (boxes[0][2], boxes[0][3]), (0, 255, 0), 2)
        if save_path is not None:
            cv2.imwrite(save_path, img_raw)
        return boxes


    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
if __name__ == '__main__':
    pic_path = "D:/yifangbin/dataset/FFIW10K/FFIW10K_test/target/tarin_mask_pic_save"
    save_path1 = "mask_result/"
    count = 0
    for pic_name in os.listdir(pic_path):
        each_pic_path = os.path.join(pic_path, pic_name)
        save_path = save_path1 + f'{count}.jpg'
        boxes = get_mask_detect(each_pic_path,save_path)
        print(count," ",boxes)
        print("----------------------")
        count+=1