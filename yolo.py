import cv2
import torch
import numpy as np
import threading 
from filterpy.kalman import KalmanFilter

# 加载YOLOv5预训练模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')  # 使用yolov5n（轻量化模型）

# 打开摄像头
cap = cv2.VideoCapture(0)

# 用于存储视频帧的全局变量
frame = None
frame_lock = threading.Lock()

# 目标追踪相关变量
track_person = False  # 是否已经锁定目标
person_bbox = None    # 锁定目标的边界框

# 卡尔曼滤波器初始化
kf = KalmanFilter(dim_x=4, dim_z=2)  # 4维状态，2维观测（位置）
kf.x = np.array([0, 0, 0, 0])  # 初始状态: [x, y, vx, vy]
kf.P *= 1000.  # 初始协方差矩阵
kf.F = np.array([[1, 0, 1, 0],  # 状态转移矩阵
                 [0, 1, 0, 1],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])
kf.H = np.array([[1, 0, 0, 0],  # 观测矩阵
                 [0, 1, 0, 0]])
kf.R = np.array([[10, 0],  # 观测噪声矩阵
                 [0, 10]])
kf.Q = np.array([[1, 0, 0, 0],  # 过程噪声矩阵
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])

# 计算IOU（交并比）函数
def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # 计算两个框的交集和并集
    x1_int = max(x1, x2)
    y1_int = max(y1, y2)
    x2_int = min(x1 + w1, x2 + w2)
    y2_int = min(y1 + h1, y2 + h2)

    # 计算交集面积
    inter_area = max(0, x2_int - x1_int) * max(0, y2_int - y1_int)
    # 计算并集面积
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0

# 定义全局的 running 变量
running = True

# 图像捕获线程
def capture_frame():
    global frame
    while running:
        ret, captured_frame = cap.read()
        if ret:
            with frame_lock:
                frame = captured_frame
        else:
            print("无法捕获视频帧")
            break

# YOLO推理线程
def yolo_inference():
    global frame, track_person, kf, person_bbox
    while running:
        with frame_lock:
            if frame is None:
                continue
            img = frame[..., ::-1]  # 转换为RGB格式
            # 使用YOLO模型进行推理
            results = model(img)

            # 获取检测结果（边界框和类别）
            boxes = results.xywh[0]  # [x, y, w, h, confidence, class]
            labels = results.names   # 类别名称

            best_bbox = None
            max_iou = 0

            # 如果正在追踪，基于IOU进行匹配
            if track_person:
                for box in boxes:
                    class_id = int(box[5])  # 类别id
                    if class_id == 0:  # person类别的id是0
                        confidence = box[4]
                        if confidence > 0.5:
                            # 计算IOU来进行匹配
                            iou = compute_iou(person_bbox, box[:4])
                            if iou > max_iou:
                                max_iou = iou
                                best_bbox = box[:4]

            # 如果找到匹配的目标
            if best_bbox is not None:
                x, y, w, h = best_bbox
                person_bbox = [x, y, w, h]
                # 使用卡尔曼滤波器更新状态
                kf.predict()  # 预测目标位置
                kf.update(np.array([x, y]))  # 用新的观测值更新卡尔曼滤波器的状态
            else:
                # 如果没有找到匹配的目标，尝试重新初始化追踪
                for box in boxes:
                    class_id = int(box[5])
                    if class_id == 0:  # person类别的id是0
                        confidence = box[4]
                        if confidence > 0.5:
                            x, y, w, h = box[:4]
                            person_bbox = [x, y, w, h]
                            kf.x = np.array([x, y, 0, 0])  # 初始化位置和速度为0
                            track_person = True
                            break

            # 获取卡尔曼滤波器的预测位置
            pred_x, pred_y = kf.x[0], kf.x[1]
            # 假设相机内参
            f_x = 640  # 焦距（单位：像素）
            f_y = 640  # 焦距（单位：像素）
            c_x = 320  # 光心横坐标（单位：像素）
            c_y = 240  # 光心纵坐标（单位：像素
            real_height = 1.7  # 例如目标的实际高度为1.7米
            # 计算边界框的绝对坐标
            if track_person and person_bbox is not None:
                x, y, w, h = person_bbox
                center_x = int(x)
                center_y = int(y)

                # 计算相机坐标系下的x, y位置
                X_cam = (center_x - c_x) / f_x
                # Y_cam = (center_y - c_y) / f_y
                image_height = h  # 使用目标的像素高度
                distance = (real_height * f_y) / image_height  # 基于相似三角形计算距离
   
    # 打印相机坐标系下的位置信息  
                print(f"相机坐标系下的x, y位置: X = {X_cam:.2f}, Y = {distance:.2f}")
                # 计算框的左上角和右下角坐标（像素）
                top_left = (int(x - w / 2), int(y - h / 2))
                bottom_right = (int(x + w / 2), int(y + h / 2))

                # 绘制目标边界框（红色矩形框）
                cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)  # 红色矩形框

                # 绘制狙击枪锁定的样式：交叉准心
                center_x = int(x)
                center_y = int(y)
                crosshair_size = 20  # 锁定样式的大小
                line_thickness = 2  # 线宽

                # 绘制垂直线
                cv2.line(frame, (center_x, center_y - crosshair_size), (center_x, center_y + crosshair_size), (0, 255, 0), line_thickness)
                # 绘制水平线
                cv2.line(frame, (center_x - crosshair_size, center_y), (center_x + crosshair_size, center_y), (0, 255, 0), line_thickness)

                # 绘制准心交点上的小圆（小圆的半径小于准心线的长度）
                circle_radius = 10  # 圆的半径，可以根据需要调整

                cv2.circle(frame, (center_x, center_y), circle_radius, (0, 255, 0), -1)  # 在交点位置绘制一个小圆

            # 显示结果
            cv2.imshow('YOLO Real-Time Detection with Kalman and IOU', frame)

        # 按键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 创建图像捕获线程
capture_thread = threading.Thread(target=capture_frame, daemon=True)

# 创建YOLO推理线程
inference_thread = threading.Thread(target=yolo_inference, daemon=True)

# 启动线程
capture_thread.start()
inference_thread.start()

# 等待线程完成
capture_thread.join()
inference_thread.join()

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()
