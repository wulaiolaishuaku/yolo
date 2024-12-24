import cv2
import numpy as np
import glob

# 棋盘格规格
board_width = 9
board_height = 6
square_size = 0.02  # 每个小格子的实际大小（单位：米）

# 棋盘格的3D坐标
objp = np.zeros((board_height * board_width, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_width, 0:board_height].T.reshape(-1, 2)
objp *= square_size

# 存储所有检测到的3D和2D点
obj_points = []  # 3D世界坐标
img_points = []  # 2D图像坐标

# 获取标定图像路径
images = glob.glob('calibration_images/*.jpg')  # 你需要指定路径

# 开始标定过程
for image_path in images:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (board_width, board_height), None)
    
    if ret:
        obj_points.append(objp)
        img_points.append(corners)
        
        # 绘制角点
        cv2.drawChessboardCorners(img, (board_width, board_height), corners, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# 执行标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# 输出结果
print("相机内参矩阵：")
print(mtx)

print("畸变系数：")
print(dist)
