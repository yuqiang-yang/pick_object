from imutils.video import VideoStream
import imutils
import time
import cv2
import pyrealsense2 as rs
import numpy as np
CameraMatrix = np.array([[678.9445784,  0.      ,   486.33432117],
 [  0.    ,     679.37426571, 282.83608883],
 [  0.     ,      0.          , 1.        ]])
#Dist = np.array([ 2.97303703e-02 , 1.36074896e+00, -2.51609520e-03, -1.30990525e-03,-6.05558291e+00])
Dist = np.zeros(5)
#初始化相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
pipeline.start(config)

# 加载 ArUCo 字典并获取 ArUCo 参数
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
arucoParams = cv2.aruco.DetectorParameters_create()

time.sleep(2.0)

# 循环视频流中的帧
while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())

    # 检测输入帧中的 ArUco 标记
    (corners, ids, rejected) = cv2.aruco.detectMarkers(color_image, arucoDict, parameters=arucoParams)
    # 验证*至少*一个 ArUco 标记被检测到
    if len(corners) > 0:
        # 展平 ArUco ID 列表
        ids = ids.flatten()
        # 循环检测到的 ArUCo 角
        for (markerCorner, markerID) in zip(corners, ids):
            # 提取标记角（始终按左上角、右上角、右下角和左下角顺序返回）
            corners1 = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners1
            # # 将每个 (x, y) 坐标对转换为整数
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            #绘制ArUCo检测的边界框
            cv2.line(color_image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(color_image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(color_image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(color_image, bottomLeft, topLeft, (0, 255, 0), 2)
            # 计算并绘制 ArUco 标记的中心 (x, y) 坐标
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(color_image, (cX, cY), 4, (0, 0, 255), -1)
            marker_pose = cv2.aruco.estimatePoseSingleMarkers(corners,0.2,CameraMatrix,Dist)    #0.2是指总长度
            R = cv2.Rodrigues(marker_pose[0])   #旋转矩阵
            t = marker_pose[1]
            cv2.drawFrameAxes(color_image,CameraMatrix,Dist,marker_pose[0],marker_pose[1],0.1)

    # 显示输出帧
    cv2.imshow( 'img',color_image)
    # 如果按下了 `q` 键，则中断循环
    k = cv2.waitKey(1) & 0xFF
    if k == ord('Q'):
        break

# 做一些清理
cv2.destroyAllWindows()

