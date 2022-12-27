

from imutils.video import VideoStream
import imutils
import time
import cv2
import pyrealsense2 as rs
import numpy as np
import glob

#处理图片，得到标定板位姿
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
rows = 6
cols = 8
CameraMatrix = np.array([[678.9445784,  0.      ,   486.33432117],
 [  0.    ,     679.37426571, 282.83608883],
 [  0.     ,      0.          , 1.        ]])
# CameraMatrix = np.array( [[667.25122397 ,  0.       ,  466.73055786],
#  [  0.       ,  665.15121673, 267.04395368],
#  [  0.       ,    0.          , 1.        ]])
Dist = np.zeros(5)
# Dist = np.array([ 0.25574219 ,-0.60770153 , 0.00147204 ,-0.04129461 , 0.50305759])
# 加载 ArUCo 字典并获取 ArUCo 参数
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
arucoParams = cv2.aruco.DetectorParameters_create()

filename_prefix = 'handeye4/img'
filename_post = '.jpg'
file_count = 28     #index from 1
R_Cam_List = []     #标定板在相机中的旋转矩阵
t_Cam_List = []
R_Rob_List = []     #机器人末端在基坐标系下的旋转矩阵
t_Rob_List = []
i=0
T_B_G = []
T_C_T = []
not_success = []
for i in range(file_count):
    fname = filename_prefix + str(i+1) +filename_post #index from 1. Depend on the file
    color_image = cv2.imread(fname)
    (corners, ids, rejected) = cv2.aruco.detectMarkers(color_image, arucoDict, parameters=arucoParams)
    if len(corners) == 1:
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
            marker_pose = cv2.aruco.estimatePoseSingleMarkers(corners,0.264,CameraMatrix,Dist)    #0.2是指总长度
            R_Camera2Target,_ = cv2.Rodrigues(marker_pose[0])   #旋转矩阵
            t_Camera2Target = marker_pose[1]        #in camera frame
            R_Cam_List.append(R_Camera2Target)
            t_Cam_List.append(t_Camera2Target)

            T = np.zeros((4,4))
            T[-1,-1] = 1
            T[:3,:3] = R_Cam_List[-1]
            T[:3,-1] = t_Cam_List[-1]
            T_C_T.append(T)
            cv2.drawFrameAxes(color_image,CameraMatrix,Dist,marker_pose[0],marker_pose[1].ravel(),0.1)  
            cv2.imwrite('res3/handeye_res'+str(i)+'.jpg', color_image)
            i+=1
    else:
        print(str(i)+'  multipule markers or no marker')
        not_success.append(i)
        i+=1

handeye_robot = np.loadtxt('handeye3/UR_configuration3.txt')
i = 0
print(not_success)
for ro in handeye_robot:    #遍历每一行，读取旋转矩阵和位移向量
    if i in not_success:
        i+=1
        continue
    i+=1
    R_Rob_List.append(np.array([[ro[0],ro[1],ro[2]],[ro[3],ro[4],ro[5]],[ro[6],ro[7],ro[8]]]))
    t_Rob_List.append(np.array([ro[9],ro[10],ro[11]]))
    T = np.zeros((4,4))
    T[-1,-1] = 1
    T[:3,:3] = R_Rob_List[-1]
    T[:3,-1] = t_Rob_List[-1]    
    T_B_G.append(T)
print(i)
R_Rob_List = np.array(R_Rob_List)
t_Rob_List = np.array(t_Rob_List)
R_Cam_List = np.array(R_Cam_List)
t_Cam_List = np.array(t_Cam_List)
a = -1
r,t = cv2.calibrateHandEye(R_gripper2base=R_Rob_List,t_gripper2base=t_Rob_List,R_target2cam=R_Cam_List,t_target2cam=t_Cam_List)

print('t',t)
print('r',r)
T = np.zeros((4,4))
T[-1,-1] = 1
T[:3,:3] = r
T[:3,-1] = t.ravel()   
T_G_C = T
temp = np.array([0,0,0,1]) # in target frame
for i in range(len(T_B_G)):
    print(str(i)+' ',T_B_G[i] @ T_G_C @ T_C_T[i] @ temp)