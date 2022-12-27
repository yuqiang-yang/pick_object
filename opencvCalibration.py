import cv2
import numpy as np
import glob
import sys
from scipy.spatial.transform import Rotation as R
# In a 2d image, the x axie points to the right while the y axie points to the bottom
# the chessborad frame's direction is determined by the rows and cols
# the origin is determined by the opencv the drawChessboard is started by red and then origin and then blue
# we can know the origin from the result image
# so !!!!!!!!!! remember to check the origin of different chessboard picture found by opencv is the same !!!!!
# because in a hand-in-eye configuration, the transformation b^T_t should be the same for different images!

# In handeye calibration, it's important to ensure that the b^T_g and c^T_t is corresponding.
# So, don't use the glob to read the image because it's unordered.

# The number of the calibration image should be larger than 20. To ensure high accuracy, it
# is better to make the camera close to the image so as to take high-quality image.

# In opencv, b^T_g is named T_gripper_to_base. It transfers a point in the gripper frame into 
# the base frame(b^P = b^T_g*g^P ). It also repesents the origin and the axis of the gripper frame
# in the base frame.

# The code is a hand-in-eye calibration. To make it work for a hand-to-eye configuration, change the 
# b^T_g(i.e. gripper to base) to the g^T_b.

# 标定的 L515（跟MATLAB结果差不多）    we can also get the intrin. by rs-sensor-control 
# 内参矩阵:
#  [[678.9445784    0.         486.33432117]
#  [  0.         679.37426571 282.83608883]
#  [  0.           0.           1.        ]]
# 畸变系数:
#  [[ 2.97303703e-02  1.36074896e+00 -2.51609520e-03 -1.30990525e-03
#   -6.05558291e+00]]

# g_R_c [[-0.99930965  0.03603957 -0.00902068]
#  [-0.0361714  -0.99923445  0.01490461]
#  [-0.00847662  0.01522061  0.99984823]]
# g_t_c [[ 0.00581441]
#  [ 0.16148753]
#  [-0.05332651]]

# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

# 获取标定板角点的位置
Calibrate_Handeye = True
rows = 6     # more specifically, this determines the x direction of the chessboard
cols = 8    # more specifically, this determines the y direction of the chessboard
realworld_size = 0.035  
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
#the origin is the left-bottom point of the chessboard
# print(objp)
objp *= realworld_size
obj_points = []  # 存储3D点
img_points = []  # 存储2D点

filename_prefix = 'handeye4/img'
filename_post = '.jpg'
file_count = 28     #index from 1

i=0
for i in range(file_count):
    fname = filename_prefix + str(i+1) + filename_post
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)
    if ret:
        obj_points.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
        if corners2[0,0,0] < corners2[-1,0,0]:  # ensure the origin of different image to be the same
            print('img'+str(i)+'reverse!')
            corners2 = np.flipud(corners2)
        if [corners2]:
            img_points.append(corners2)
        cv2.drawChessboardCorners(img, (rows, cols), corners2, ret)  # 记住，OpenCV的绘制函数一般无返回值
        i += 1
        cv2.imwrite('res/corner'+str(i)+'.png', img)
        cv2.waitKey(0)
    else:
        print(str(fname))   
#cv2.destroyAllWindows()

# 标定
obj_points = np.array(obj_points)
img_points = np.array(img_points)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

print("ret:", ret)
print("内参矩阵:\n", mtx) # 内参数矩阵
print("畸变系数:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
# print("旋转向量:\n", np.array(rvecs))  # 旋转向量  # 外参数  in the camera frame
# print("平移向量:\n", np.array(tvecs))  # 平移向量  # 外参数  in the camera frame

cv2.waitKey(0)
cv2.destroyAllWindows()
if Calibrate_Handeye == False:
    sys.exit()   

handeye_robot = np.loadtxt('handeye4/UR_configuration4.txt')
R_Cam_List = []     #标定板在相机中的旋转矩阵
t_Cam_List = []
R_Rob_List = []     #机器人末端在基坐标系下的旋转矩阵
t_Rob_List = []
T_B_G = []  # gripper in base frame
T_C_T = []  # target in camera frame
for i in range(len(rvecs)):
    T = np.zeros((4,4))
    T[:3,:3] = R.from_rotvec(rvecs[i].ravel()).as_matrix()
    T[:3,-1] = tvecs[i].ravel()
    T[-1,-1] = 1
    T_C_T.append(T)
    R_Cam_List.append(T[:3,:3])
    t_Cam_List.append(T[:3,-1])

for ro in handeye_robot:    #遍历每一行，读取旋转矩阵和位移向量
    R_Rob_List.append(np.array([[ro[0],ro[1],ro[2]],[ro[3],ro[4],ro[5]],[ro[6],ro[7],ro[8]]]))
    t_Rob_List.append(np.array([ro[9],ro[10],ro[11]]))
    T = np.zeros((4,4))
    T[:3,:3] = R_Rob_List[-1]
    T[:3,-1] = t_Rob_List[-1]
    T[-1,-1] = 1
    T_B_G.append(T)
    
R_Rob_List = np.array(R_Rob_List)
t_Rob_List = np.array(t_Rob_List)
R_Cam_List = np.array(R_Cam_List)
t_Cam_List = np.array(t_Cam_List)

R_gc,t_gc = cv2.calibrateHandEye(R_Rob_List,t_Rob_List,R_Cam_List,t_Cam_List) #the camera frame transformation in the gripper frame
print('g_R_c',R_gc)
print('g_t_c',t_gc)

#test the handeye result
print('############test the calibration result##############')

T = np.zeros((4,4))
T[:3,:3] = R_gc
T[:3,-1] = t_gc.ravel()
T[-1,-1] = 1
res = []
T_G_C = T
print(T_G_C)
temp = np.array([0,0,0,1]) # in target frame
for i in range(len(T_B_G)):
    print(str(i)+' ',T_B_G[i] @ T_G_C @ T_C_T[i] @ temp)    #tranformation of the chessboard origin from the target frame to the base frame 
