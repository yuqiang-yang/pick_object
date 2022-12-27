#record chessboard image and the configuration of the robot arm
import pyrealsense2 as rs
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import rtde_receive
ip = "192.168.100.2"
rtde_r = rtde_receive.RTDEReceiveInterface(ip)
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

i = 0
UR_configuration = []
while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    cv2.imshow('img',color_image)

    k = cv2.waitKey(1) & 0xFF
    # print(rtde_r.getActualTCPPose())
    if k== ord('J') or k == ord('j'):  # 按j保存一张图片
        i += 1
        firename=str('object/img'+str(i)+'.jpg')
        end_pose = rtde_r.getActualTCPPose()
        R_BE = R.from_rotvec(end_pose[3:]).as_matrix()
        # print('end_pose',end_pose)
        # print('Rbe',R_BE)
        # print('Rbe',R_BE.ravel())
        UR_configuration.append(np.hstack((R_BE.ravel(),end_pose[:3])))
        print(UR_configuration)
        cv2.imwrite(firename, color_image)
        print('写入：',firename)
    if k == ord('Q') or k == ord('q'):
        if len(UR_configuration) > 0:
            np.savetxt('handeye4/UR_configuration4.txt',np.array(UR_configuration))
        break