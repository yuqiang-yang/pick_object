#用Opencv采集realsense图片 J采集 Q退出
import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

i = 0
while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    cv2.imshow('img',color_image)

    k = cv2.waitKey(1) & 0xFF
    if k== ord('J'):  # 按j保存一张图片
        i += 1
        firename=str('object/img'+str(i)+'.jpg')
        cv2.imwrite(firename, color_image)
        print('写入：',firename)
    if k == ord('Q'):
        break