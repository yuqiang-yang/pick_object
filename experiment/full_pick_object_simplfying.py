#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append('.')
import numpy as np
import math
import time
import rospy
import qpsolvers as qp
import spatialmath as sm

from trajectory_msgs.msg import JointTrajectory
from math import cos, acos, sin, asin, sqrt, exp, atan, atan2, pi, tan, ceil
from jacobi.ur5e_car_kinematics_class_2 import UR5e_car_kinematics
import matplotlib.pyplot as plt
from trajectory_msgs.msg import *
from control_msgs.msg import *
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import roboticstoolbox as rtb
import rtde_control
import rtde_receive
import threading
import cv2
import scipy.ndimage as ndimage
import pyrealsense2 as rs
import torch
from skimage.draw import circle
from skimage.feature import peak_local_max
from models.ggcnn import GGCNN
from models.ggcnn2 import GGCNN2

from scipy.spatial.transform import Rotation as R
from robotiq_gripper import *


ur5e_car = UR5e_car_kinematics()
car_sub_record = []
#UR state subsciber callback function
#FK to end pose 
Stop_flag = False

ip = "192.168.100.2"
rtde_r = rtde_receive.RTDEReceiveInterface(ip)
for i in range(3):
    try:
        rtde_c = rtde_control.RTDEControlInterface(ip)
        break
    except Exception:
        time.sleep(3)
        print('keep trying to connect RTDE Control')
        if i == 2:
            sys.exit()

model = GGCNN2()
MODEL_FILE = 'ggcnn_epoch_50_cornell'
model.load_state_dict(torch.load('models/epoch_50_cornell_statedict.pt'))
device = torch.device("cuda:0")
gripper = RobotiqGripper()
gripper.connect(ip, 63352)
gripper.activate()
print("Testing gripper...")
gripper.move_and_wait_for_pos(40, 255, 255)
model = model.cuda()
fx = 458.488
fy = 458.477
cx = 324.703
cy = 245.402
JiTing = False
print('RTDE connect successfully')
def press_enter_to_JiTing():#不是完全的急停
    global JiTing
    key=input()
    JiTing=True
    key=input()
    JiTing=True
    key=input()
    JiTing=True
    key=input()
    JiTing=True
    key=input()
    JiTing=True
    sys.exit()  #exit this input thread
listener=threading.Thread(target=press_enter_to_JiTing)
listener.start()
def get_homo(R,t):
    assert R.shape == (3,3) and t.ravel().shape[0] == 3
    T = np.zeros((4,4))
    T[:3,:3] = R
    T[:3,-1] = t.ravel()
    T[-1,-1] = 1
    return T

t_RGB_DEPTH = np.array([-3.97217e-05,-0.01427,0.00537016])
R_RGB_DEPTH =np.array([[0.999997,-0.00112963,0.00210261],
                        [0.0011826,0.999678,-0.0253654],
                        [-0.00207327,0.0253678,0.999676]])
R_g_RGB = np.array([[-0.71379096, -0.6997661,   0.02880736],
                [ 0.70026391, -0.71376734,  0.01290856],
                [ 0.01152878,  0.02938676,  0.99950163]])
t_g_RGB  = np.array([0.11962443, 0.11887692,-0.06505127])
T_RGB_DEPTH = get_homo(R_RGB_DEPTH,t_RGB_DEPTH)
T_g_RGB = get_homo(R_g_RGB,t_g_RGB)
T_g_c = T_g_RGB @ np.linalg.pinv(T_RGB_DEPTH)       #transformation from gripper frame to camera frame
# T_g_c = T_g_RGB @ (T_RGB_DEPTH)       #transformation from gripper frame to camera frame

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
print('222')
colorizer = rs.colorizer()
colorizer.set_option(rs.option.visual_preset,1)
colorizer.set_option(rs.option.min_distance,0)
colorizer.set_option(rs.option.max_distance,1)

pipe_profile = pipeline.start(config)

depth_sensor = pipe_profile.get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.visual_preset, 5) # 5 is short range, 3 is low ambient light

# Execution Timing
class TimeIt:
    def __init__(self, s):
        self.s = s
        self.t0 = None
        self.t1 = None
        self.print_output = False

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, t, value, traceback):
        self.t1 = time.time()
        if self.print_output:
            print('%s: %s' % (self.s, self.t1 - self.t0))


First_Flag = True
prev_mp = np.zeros(2)

def process_depth_image(depth, crop_size, out_size=300, return_mask=False, crop_y_offset=0):
    if depth.ndim == 3:
        imh, imw, _ = depth.shape
    else:
        imh, imw = depth.shape
    with TimeIt('1'):
        # Crop.
        depth_crop = depth[(imh - crop_size) // 2 - crop_y_offset:(imh - crop_size) // 2 + crop_size - crop_y_offset,
                           (imw - crop_size) // 2:(imw - crop_size) // 2 + crop_size]
    # depth_nan_mask = np.isnan(depth_crop).astype(np.uint8)

    # Inpaint
    # OpenCV inpainting does weird things at the border.
    with TimeIt('2'):
        depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        depth_nan_mask = np.isnan(depth_crop).astype(np.uint8)
        depth_far_mask = (depth_crop > 0.7).astype(np.uint8)

    with TimeIt('3'):
        depth_crop[depth_nan_mask==1] = 0
        depth_crop[depth_far_mask==1] = 0

    with TimeIt('4'):
        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        depth_scale = np.abs(depth_crop).max()
        depth_crop = depth_crop.astype(np.float32) / depth_scale  # Has to be float32, 64 not supported.

        with TimeIt('Inpainting'):
            depth_crop = cv2.inpaint(depth_crop, depth_nan_mask, 1, cv2.INPAINT_NS)

        # Back to original size and value range.
        depth_crop = depth_crop[1:-1, 1:-1]
        depth_crop = depth_crop * depth_scale

    with TimeIt('5'):
        # Resize
        depth_crop = cv2.resize(depth_crop, (out_size, out_size), cv2.INTER_AREA)

    if return_mask:
        with TimeIt('6'):
            depth_nan_mask = depth_nan_mask[1:-1, 1:-1]
            depth_far_mask = depth_far_mask[1:-1, 1:-1]
            depth_nan_mask = cv2.resize(depth_nan_mask, (out_size, out_size), cv2.INTER_NEAREST)
            depth_far_mask = cv2.resize(depth_far_mask, (out_size, out_size), cv2.INTER_NEAREST)
        return depth_crop, depth_nan_mask,depth_far_mask
    else:
        return depth_crop

def predict(depth_raw, process_depth=True, crop_size=300, out_size=300, depth_nan_mask=None, crop_y_offset=0, filters=(2.0, 1.0, 1.0)):
    if process_depth:
        depth, depth_nan_mask,depth_far_mask = process_depth_image(depth_raw, crop_size, out_size=out_size, return_mask=True, crop_y_offset=crop_y_offset)
    depth_raw[depth_raw>0.7]=0
    # Inference
    depth_mean = depth.mean()
    depth = np.clip((depth - depth.mean()), -1, 1)
    depthT = torch.from_numpy(depth.reshape(1, 1, out_size, out_size).astype(np.float32)).to(device)
    with torch.no_grad():
        pred_out = model(depthT)

    points_out = pred_out[0].cpu().numpy().squeeze()
    points_out[depth_nan_mask==1] = 0
    points_out[depth_far_mask==1] = 0

    # Calculate the angle map.
    cos_out = pred_out[1].cpu().numpy().squeeze()
    sin_out = pred_out[2].cpu().numpy().squeeze()
    ang_out = np.arctan2(sin_out, cos_out) / 2.0

    width_out = pred_out[3].cpu().numpy().squeeze() * 150.0  # Scaled 0-150:0-1

    # Filter the outputs.
    if filters[0]:
        points_out = ndimage.filters.gaussian_filter(points_out, filters[0])  # 3.0
    if filters[1]:
        ang_out = ndimage.filters.gaussian_filter(ang_out, filters[1])
    if filters[2]:
        width_out = ndimage.filters.gaussian_filter(width_out, filters[2])

    points_out = np.clip(points_out, 0.0, 1.0-1e-3)


    with TimeIt('Control'):
        # Calculate the best pose from the camera intrinsics.
        maxes = None
        ALWAYS_MAX = False  # Use ALWAYS_MAX = True for the open-loop solution.
        global First_Flag
        global prev_mp
        if First_Flag:  # > 0.34 initialises the max tracking when the robot is reset.
            # Track the global max.
            print('eeeeeeeeeeeeeee')
            First_Flag= False
            max_pixel = np.array(np.unravel_index(np.argmax(points_out), points_out.shape))
            #prev_mp = max_pixel.astype(np.int)
        else:
            # Calculate a set of local maxes.  Choose the one that is closes to the previous one.
            maxes = peak_local_max(points_out, min_distance=10, threshold_abs=0.1, num_peaks=1)
            if maxes.shape[0] == 0:
                return
            # maxes = maxes[depth_crop]     
            max_pixel = maxes[np.argmin(np.linalg.norm(maxes - prev_mp, axis=1))]

            # Keep a global copy for next iteration.
            prev_mp = (max_pixel * 0.25 + prev_mp * 0.75).astype(np.int)
        prev_mp = (max_pixel * 0.25 + prev_mp * 0.75).astype(np.int)
        ang = ang_out[max_pixel[0], max_pixel[1]]
        width = width_out[max_pixel[0], max_pixel[1]]

        # Convert max_pixel back to uncropped/resized image coordinates in order to do the camera transform.
        # max_pixel = ((np.array(max_pixel) / 300.0 * crop_size) + np.array([(480 - crop_size)//2, (640 - crop_size) // 2]))
        max_pixel = ((np.array(prev_mp) / 300.0 * crop_size) + np.array([(480 - crop_size)//2, (640 - crop_size) // 2]))

        max_pixel = np.round(max_pixel).astype(np.int)

        point_depth = depth_raw[max_pixel[0], max_pixel[1]]

        # These magic numbers are my camera intrinsic parameters.
        x = (max_pixel[1] - cx)/(fx) * point_depth
        y = (max_pixel[0] - cy)/(fy) * point_depth
        z = point_depth

        if np.isnan(z) or z == 0:
            print('the depth is zero. return!')
            return

    with TimeIt('Draw'):
        # Draw grasp markers on the points_out and publish it. (for visualisation)
        grasp_img = np.zeros((300, 300, 3), dtype=np.uint8)
        grasp_img[:,:,2] = (points_out * 255.0)

        grasp_img_plain = grasp_img.copy()

        rr, cc = circle(prev_mp[0], prev_mp[1], 5)
         
        for i in range(rr.shape[0]):
            if rr[i] >= 300:
                rr[i] = 299
        for i in range(cc.shape[0]):
            if cc[i] >= 300:
                cc[i] = 299
        grasp_img[rr, cc, 0] = 0
        grasp_img[rr, cc, 1] = 255
        grasp_img[rr, cc, 2] = 0
        
        depth = depth + depth_mean
        depth_center = depth[prev_mp[0],prev_mp[1]]
        for i in range(rr.shape[0]):
            rr[i] = int(rr[i]*crop_size/300.0)
            cc[i] = int(cc[i]*crop_size/300.0)
    # return points_out, ang_out[prev_mp[0],prev_mp[1]], width_out[prev_mp[0],prev_mp[1]], grasp_img,depth_center,rr,cc,x,y,z
    return points_out, ang,width, grasp_img,depth_center,rr,cc,x,y,z


#car state subscriber callback function
def car_sub_callback(odom):
    global Stop_flag

    x = odom.pose.pose.orientation.x
    y = odom.pose.pose.orientation.y
    z = odom.pose.pose.orientation.z
    w = odom.pose.pose.orientation.w
    theta = math.atan2(2*(w*z+x*y),1-2*(y**2+z**2))
    if Stop_flag == True:
        sys.exit()
    car_sub_record.append([odom.pose.pose.position.x, odom.pose.pose.position.y, theta ,odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z])


#given tow axieangle, calculate the orientation error
def get_delta_angle(Rc,Rd):
    
    delta_R=np.matmul(Rd,np.transpose(Rc))
    try:
        d_phi=acos((delta_R[0,0]+delta_R[1,1]+delta_R[2,2]-1)/2)
    except:
        print('acos error')
        return np.array([0,0,0])
    s_d_phi=sin(d_phi)
    if abs(s_d_phi)<0.00001:
        return 0.0,0.0,0.0
        pass
    else:
        k=np.array([delta_R[2,1]-delta_R[1,2],delta_R[0,2]-delta_R[2,0],delta_R[1,0]-delta_R[0,1]])*1/(2*s_d_phi)
        delta_phi=k*d_phi
        return delta_phi

def safe_area(Ain,xi,xs,xc,gain):
    '''
    Ain, the i row of Jacobian
    xi, the influence distance
    xs, the maximum safe distance
    xc, the current distance

    return Ain(a row), bin(a scalar)
    '''
    if xc > 0 and xc > xi:
        b = gain*(xs - xc)/(xs - xi)
        # b = 0 if b < 0 else b
        return Ain,b
    elif xc < 0 and -xc > xi:
        b = -gain*(xs + xc)/(xi - xs)
        # b = 0 if b > 0 else b
        return -Ain,b        
    else:
        return np.zeros(6), 0

 
def get_image_and_detect():

    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    depth = np.asanyarray(depth_frame.get_data())*0.00025

    imh, imw, _ = depth_image.shape
    crop_size = 300
    depth_crop = depth_image[(imh - crop_size) // 2:(imh - crop_size) // 2 + crop_size,
                    (imw - crop_size) // 2:(imw - crop_size) // 2 + crop_size]

    depth_raw_crop = depth[(imh - crop_size) // 2:(imh - crop_size) // 2 + crop_size,
                    (imw - crop_size) // 2:(imw - crop_size) // 2 + crop_size]                    
    # color_image = np.asanyarray(color_frame.get_data())
    res = predict(depth,filters=(10,2,2),crop_size=crop_size) 
    if res != None:
        points_out, ang_out, width_out, grasp_img, depth_center ,rr ,cc,xx,yy,zz= res 
    else:
        return None
    return points_out, ang_out, width_out, grasp_img, depth_crop ,rr ,cc,xx,yy,zz  
First_QP = True
θε0 = 0
def QP_IK(T_target,T_end,jacobi):
    global First_QP
    global θε0

    eT = np.linalg.pinv(T_end) @ T_target
    et = np.sum(np.abs(eT[:3,-1]))
    # et = np.sum(np.abs(T_target[:3,-1]-T_end[:3,-1]))
    print('et',et)
    # penerty term for speed control(UR)
    Y = 0.01

# quadratic part (DOF + slack term for the EE velocity tracking error)
    Q = np.eye(2 + ur_robot.n + 6)
    Q[:2 + ur_robot.n, :2 + ur_robot.n] *= Y
    Q[0, 0] *= 10/et
    Q[1, 1] *= 10/et
    Q[-7, -7] *= 99999 #dynamic weight
    Q[-6:,-6:] = (1 /et) * np.eye(6)
# linear part
    c = np.concatenate((np.zeros(2), -1 *ur_robot.jacobm(arm_state).reshape((ur_robot.n,)), np.zeros(6)))
    # c = np.concatenate((np.zeros(2), -1/et*ur_robot.jacobm(arm_state).reshape((ur_robot.n,)), np.zeros(6)))

    kε = 0.2
    bTe = ur_robot.fkine(arm_state).A
    θε = math.atan2(-bTe[0, -1], bTe[1, -1]) + 0    #the x axie of UR is right direction, the y axie of UR is forward direction
    if First_QP:
        First_QP = False
        θε0 = θε
    ε = kε * (θε - θε0) 
    c[1] = -ε

    # c[2] = ε

# position servo in caresian space
    # position servo in caresian space
    v = np.zeros(6)
    v[:3] = 1*(T_target[:3,-1]-T_end[:3,-1])      
    if np.linalg.norm(v[:3]) < 0.05:
        v[:3] *= 0.05/np.linalg.norm(v[:3])
        print('norm',np.linalg.norm(v[:3]))

    w = get_delta_angle(T_end[:3,:3],T_target[:3,:3])
    v[3:] = w*2
    if et < 1:
        v[3:] = np.max([1,et/1])*w
# equality constraints
    Aeq = np.c_[jacobi, np.eye(6)]
    beq = v.reshape((6,))

# inequality constraints
    Ain = np.zeros((2 + ur_robot.n + 6, 2 + ur_robot.n + 6))
    bin = np.zeros(2 + ur_robot.n + 6)

    Ain[2,2:2+ur_robot.n],bin[2] = safe_area(ur_robot.jacob0(arm_state)[1],0.1,0.55,bTe[1, -1],1)     #x axie range limit
    Ain[3,2:2+ur_robot.n],bin[3] = safe_area(-ur_robot.jacob0(arm_state)[0],0.1,0.55,-bTe[0, -1],1)   #y axie range limit
    # print('Ain',Ain[2,2:2+ur_robot.n])
    # print('bin',bin)
# velocity damper to limit the joint position
    ps = 0.1    # minimum angle in radians
    pi = 0.9    # influence angle in radians
    # Ain[2: ur_robot.n+2, 2: ur_robot.n+2], bin[2: ur_robot.n+2] = ur_robot.joint_velocity_damper(ps, pi, ur_robot.n)
    # print('bin',bin)
# velocity bound
    lb = -2*np.ones(2 + ur_robot.n + 6)
    ub = 2*np.ones(2 + ur_robot.n + 6)


#solve QP
    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq,solver='quadprog')
    if et > 1:
        qd *= 0.5 / et
    else:
        qd *= 0.7
    print('θε',θε)
    return qd,et
# # test the grasp
# if __name__ == "__main__":
#     print('enter main func')
#     joint_init = np.array([-256.74,-88.31,93.49,-95.14,270.32,-123.62])/57.3
#     cnt = 0
#     desk_height = 0.0115
#     for i in range(5):
#         rtde_c.moveJ(joint_init,0.5,0.2)
#         gripper.move_and_wait_for_pos(0, 120, 120)  
#         while JiTing == False:
#             cnt +=1
#             res = get_image_and_detect()
#             if res != None:
#                 points_out, ang_out, width_out, grasp_img, depth_crop,rr,cc,xx,yy,zz = res
#                 # print('zz',zz)
#             else:
#                 print('no detect!!')
#                 continue
#             cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
#             cv2.imshow('RealSense', grasp_img)
#             depth_crop[rr,cc,0] = 255
#             depth_crop[rr,cc,1] = 0
#             depth_crop[rr,cc,2] = 0
            
#             cv2.imshow('RealSense2', depth_crop)
#             cv2.waitKey(1)   
#             P_c = np.array([xx,yy,zz,1]).reshape(-1,1)  #in camera frame


#             end_pose = rtde_r.getActualTCPPose()
#             R_b_g = R.from_rotvec(end_pose[3:]).as_matrix()

#             t_b_g = np.array(end_pose[:3])
#             T_b_g = get_homo(R_b_g,t_b_g)

#             P_b = T_b_g @ T_g_c @ P_c   # in base frame
            
#             current_euler = R.from_rotvec(end_pose[3:]).as_euler(seq = 'xyz')
#             current_euler[2] += ang_out
#             Pose_b = R.from_euler('xyz',current_euler).as_rotvec()

 
#             if cnt % 20 == 0:
#                 print('grip position in base:',P_b[:-1].ravel())
#                 print('grip pose in base:',Pose_b)
#                 print('current_euler',current_euler)
#                 print('ang_out',ang_out)
#                 print('height', P_b[2])
#             if cnt % 5 == 0 and abs(P_b[2] - desk_height) < 0.01:
#                 print('on the desk')
#                 continue
#         JiTing = False
#         P_c = np.array([xx,yy,zz,1]).reshape(-1,1)  #in camera frame
#         end_pose = rtde_r.getActualTCPPose()
#         R_b_g = R.from_rotvec(end_pose[3:]).as_matrix()

#         t_b_g = np.array(end_pose[:3])
#         T_b_g = get_homo(R_b_g,t_b_g)

#         P_b = T_b_g @ T_g_c @ P_c   # in base frame
        
#         current_euler = R.from_rotvec(end_pose[3:]).as_euler(seq = 'xyz')
#         current_euler[2] += ang_out
#         Pose_b = R.from_euler('xyz',current_euler).as_rotvec()

#         target_pose = np.zeros(6)
#         target_pose[:3] = P_b[:-1].ravel()
#         target_pose[2] = 0.264
#         target_pose[3:] = Pose_b.ravel()
#         rtde_c.servoStop(0.5)
#         rtde_c.moveL(target_pose,0.1,0.1)  
#         griper_status = gripper.move_and_wait_for_pos(255, 120, 120)[1]  == RobotiqGripper.ObjectStatus.STOPPED_INNER_OBJECT 
#         print('gripper status',griper_status)     
#         time.sleep(2)  
#     pipeline.stop()
#     sys.exit()
if __name__ == "__main__":
    desk_height = 0.0115
    #init ros publisher 
    print('enter main func')
    rospy.init_node("Peter_Control")
    base_pub = rospy.Publisher ("/mobile_base/cmd_vel", Twist,queue_size=0) 
    base_sub = rospy.Subscriber("/mobile_base/odom",Odometry,car_sub_callback)


    ur_robot = rtb.models.UR5()

    time.sleep(0.5)     #need to wait for a short time. If not, the node cannot work
    #prepare for publish
    msg = JointTrajectory()
    vel = Twist()
    np.set_printoptions(threshold=np.inf)   #for convenient showing

    #set the car and UR initial state
    vel.linear.x = 0
    base_pub.publish(vel)
    joint_init0 = np.array([-4.78,-2.14,2.01,-1.51,4.75,-130.62/57.3])
    joint_init = np.array([-4.78,-2.14,2.01,-2.54,4.78,-130.62/57.3])

    time.sleep(1)

    # given a target pose T_target 4*4
    arm_state = rtde_r.getActualQ()

    T_target0 =  np.array([[ 0.79784442,  0.59861218,  0.07146851,  0.49482654],
    [ 0.5979412,  -0.80086795,  0.03281529,  0.11875098],
    [ 0.07688047,  0.01655247, -0.99690291,  1.07793187],
    [ 0.,          0.,          0.,          1.        ]])
    rx0_in_arm = np.array([0.11859253125685025, -0.27515036005868604, 0.4786321147886782, 2.82222442, -1.3805239505646434, 0.0])
    # T_target0[:3,:3] = R.from_rotvec([2.923,-1.151,0]).as_matrix()
    rtde_c.moveJ(joint_init,0.5,0.2)
    arm_state = rtde_r.getActualQ()
    T_target = T_target0.copy()
    T_target[0,-1] = 1.5
    T_target[2,-1] = T_target0[2,-1] - 0.0

    state_flag = 0  #state machine 

    dt = 0.1
    car_pre_linear_vel = 0
    car_pre_angular_vel = 0
    car_linear_acc = 0.4
    car_angular_acc = 0.4
    car_linear_vel = 1
    car_angular_vel = 1

    arm_max_acc = 2
    arm_max_vel = 1
    ttt = 100
    last_arm_vel = np.zeros(6)
    # get current pose T_current
    success_cnt = 0
    scan_cnt = 0
    print('enter the loop')
    while JiTing == False:

        arm_state = rtde_r.getActualQ()
        car_state = car_sub_record[-1][:3]
        T_end = ur5e_car.get_end_effector_posture(car_state + arm_state)
        jacobi = ur5e_car.get_jacobian_lx(car_state + arm_state)
        jacobi = np.array(jacobi)
        T_end = np.array(T_end)
        qd = np.zeros(8)
        if state_flag == 0:
            qd,et = QP_IK(T_target,T_end,jacobi)
            if et < 0.05:
                rtde_c.servoStop(0.2)        
                time.sleep(0.5)
                end_state = rtde_r.getActualTCPPose()
                arm_state = rtde_r.getActualQ()
                car_state = car_sub_record[-1][:3]                
                T_end = np.array(ur5e_car.get_end_effector_posture(car_state + arm_state))

                end_state[0] += T_target[1,-1] - T_end[1,-1]
                end_state[1] -= T_target[0,-1] - T_end[0,-1]
                end_state[2] += T_target[2,-1] - T_end[2,-1]
                end_state[3:] = rx0_in_arm[3:]
                rtde_c.moveL(end_state,0.2,0.1)
                state_flag = 1
                ttt = 10
                end_state = rtde_r.getActualTCPPose()
                arm_state = rtde_r.getActualQ()
                car_state = car_sub_record[-1][:3] 
                joint_reset = arm_state
                qd = np.zeros(8)
        elif state_flag == 1:
            no_detect_flag = False
            res = get_image_and_detect()
            if res != None:
                points_out, ang_out, width_out, grasp_img, depth_crop ,rr ,cc,xx,yy,zz = res
            else:
                print('no detect!!')
                no_detect_flag = True
            if not no_detect_flag:
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', grasp_img)
                depth_crop[rr,cc,0] = 255
                depth_crop[rr,cc,1] = 0
                depth_crop[rr,cc,2] = 0

                cv2.imshow('RealSense2', depth_crop)
                cv2.waitKey(1)

                P_c = np.array([xx,yy,zz,1]).reshape(-1,1)  #in camera frame

                end_pose = rtde_r.getActualTCPPose()
                R_b_g = R.from_rotvec(end_pose[3:]).as_matrix()

                t_b_g = np.array(end_pose[:3])
                T_b_g = get_homo(R_b_g,t_b_g)

                P_b = T_b_g @ T_g_c @ P_c   # in base frame
                
                current_euler = R.from_rotvec(end_pose[3:]).as_euler(seq = 'xyz')
                current_euler[2] += ang_out
                Pose_b = R.from_euler('xyz',current_euler).as_rotvec()

    
                print('grip position in base:',P_b[:-1].ravel())
                print('grip pose in base:',Pose_b)
                # print('current_euler',current_euler)
                print('ang_out',ang_out)
            if  abs(P_b[2] - desk_height) < 0.01 or no_detect_flag:
                print('on the desk')
                print()
                end_state = rtde_r.getActualTCPPose()
                scan_cnt +=1

                if scan_cnt % 100 == 20:
                    end_state[0] += 0.1
                elif scan_cnt % 100 == 40:
                    end_state[1] -= 0.1
                elif scan_cnt % 100 == 60:
                    end_state[0] -= 0.25
                elif scan_cnt %100 == 80:
                    end_state[1] += 0.1
                elif scan_cnt %100 == 0:
                    end_state[0] += 0.15
                else: 
                    continue
                rtde_c.servoStop(0.3)  
                end_state[3:] = rx0_in_arm[3:]
                # rtde_c.servoL(end_state,0,0,0.05,0.1,200)
                rtde_c.moveL(end_state,0.1,0.1)
                arm_state = rtde_r.getActualQ()
                continue
            ttt -=1

            if zz > 0.1 and zz < 0.6 and ttt <= 0:
                scan_cnt = 0
                state_flag = 2
        elif state_flag == 2:
            res = get_image_and_detect()
            if res != None:
                points_out, ang_out, width_out, grasp_img, depth_crop ,rr ,cc,xx,yy,zz = res
            else:
                print('no detect!!')
                continue
            P_c = np.array([xx,yy,zz,1]).reshape(-1,1)  #in camera frame
            end_pose = rtde_r.getActualTCPPose()
            R_b_g = R.from_rotvec(end_pose[3:]).as_matrix()

            t_b_g = np.array(end_pose[:3])
            T_b_g = get_homo(R_b_g,t_b_g)

            P_b = T_b_g @ T_g_c @ P_c   # in base frame
            
            current_euler = R.from_rotvec(end_pose[3:]).as_euler(seq = 'xyz')
            current_euler[2] += ang_out
            Pose_b = R.from_euler('xyz',current_euler).as_rotvec()

            target_pose = np.zeros(6)
            target_pose[:3] = P_b[:-1].ravel()
            target_pose[1] += 0.015
            target_pose[2] = 0.266
            target_pose[3:] = Pose_b.ravel()
            rtde_c.servoStop(0.5)
            rtde_c.moveL(target_pose,0.2,0.2)  
            arm_state = rtde_r.getActualQ()
            griper_res = gripper.move_and_wait_for_pos(255, 120, 120)  
            griper_status = griper_res[1]== RobotiqGripper.ObjectStatus.STOPPED_INNER_OBJECT  
            gripper_width = griper_res[0]
            print('gripper status',griper_status)     
            print('gripper width',gripper_width)
            time.sleep(1)  
            # pick the object
            rtde_c.servoStop(0.3)

            if griper_status and gripper_width < 225:
                rtde_c.moveJ(joint_init,0.5,0.5)
                arm_state = rtde_r.getActualQ()
                state_flag = 3
                T_target = T_target0.copy()
                T_target[0,-1] = -1.5
                T_target[1,-1] += (success_cnt % 7)*0.07
                T_target[2,-1] =  T_target0[2,-1] -0.025
                T_target[:3,:3] = T_target[:3,:3]* sm.SE3.Rz(np.pi)
            else:
                rtde_c.moveJ(joint_reset,0.5,0.3)
                arm_state = rtde_r.getActualQ()
                gripper.move_and_wait_for_pos(40, 255, 255)
                ttt = 10
                state_flag = 1
                
        elif state_flag == 3:

            print('target',T_target)
            qd,et = QP_IK(T_target,T_end,jacobi)
            print(('end',T_end))
            if et < 0.05:
                rtde_c.servoStop(0.3)        
                time.sleep(0.5)
                end_state = rtde_r.getActualTCPPose()
                arm_state = rtde_r.getActualQ()
                car_state = car_sub_record[-1][:3]                
                T_end = np.array(ur5e_car.get_end_effector_posture(car_state + arm_state))

                end_state[0] += T_target[1,-1] - T_end[1,-1]
                end_state[1] -= T_target[0,-1] - T_end[0,-1]
                end_state[2] += T_target[2,-1] - T_end[2,-1]
                rtde_c.moveL(end_state,0.1,0.1)
                arm_state = rtde_r.getActualQ()
                gripper.move_and_wait_for_pos(40, 255, 255)
                time.sleep(1)
                rtde_c.moveJ(joint_init,1,0.5)
                state_flag = 0
                arm_state = rtde_r.getActualQ()
                T_target = T_target0.copy()
                T_target[0,-1] = 1.5
                T_target[2,-1] = T_target0[2,-1] - 0.0
                success_cnt +=1
                qd = np.zeros(8)

        for i in range(6):
            if abs(qd[2+i] - last_arm_vel[i]) > arm_max_acc:    #acc
                qd[2+i] = last_arm_vel[i] + arm_max_acc*dt*np.sign(qd[2+i]-last_arm_vel[i])
            if abs(qd[2+i]) > arm_max_vel:
                qd[2+i] = arm_max_vel*np.sign(qd[2+i])
        qd[7] = 0
        rtde_c.servoJ(arm_state + qd[2:8]*dt,0,0,0.05,0.15,200)
        last_arm_vel = qd[2:8]


        vel.linear.x = qd[0]
        vel.angular.z = qd[1]
        #linear acc limits. The angular acc limits is considered in time optimal 1d tracking
        if abs(car_pre_linear_vel-vel.linear.x) > car_linear_acc*dt:
            vel.linear.x = car_pre_linear_vel + np.sign(vel.linear.x-car_pre_linear_vel)*car_linear_acc*dt
        car_pre_linear_vel = vel.linear.x
        car_pre_angular_vel = vel.angular.z
        if abs(vel.linear.x) > car_linear_vel:
            vel.linear.x = car_linear_vel*np.sign(vel.linear.x)
            # print('the linear vel is too large')
        if abs(vel.angular.z) > car_angular_vel:
            vel.angular.z = car_angular_vel*np.sign(vel.angular.z)
            # print('the angular vel is too large')
        base_pub.publish(vel)  
        print('state',state_flag)
        # print('qd',qd)
        # print('car',vel.linear.x,' ',vel.angular.z)
        # print('manipula',ur_robot.manipulability(arm_state))
        print()
        time.sleep(dt)
    rtde_c.disconnect()
    pipeline.stop()
    

