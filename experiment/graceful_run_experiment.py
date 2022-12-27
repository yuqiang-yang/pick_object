#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append('.')
import numpy as np
import math
import time
import random
import rospy
import qpsolvers as qp

from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from math import cos, acos, sin, asin, sqrt, exp, atan, atan2, pi, tan, ceil
from jacobi.ur5e_car_kinematics_class_2 import UR5e_car_kinematics
import matplotlib.pyplot as plt
from trajectory_msgs.msg import *
from control_msgs.msg import *
from geometry_msgs.msg import Twist, PoseStamped
from gazebo_msgs.msg import LinkStates
from nav_msgs.msg import Odometry
import roboticstoolbox as rtb
from scipy.spatial.transform import Rotation
import rtde_control
import rtde_receive
import threading
ur5e_car = UR5e_car_kinematics()
ur_end_record = []

car_sub_record = []
car_coor_record=[]
ur_coor_record=[]
wb_record = []
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
JiTing = False
print('RTDE connect successfully')
def press_enter_to_JiTing():#不是完全的急停
    global JiTing
    key=input()
    JiTing=True
    key=input()
    JiTing=True
    sys.exit()  #exit this input thread
listener=threading.Thread(target=press_enter_to_JiTing)
listener.start()

#given tow axieangle, calculate the orientation error
def get_delta_angle(Ac,Ad):
    Rc = axixAngel2R(Ac)
    Rd = axixAngel2R(Ad)

    delta_R=np.matmul(Rd,np.transpose(Rc))
    d_phi=acos((delta_R[0,0]+delta_R[1,1]+delta_R[2,2]-1)/2)
    s_d_phi=sin(d_phi)
    if abs(s_d_phi)<0.00001:
        return 0.0,0.0,0.0
        pass
    else:
        k=np.array([delta_R[2,1]-delta_R[1,2],delta_R[0,2]-delta_R[2,0],delta_R[1,0]-delta_R[0,1]])*1/(2*s_d_phi)
        delta_phi=k*d_phi
        return delta_phi


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


if __name__ == "__main__":

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
    joint_init=np.array([-90+90, -90, -90, -90, 90, -90])/57.3
    joint_init=np.array([-1.51,-0.38,-2.14,-2.14,1.63,-90/57.3])
    joint_init=np.array([-84.08, -46.82, -95.58, -127.65, 90.31, -157.37])/57.3
   
    print('before',rtde_r.getActualQ())
    rtde_c.moveJ(joint_init,0.5,0.1)
    print('init')
    time.sleep(2)
    car_sub_record.clear()
    time.sleep(0.5)
    
    # given a target pose T_target 4*4
    arm_state = rtde_r.getActualQ()
    T_target  = ur5e_car.get_end_effector_posture(car_sub_record[-1][:3] + arm_state)
    T_target = np.array(T_target)
    
    T_target[0, -1] -= 3
    T_target[1, -1] += 0
    T_target[2, -1] -= 0.1
    dt = 0.1
    car_pre_linear_vel = 0
    car_pre_angular_vel = 0
    car_linear_acc = 0.3
    car_angular_acc = 0.5
    car_linear_vel = 0.2
    car_angular_vel = 0.48

    arm_max_acc = 1
    arm_max_vel = 1
    last_arm_vel = np.zeros(6)
    # get current pose T_current
    print('enter the loop')
    while JiTing == False:

        arm_state = rtde_r.getActualQ()
        car_state = car_sub_record[-1][:3]
        T_end = ur5e_car.get_end_effector_posture(car_state + arm_state)
        jacobi = ur5e_car.get_jacobian_lx(car_state + arm_state)
        jacobi = np.array(jacobi)
        T_end = np.array(T_end)

    # set QP cost and constraints
        # get current tracking error(only position) in cartesian space

        eT = np.linalg.pinv(T_end) @ T_target
        eT = np.matmul(np.linalg.pinv(T_end) , T_target)
        et = np.sum(np.abs(eT[:3,-1]))
        print('et',et)
    # penerty term for speed control(UR)
        Y = 0.01

    # quadratic part (DOF + slack term for the EE velocity tracking error)
        Q = np.eye(2 + ur_robot.n + 6)
        Q[:2 + ur_robot.n, :2 + ur_robot.n] *= Y
        Q[0, 0] *= 3/et
        Q[1, 1] *= 0.1/et
        Q[-6:,-6:] = (1.0 /et) * np.eye(6)
    # linear part
        c = np.concatenate((np.zeros(2), -0.4*ur_robot.jacobm(arm_state).reshape((ur_robot.n,)), np.zeros(6)))
        # c = np.concatenate((np.zeros(2), -1/et*ur_robot.jacobm(arm_state).reshape((ur_robot.n,)), np.zeros(6)))

        kε = 0.3
        bTe = ur_robot.fkine(arm_state).A
        θε = math.atan2(-bTe[0, -1], bTe[1, -1]) + 0    #the x axie of UR is right direction, the y axie of UR is forward direction
        ε = kε * θε 
        c[1] = -ε

        # c[2] = ε

        print('c',c)
    # position servo in caresian space
        v = np.zeros(6)
        v[:3] = 0.5*(T_target[:3,-1]-T_end[:3,-1])          
        w = get_delta_angle(T_end[:3,:3],T_target[:3,:3])
        v[3:] = w
    # equality constraints
        Aeq = np.c_[jacobi, np.eye(6)]
        beq = v.reshape((6,))
        # Aeq = np.r_[Aeq, np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(1,-1),np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(1,-1)]
        # beq = np.r_[beq, -0.4, 0]
        # print('Aeq',Aeq)
        # print('beq',beq)
    # inequality constraints
        Ain = np.zeros((2 + ur_robot.n + 6, 2 + ur_robot.n + 6))
        bin = np.zeros(2 + ur_robot.n + 6)

        Ain[2,2:2+ur_robot.n],bin[2] = safe_area(ur_robot.jacob0(arm_state)[1],0.1,0.4,bTe[1, -1],1)     #x axie range limit
        Ain[3,2:2+ur_robot.n],bin[3] = safe_area(-ur_robot.jacob0(arm_state)[0],0.1,0.4,-bTe[0, -1],1)   #y axie range limit
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

        # qd = qp.solve_qp(Q, c, A=Aeq, b=beq ,solver='cvxopt')
 

    #control the robot
        if et > 0.5:
            qd *= 0.7 / et
        else:
            qd *= 1.4
        if et < 0.03:
            break
            break
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
            print('the linear vel is too large')
        if abs(vel.angular.z) > car_angular_vel:
            vel.angular.z = car_angular_vel*np.sign(vel.angular.z)
            print('the angular vel is too large')
        base_pub.publish(vel)  

        print('qd',qd)
        print('car',vel.linear.x,' ',vel.angular.z)
        print('manipula',ur_robot.manipulability(arm_state))
        print('θε',θε)

        # print('ur vel',ur_robot.jacob0(arm_state) @ qd[2:-6])
        print()
        time.sleep(dt)
    rtde_c.disconnect()
    

