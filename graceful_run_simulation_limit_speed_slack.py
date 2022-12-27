#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append('.')
import numpy as np
import math
import time
import seaborn as sns
import random
import rospy
import qpsolvers as qp
import spatialmath as sm

from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from math import cos, acos, sin, asin, sqrt, exp, atan, atan2, pi, tan, ceil
from jacobi.ur5e_robot_Jacob_tool_length0145 import *
from jacobi.ur5e_car_kinematics_class_2 import UR5e_car_kinematics
import matplotlib.pyplot as plt
from trajectory_msgs.msg import *
from control_msgs.msg import *
from geometry_msgs.msg import Twist, PoseStamped
from gazebo_msgs.msg import LinkStates
from nav_msgs.msg import Odometry
import roboticstoolbox as rtb
from scipy.spatial.transform import Rotation
ur5e_car = UR5e_car_kinematics()
# ur5e_car.get_jacobian_lx()
# ur5e_car.get_end_effector_posture()
ur_sub_record = []
ur_end_record = []
gazebo_sub_record = []

car_sub_record = []
car_coor_record=[]
ur_coor_record=[]
wb_record = []
#UR state subsciber callback function
#FK to end pose 
Stop_flag = False
New_Goal_Flag = False
Goal = [0,0,0]  #x y orentation
def ur_sub_callback(state):
    ur_sub_record.append(list(state.actual.positions))

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
# def gazebo_sub_callback(state):
#     R = Rotation.from_quat([state.pose[-2].orientation.x, state.pose[-2].orientation.y, state.pose[-2].orientation.z, state.pose[-2].orientation.w]).as_matrix()
#     T = np.zeros((4,4))
#     T[0:3,0:3] = R
#     T[0,-1] = state.pose[-2].position.x
#     T[1,-1] = state.pose[-2].position.y
#     T[2,-1] = state.pose[-2].position.z
#     T[-1,-1] = 1
#     gazebo_sub_record.append(T)
def goal_sub_callback(msgs):
    global New_Goal_Flag
    New_Goal_Flag = True
    Goal[0] = msgs.pose.position.x
    Goal[1] = msgs.pose.position.y

    x = msgs.pose.orientation.x
    y = msgs.pose.orientation.y
    z = msgs.pose.orientation.z
    w = msgs.pose.orientation.w
    theta = math.atan2(2*(w*z+x*y),1-2*(y**2+z**2))
    Goal[2] = theta
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
        return np.array([0,0,0])
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
# if __name__ == "__main__":

#     print(safe_area(np.ones(6),0.2,0.5,-0.3,1))
#     sys.exit() 

if __name__ == "__main__":

    #init ros publisher 
    rospy.init_node("Peter_Control")
    ur_pub = rospy.Publisher ("/arm_controller/command", JointTrajectory,queue_size=0)  
    base_pub = rospy.Publisher ("/mobile_base_controller/cmd_vel", Twist,queue_size=0) 
    ur_sub = rospy.Subscriber("/arm_controller/state",JointTrajectoryControllerState,ur_sub_callback)
    base_sub = rospy.Subscriber("/mobile_base_controller/odom",Odometry,car_sub_callback)
    # gazebo_sub = rospy.Subscriber("/gazebo/link_states",LinkStates,gazebo_sub_callback)
    goal_sub = rospy.Subscriber("/move_base_simple/goal",PoseStamped,goal_sub_callback)

    ur_robot = rtb.models.UR5()


    time.sleep(0.5)     #need to wait for a short time. If not, the node cannot work
    #prepare for publish
    msg = JointTrajectory()
    vel = Twist()
    np.set_printoptions(threshold=np.inf)   #for convenient showing
    msg.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint','wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    point = JointTrajectoryPoint()

    #set the car and UR initial state
    vel.linear.x = 0
    base_pub.publish(vel)
    joint_init=np.array([-1.51,-0.38,-2.14,-2.14,1.63])
    # joint_init=np.array([-38.77 , -41.84, -104.27, -123.93, 90.33,-134])/57.3
    # joint_init = np.array([-1.34,-1.13,-1.38,-1.58,1.58,-2.21])
    joint_init0 = np.array([-4.78,-2.14,2.01,-1.51,4.75,-3.18])
    joint_init = np.array([-4.78,-2.14,2.01,-2.14,4.78,-3.18])

    # joint_init = np.array([-1.01,-0.99,-1.67,-1.51,1.13,-3.18])
    point.positions = joint_init0
    point.time_from_start = rospy.Duration(1,0)
    msg.points = [point]
    msg.header.seq = 0
    ur_pub.publish(msg)
    time.sleep(1.5)
    ur_sub_record.clear()
    car_sub_record.clear()
    gazebo_sub_record.clear()
    time.sleep(0.5)
    
    # given a target pose T_target 4*4

    T_target  = ur5e_car.get_end_effector_posture(car_sub_record[-1][:3] + ur_sub_record[-1])
    T_target = np.array(T_target)  

    T_target0 = np.array(ur5e_car.get_end_effector_posture(car_sub_record[-1][:3] + ur_sub_record[-1]))
    switch_flag = False



    point.positions = joint_init
    point.time_from_start = rospy.Duration(1,0)
    msg.points = [point]
    msg.header.seq = 0
    ur_pub.publish(msg)
    time.sleep(1.5)
    arm_state = ur_sub_record[-1]
    bTe = ur_robot.fkine(arm_state).A
    θε0 = math.atan2(-bTe[0, -1], bTe[1, -1]) + 0
    print('θε0',θε0)
    # LOOP
    # get current pose T_current
    cmax = 0
    while New_Goal_Flag == False:
        time.sleep(0.1)
    New_Goal_Flag = False

    T_target  = T_target0.copy()

    T_target[0, -1] = Goal[0]
    T_target[1, -1] = Goal[1]
    theta = atan2(Goal[1] - 0,Goal[0] -0)
    T_target[:3,:3] = T_target[:3,:3]#* sm.SE3.Rz(theta)
    print('theta',theta)
    time.sleep(0.5)
    T_target = T_target0.copy()
    T_target[0,-1] += 1.5
    # T_target[2,-1] -= 0.1
    # T_target[1,-1] = 0
    T_target[:3,:3] = T_target[:3,:3]#* sm.SE3.Rz(np.pi)
    dt = 0.1

    arm_max_acc = 11
    arm_max_vel = 11
    car_linear_acc = 0.4
    car_angular_acc = 0.4
    car_linear_vel = 1
    car_angular_vel = 1

    arm_max_acc = 2
    arm_max_vel = 1
    car_pre_linear_vel = 0
    car_pre_angular_vel = 0


    last_arm_vel = np.zeros(6)

    first_flag = True   

    wb_record = [] 
    while True:
        if New_Goal_Flag:
            New_Goal_Flag = False
            first_flag = True   
            wb_record = [] 
            point.positions = joint_init
            point.time_from_start = rospy.Duration(1,0)
            msg.points = [point]
            msg.header.seq = 0
            ur_pub.publish(msg)
            time.sleep(1.5)
            car_pre_linear_vel = 0
            car_pre_angular_vel = 0
            last_arm_vel = np.zeros(6)
            if switch_flag==True:
                switch_flag = False
                T_target = T_target0.copy()
                T_target[0,-1] = -3
                # T_target[1,-1] = 0
                T_target[:3,:3] = T_target[:3,:3]* sm.SE3.Rz(np.pi)
            else:
                switch_flag = True
                T_target = T_target0.copy()
                T_target[0,-1] = 0.5
                # T_target[1,-1] = 0
            # T_target  = T_target0.copy()
            # T_target[0, -1] = Goal[0]
            # T_target[1, -1] = Goal[1]
            # theta = atan2(Goal[1] - 0,Goal[0] - 0)
            # T_target[:3,:3] = T_target[:3,:3]* sm.SE3.Rz(theta)
            # print('theta',theta)
            time.sleep(0.5)

                
        arm_state = ur_sub_record[-1]
        car_state = car_sub_record[-1][:3]
        T_end = ur5e_car.get_end_effector_posture(car_state + arm_state)
        jacobi = ur5e_car.get_jacobian_lx(car_state + arm_state)
        jacobi = np.array(jacobi)
        T_end = np.array(T_end)
        wb_record.append([T_end[0,-1],T_end[1,-1],T_end[2,-1]])
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
        Q[0, 0] *= 10/(et)
        Q[1, 1] *= 2/(et)
        
        Q[-7, -7] *= 99999 #dynamic weight

        Q[-6:,-6:] = (1 /et) * np.eye(6)
    # linear part
        c = np.concatenate((np.zeros(2), -1*ur_robot.jacobm(arm_state).reshape((ur_robot.n,)), np.zeros(6)))
        # c = np.concatenate((np.zeros(2), -1/et*ur_robot.jacobm(arm_state).reshape((ur_robot.n,)), np.zeros(6)))

        kε = 0.2
        bTe = ur_robot.fkine(arm_state).A
        θε = math.atan2(-bTe[0, -1], bTe[1, -1]) + 0    #the x axie of UR is right direction, the y axie of UR is forward direction
        ε = kε * (θε - θε0) 
        c[1] = -ε

        # print('c',c)
    # position servo in caresian space
        v = np.zeros(6)
        v[:3] = 1*(T_target[:3,-1]-T_end[:3,-1])      
        if np.linalg.norm(v[:3]) < 0.05:
            v[:3] *= 0.05/np.linalg.norm(v[:3])
            print('norm',np.linalg.norm(v[:3]))

        w = get_delta_angle(T_end[:3,:3],T_target[:3,:3])
        v[3:] = w*2.5
        if et < 1:
            v[3:] = np.max([0.7,et/0.5])*w
        # print('v',v)
    # equality constraints
        Aeq = np.c_[jacobi, np.eye(6)]
        beq = v.reshape((6,))
        # Aeq = np.r_[Aeq, np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(1,-1),np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(1,-1)]
        # beq = np.r_[beq, -0.4, 0]
        # beq = np.r_[beq, -0.4, 0]
        # print('Aeq',Aeq)
        # print('beq',beq)
    # inequality constraints
        Ain = np.zeros((2 + ur_robot.n + 6, 2 + ur_robot.n + 6))
        bin = np.zeros(2 + ur_robot.n + 6)

        Ain[2,2:2+ur_robot.n],bin[2] = safe_area(ur_robot.jacob0(arm_state)[1],0.1,0.6,bTe[1, -1],1)     #x axie range limit
        Ain[3,2:2+ur_robot.n],bin[3] = safe_area(-ur_robot.jacob0(arm_state)[0],0.1,0.6,-bTe[0, -1],1)   #y axie range limit
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
        if et > 1:
            qd *= 0.4 / et
        else:
            qd *= 0.8
        # qd*=2
        for i in range(6):
            if abs(qd[2+i] - last_arm_vel[i]) > arm_max_acc:    #acc
                qd[2+i] = last_arm_vel[i] + arm_max_acc*dt*np.sign(qd[2+i]-last_arm_vel[i])
            if abs(qd[2+i]) > arm_max_vel:
                qd[2+i] = arm_max_vel*np.sign(qd[2+i])
        last_arm_vel = qd[2:8]
        point.positions = arm_state +  qd[2:-6] * 0.1
        point.time_from_start = rospy.Duration(0.1,0)
        msg.points = [point]
        msg.header.seq = 0
        ur_pub.publish(msg)

        vel.linear.x = qd[0]
        vel.angular.z = qd[1]
        #linear acc limits. The angular acc limits is considered in time optimal 1d tracking
        if abs(car_pre_linear_vel-vel.linear.x) > car_linear_acc*dt:
            vel.linear.x = car_pre_linear_vel + np.sign(vel.linear.x-car_pre_linear_vel)*car_linear_acc*dt
        if abs(car_pre_angular_vel-vel.angular.z) > car_angular_acc*dt:
            vel.angular.z = car_pre_angular_vel + np.sign(vel.angular.z-car_pre_angular_vel)*car_angular_acc*dt

        if abs(vel.linear.x) > car_linear_vel:
            vel.linear.x = car_linear_vel*np.sign(vel.linear.x)
        if abs(vel.angular.z) > car_angular_vel:
            vel.angular.z = car_angular_vel*np.sign(vel.angular.z)
        car_pre_linear_vel = vel.linear.x
        car_pre_angular_vel = vel.angular.z
        base_pub.publish(vel)
        # print('car lin',qd[0],'ang',qd[1])
        # print('ratio',qd[1]/qd[2])
        # print('manipula',ur_robot.manipulability(arm_state))
        print()

        print('θε',θε)
        if et < 0.03 and first_flag == True:
            first_flag = False
            plt.figure()
            plt.plot(np.array(car_sub_record)[:,-1])
            plt.plot(np.array(car_sub_record)[:,-3])
            plt.legend(('angular','linear'))
            plt.figure()
            plt.plot(np.array(wb_record)[:,0])
            plt.plot(np.array(wb_record)[:,1])
            plt.plot(np.array(wb_record)[:,2])
            plt.legend(('x','y','z'))
            plt.show()
            car_sub_record.clear()
            wb_record.clear()
            time.sleep(0.5)
        time.sleep(0.1)

    

