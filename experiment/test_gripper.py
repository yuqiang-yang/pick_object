import robotiq_gripper as robotiq_gripper
from robotiq_gripper import *
ip = '192.168.100.2'
print("Creating gripper...")
gripper = RobotiqGripper()
print("Connecting to gripper...")
gripper.connect(ip, 63352)
print("Activating gripper...")
gripper.activate()
print("Testing gripper...")
flag = False
while True:
    if flag:
        flag = False
        print(gripper.move_and_wait_for_pos(0, 255, 120)[1] == RobotiqGripper.ObjectStatus.STOPPED_INNER_OBJECT)
    else:
        flag = True
        print(gripper.move_and_wait_for_pos(255, 255, 120)[1] == RobotiqGripper.ObjectStatus.STOPPED_INNER_OBJECT)
