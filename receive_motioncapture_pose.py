# roslaunch vrpn_client_ros sample.launch server:=10.1.1.198
# subscribe  /vrpn_client_node/Tracker7/pose  
# Type:geometry_msgs/PoseStamped

import rospy
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
import threading
import tty,termios
import sys
import time
import numpy as np
KEY = '1'  #emergency stop flag
class _Getch:       
    def __call__(self):
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch

def press_enter_to_JiTing():#不是完全的急停
    global KEY
    while True:

        inkey = _Getch()
        KEY=inkey()
        termios.tcflush(sys.stdin,termios.TCIFLUSH)
        if ord(KEY) == 26 or ord(KEY) == 27 or ord(KEY) == 3 or KEY == 'Q' or KEY == 'q':
            break   
        time.sleep(0.3)
  
listener=threading.Thread(target=press_enter_to_JiTing)
listener.start()


receiveBuffer = []
def receiveCb(pose:PoseStamped):
    rotation = R.from_quat((pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w)).as_matrix().ravel()
    translation = np.array((pose.pose.position.x, pose.pose.position.y, pose.pose.position.z)) / 1000.0
    full = np.hstack((rotation,translation))
    receiveBuffer.append(full)
    


writeBuffer = []
if __name__ == "__main__":
    rospy.init_node("MotionCaptureReceiveNode")
    ur_sub = rospy.Subscriber("/vrpn_client_node/Tracker7/pose",PoseStamped,receiveCb)
    
    while ur_sub.get_num_connections() == 0:
        time.sleep(0.1)
    print('subscribe successfully!' + str(ur_sub.get_num_connections()))

    while not (ord(KEY) == 26 or ord(KEY) == 27 or ord(KEY) == 3  or KEY == 'Q' or KEY == 'q'):

        if KEY == '1':
            continue
        if KEY == 'J' or KEY == 'j':
            writeBuffer.append(receiveBuffer[-1])
            print(receiveBuffer[-1])
            print()            
        KEY = '1'
    np.savetxt('motionCapturePose.txt',np.array(writeBuffer))

