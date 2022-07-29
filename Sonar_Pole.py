# Gather sonar data from symmetrical pole

from gc import garbage
import numpy as np
import os
from DAQ import Ports
from DAQ.Observe import Observe
import time
import sys

from pycreate2 import Create2
from ViveServer.Client import Client
from RobotGPS import GPS
from Motion import Motion

IP_ADDRESS = '192.168.8.166' #IP address of elisive-calf on Alfred CAPCOM

def wrap_to_180(a):
    if a > 180:
        a -= 360
    elif a< -180:
        a += 360
    return a

if __name__ == '__main__':
    port = Ports.get_port('SER=DN0267BH')
    robot = Create2(port=port)
    observe = Observe()
    gps_robot = GPS.GPS()
    move_robot = Motion.RobotTranslator(robot)
    
    # Initialize connection with ViveServer
    c = Client(IP_ADDRESS)
    print('Client initiated')
    c.connect()
    print('Client Connected')

    # Start the robot in safe mode
    robot.start()
    robot.safe()    

    # Pole test
    start_dist = 0.5    # meters
    final_dist = 3.0    # meters
    dist_inc = 0.25     # meters
    ang_inc = 2         # degrees

    dist = np.arange(start_dist, final_dist+dist_inc, dist_inc)
    for x in dist:
        temp_l = np.zeros((181,7000,5))
        temp_r = np.zeros((181,7000,5))
        
        for a in range(0, 181, ang_inc):
            # Modifying coordinates for run
            y = 0
            temp_a = a  # before modifying angle, make sure we store temp data in right index of temp_l/temp_r
            a += 90
            a = wrap_to_180(a)

            # travel code
            gps_robot.get_robot_pose(c)
            if abs(wrap_to_180(gps_robot.theta - a)) >= 3:
                gps_robot.travel(robot, c, x, y, a)
            else:
                move_robot.turn(a=ang_inc, bot=robot, smooth_stop=False, w_error=3.416)

            # echo code
            left5 = np.zeros((7000,5))
            right5 = np.zeros((7000,5))

            # grab 5 measurements of echo data in the current pose
            for k in range(5):
                observe.one_echo()
                left5[:,k] = observe.echo['left']
                right5[:,k] = observe.echo['right']

            temp_l[int(temp_a)] = left5
            temp_r[int(temp_a)] = right5

        # save trial_dist
        path = '/home/mendel/SonoBono/echo_data/'
        file_name = str(x) + ".npz"
        np.savez( os.path.join(path, file_name), left_data = temp_l, right_data = temp_r)
        print("Saved ", file_name, "to path ", path)
        garbage = [temp_l, temp_r, left5, right5]
        del temp_l, temp_r, left5, right5
        del garbage

    robot.stop()
