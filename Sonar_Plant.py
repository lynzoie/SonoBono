# Gather sonar data from symmetrical pole
import numpy as np
import os
from DAQ import Ports
from DAQ.Observe import Observe
import time

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

    # Initialize locating robot and moving robot objects
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

    # Plant test
    ang_rot = 45        # degree increment to circle around plant 
    start_dist = 0.5    # meters
    final_dist = 3.0    # meters
    dist_inc = 0.25     # meters
    ang_inc = 2         # degree increment to sweep at xy coordinate

    ang = np.arange(0, 360+ang_rot,ang_rot)
    for x in range(len(ang)): ang[x] = wrap_to_180(ang[x])
    rad = np.arange(start_dist, final_dist+dist_inc, dist_inc)

    for angle in ang: # current angle in sweep around plant
        for r in rad:
            # Record 180 degree sweep of one xy coordinate 
            temp_l = np.zeros((181,7000,5))
            temp_r = np.zeros((181,7000,5))

            # Calculate xy coordinate based on angle sweep and radius from plant
            x = r*np.cos(np.radians(angle)) 
            y = r*np.sin(np.radians(angle))
            
            start_ang = wrap_to_180(angle+90)
            final_ang = wrap_to_180(start_ang + 180)
            val_start = start_ang if start_ang < final_ang else final_ang
            val_end = start_ang if start_ang > final_ang else final_ang
            
            for cur_ang in range(val_start, val_end+ang_inc, ang_inc):
                if start_ang > final_ang:
                    cur_ang = wrap_to_180(cur_ang + 180)
                
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

                temp_l[int(cur_ang)] = left5
                temp_r[int(cur_ang)] = right5

            # save trial_dist
            path = '/home/mendel/SonoBono/echo_data/'
            file_name = str(r) + "_" + str(angle) + "_" + str(x) + "_" + str(y) + ".npz"
            np.savez( os.path.join(path, file_name), left_data = temp_l, right_data = temp_r)
            print("Saved ", file_name, "to path ", path)
            garbage = [temp_l, temp_r, left5, right5]
            del temp_l, temp_r, left5, right5
            del garbage

    robot.drive_stop()
