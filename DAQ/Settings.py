import numpy

connect_sonar = True
connect_lidar = False
connect_servo = False
sweep_steps = 1
# --> this is the number of evenly divided steps between -45 and +45 degrees
# for examples, with sweep_steps = 5, servo will sweep at -45, -22.5, 0, +22.5, +45

default_repeats = 1

servo_pos45deg = 2105
servo_neg45deg = 1256

#ports on windows
# Only used when running on linux
servo_port = 'COM4'
sonar_port = 'COM5'

#servo_range = [400, 2000]
if connect_servo:
    servo_positions = numpy.linspace(servo_neg45deg, servo_pos45deg, sweep_steps)
else:
    servo_zero = (servo_pos45deg + servo_neg45deg)/2

start_freq = 40000 +1
end_freq = 40000 - 1
samples = 100
measurement_pause = 0.1
servo_pause = 1
