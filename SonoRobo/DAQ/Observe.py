import os
#import re
#import shutil
#import threading
#import time
#import tkinter as tk
#from tkinter import W, E

#import matplotlib
import numpy as np
#import pandas
import platform
import os

#import Device
import Logger

#import Library
import Settings
import Ports
import Sonar

import matplotlib.pyplot as plt


class Observe:
    def __init__(self):
        # Prepare Logger
        self.logger = Logger.Logger('DAQgui')

        # list ports
        self.os = platform.system()
        wd = os.getcwd()

        self.logger.print_log('Runing on ' + self.os)
        self.logger.print_log('Working directory: ' + wd)
        self.logger.print_log('Connected ports')
        p = Ports.Ports()
        p.print()

        # Read settings
        self.connect_sonar = Settings.connect_sonar

        if self.connect_sonar:
            self.logger.print_log('Connecting to Sonar')
            self.sonar = Sonar.Sonar()
            if self.os == 'Linux': self.sonar.connect()
            if self.os == 'Windows': self.sonar.connect(Settings.sonar_port)

            start_freq = Settings.start_freq
            end_freq = Settings.end_freq
            samples = Settings.samples
            self.sonar.set_signal(start_freq, end_freq, samples)
            self.sonar.build_charge()

        # Set initial values
        #self.counter_value.set(0)
        #self.repeat_value.set(Settings.default_repeats)
        #self.status_value.set('Ready')
        self.logger.print_log('Ready')

        self.echo = {'left': np.zeros(7000), 'right': np.zeros(7000)}

    def one_echo(self):
        if self.connect_sonar:
            data = self.sonar.measure()
            data = Sonar.convert_data(data, 7000)
            print(data.shape)
            self.echo['left'] = data[:,0]
            self.echo['right'] = data[:,1]
        
        return self.echo

if __name__ == "__main__":
    run = True
    observe = Observe()
    while run:
        text = input('Nothing to observe, something to stop: ')
        if text == '':
            observe.one_echo()
            print('LEFT = ')
            print(observe.echo['left'])
            print('RIGHT = ')
            print(observe.echo['right'])

            plt.plot(observe.echo['left'], alpha = 0.6, label = 'LEFT')
            plt.plot(observe.echo['right'], alpha = 0.6, label = 'RIGHT')
            plt.legend()
            plt.show()

        elif text == 'cf': #close fig
            plt.close()
        else:
            break


    