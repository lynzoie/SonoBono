#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:17:23 2017

@author: dieter
"""
import os
import re
import shutil
import threading
import time
import tkinter as tk
from tkinter import W, E

import matplotlib
import numpy
import pandas
import platform
import os

import Device
import Logger

import Library
import Settings
import Ports
import Sonar


matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # NavigationToolbar2TkAgg
from matplotlib.figure import Figure


def process_scans(scans):
    scans = str(scans)
    result = re.findall('\d+', scans)
    result = [int(x) for x in result]
    converted = []
    while len(result) > 0:
        angle = result.pop(0)
        distance = result.pop(0)
        strength = result.pop(0)
        converted.append([angle, distance, strength])
    return converted


class HelloApp:
    def __init__(self, master):        
        # Prepare Logger
        self.logger = Logger.Logger('DAQgui')
        
        #list ports
        self.os = platform.system()
        wd = os.getcwd()
        
        self.logger.print_log('Runing on ' + self.os)
        self.logger.print_log('Working directory: ' + wd)
        self.logger.print_log('Connected ports')
        p = Ports.Ports()
        p.print()

        # Read settings
        self.connect_lidar = Settings.connect_lidar
        self.connect_sonar = Settings.connect_sonar
        self.connect_servo = Settings.connect_servo

        self.folder_name = tk.StringVar()
        self.counter_value = tk.IntVar()
        self.repeat_value = tk.IntVar()
        self.status_value = tk.StringVar()

        # Figure
        self.fig = Figure(figsize=(10, 5), dpi=100)
        self.axis1 = self.fig.add_subplot(121)
        self.axis2 = self.fig.add_subplot(122)

        # Define widgets
        self.master = master
        self.folder = tk.Entry(textvariable=self.folder_name, width=30)
        self.repeats = tk.Entry(textvariable=self.repeat_value, width=10, justify=tk.CENTER)
        self.counter = tk.Label(textvariable=self.counter_value, width=10, justify=tk.CENTER)
        self.status = tk.Message(textvariable=self.status_value, width=500)
        self.measure = tk.Button(master, text="Measure")

        # Figure widget
        self.canvas = FigureCanvasTkAgg(self.fig, self.master)
        self.canvas_widget = self.canvas.get_tk_widget()

        # Add the widgets to their parents
        self.folder.grid(row=0, column=0, columnspan=2, sticky=W + E)
        self.counter.grid(row=0, column=2, sticky=W + E)
        self.repeats.grid(row=1, column=0, sticky=W + E)
        self.measure.grid(row=2, column=0, sticky=W + E)
        self.status.grid(row=4, column=0, columnspan=3, sticky=W + E)
        self.canvas_widget.grid(row=3, column=0, columnspan=3)
        # Scan variables
        self.current_scan = None
        self.current_scan_time = None

        # Prepare Sonar
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

        if self.connect_lidar:
            self.logger.print_log('Connecting to Lidar')
            self.scan_thread = threading.Thread(target=self.scanning)
            self.scan_thread.start()
        
        self.servo_positions = [0]
        if self.connect_servo:
            self.logger.print_log('Connecting to servo')
            if self.os == 'Linux': self.servo_board = Device.BoardDevice()
            if self.os == 'Windows': self.servo_board = Device.BoardDevice(Settings.servo_port)
            self.servo_positions = Settings.servo_positions

        # Bindings
        self.measure.bind('<ButtonPress>', self.do_measurement)
        # master.protocol("WM_DELETE_WINDOW", self.on_close)

        # Set initial values
        self.counter_value.set(0)
        self.repeat_value.set(Settings.default_repeats)
        self.status_value.set('Ready')
        self.logger.print_log('Ready')
        
    def scanning(self):
        from sweeppy import Sweep
        port = Ports.get_port('FT230X Basic UART')
        with Sweep(port) as sweep:
            sweep.start_scanning()
            for scan in sweep.get_scans():
                data = ('{}\n'.format(scan))
                self.current_scan = process_scans(data)
                self.current_scan_time = time.asctime()

    def get_scans(self, n=3):
        scans = self.current_scan
        stamp = self.current_scan_time
        message = 'Got scan %i/%i ' % (1, n)
        self.status_value.set(message)
        self.status.update_idletasks()
        for x in range(n):
            while self.current_scan_time == stamp: time.sleep(0.1)
            scans = scans + self.current_scan
            stamp = self.current_scan_time
            message = 'Got scan %i/%i ' % (x + 1, n)
            self.status_value.set(message)
            self.status.update_idletasks()

        all_samples = pandas.DataFrame(scans)
        all_samples.columns = ['degrees', 'distance', 'strength']
        all_samples['degrees'] = all_samples['degrees'] / 1000  # milli-degress to degrees
        all_samples['rad'] = numpy.deg2rad(all_samples['degrees'])
        all_samples['distance'] = all_samples['distance'] / 10  # cm to mm
        all_samples['x'] = all_samples['distance'] * numpy.cos(all_samples['rad'])
        all_samples['y'] = all_samples['distance'] * numpy.sin(all_samples['rad'])
        return all_samples

    def do_measurement(self, event):
        folder = self.folder_name.get()
        if folder == '':
            self.logger.print_log('Provide a measurement name.')
            return
        data_folder = os.path.join('data', folder)
        Library.make_folder(data_folder)
        shutil.copy('Settings.py', data_folder + '/Settings.py')
        files = Library.get_files(data_folder)
        files.remove('Settings.py')
        numbers = Library.extract_numbers(files)
        current_counter = max(numbers) + 1
        current_counter_str = str(current_counter).rjust(4, '0')
        self.counter_value.set(current_counter)
        repeats = self.repeat_value.get()

        #
        # Get acoustic data
        #

        distance_axis = (0.5 * 340 * numpy.arange(0, 7000) / 300000)
        n_positions = len(self.servo_positions)
        all_data = numpy.empty((7000, 2, repeats, n_positions))
        for position_i in range(n_positions):
            position = self.servo_positions[position_i]
            if self.connect_servo:
                self.servo_board.device.set_target(0, position)
                time.sleep(Settings.servo_pause)

            for repetition in range(repeats):
                data = numpy.random.rand(7000, 2)
                message = 'Performing measurement %s, %i/%i @ position %i ' % (current_counter_str, repetition + 1, repeats, position)
                self.logger.print_log(message)
                self.status_value.set(message)
                self.status.update_idletasks()

                if self.connect_sonar:
                    data = self.sonar.measure()
                    data = Sonar.convert_data(data, 7000)
                all_data[:, :, repetition, position_i] = data
                time.sleep(Settings.measurement_pause)

            current_measurement_data = all_data[:, :, :, position_i]
            mean_data = numpy.mean(current_measurement_data, axis=(2))
            #print(current_measurement_data.shape, mean_data.shape)
            self.axis1.clear()
            self.axis1.plot(distance_axis, mean_data, alpha=0.5)
            self.axis1.set_title('Acoustic data')
            self.canvas.draw()
            time.sleep(Settings.servo_pause)

        output_file = os.path.join(data_folder, 'measurement' + current_counter_str + '.npy')
        numpy.save(output_file, all_data)

        #
        # Get LIDAR DATA
        #
        if self.connect_lidar:
            message = 'Performing LIDAR measurement'
            self.logger.print_log(message)
            self.status_value.set(message)
            self.status.update_idletasks()

            lidar_data = self.get_scans()
            plot_data = lidar_data[lidar_data['strength'] > 50]
            mns = plot_data.groupby('degrees')
            mns = mns.mean()
            mns = mns.reset_index()

            self.axis2.clear()
            self.axis2.scatter(mns['x'], mns['y'], s=0.1)
            self.axis2.axis('equal')
            self.canvas.draw()
            output_file = os.path.join(data_folder, 'measurement' + current_counter_str + '.csv')
            lidar_data.to_csv(output_file)

        message = 'Measurement %s completed' % (current_counter_str)
        self.logger.print_log(message)
        self.status_value.set(message)
        self.status.update_idletasks()


if __name__ == "__main__":
    root = tk.Tk()
    app = HelloApp(root)
    root.mainloop()
