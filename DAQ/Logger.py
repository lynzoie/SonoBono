import logging
import os
import os.path
import sys


def lst2str(lst):
    if isinstance(lst, str): return lst
    text = ''
    for x in lst: text += str(x) + ' '
    text = text.rstrip(' ')
    return text


class Logger:
    def __init__(self, name, file_name=None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        # create file handler which logs even debug messages
        log_folder = os.path.join(os.getcwd(), 'logs')
        if file_name is None: file_name = name
        if not os.path.exists(log_folder): os.mkdir(log_folder)
        log_file = os.path.join(log_folder, file_name + '.log')
        file_logger = logging.FileHandler(log_file, mode='w')
        file_logger.setLevel(logging.INFO)
        # create console handler with a higher log level
        console_logger = logging.StreamHandler(sys.stdout)
        console_logger.setLevel(logging.INFO)
        # create formatter and add it to the handlers
        #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        name = name.ljust(20)
        formatter = logging.Formatter(name + ' - %(asctime)s - %(levelname)s - %(message)s')
        file_logger.setFormatter(formatter)
        console_logger.setFormatter(formatter)
        # add the handlers to the logger
        self.logger.addHandler(file_logger)
        self.logger.addHandler(console_logger)

    def print_log(self, text, level='i'):
        text = lst2str(text)
        if level == 'i': self.logger.info(text)
        if level == 'w': self.logger.warning(text)
        if level == 'c': self.logger.critical(text)

    def info(self, text):
        self.logger.info(text)

    def warning(self, text):
        self.logger.warning(text)

    def critical(self, text):
        self.logger.critical(text)
