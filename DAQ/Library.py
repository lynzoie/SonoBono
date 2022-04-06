import os
import shutil
import re

import numpy
from natsort import natsorted


def make_folder(folder, empty=False):
    if os.path.exists(folder) and empty: shutil.rmtree(folder)
    if not os.path.exists(folder): os.makedirs(folder)


def get_files(folder):
    contents = os.listdir(folder)
    files = []
    for entry in contents:
        full_path = os.path.join(folder, entry)
        if os.path.isfile(full_path): files.append(entry)
    return files


def get_type_files(folder, type):
    filtered = []
    files = get_files(folder)
    for x in files:
        if x.endswith(type): filtered.append(x)
    return filtered


def extract_numbers(file_list):
    numbers = []
    pattern = '\d+'
    for file_name in file_list:
        matches = re.findall(pattern, file_name)
        match = matches[-1]
        match = int(match)
        numbers.append(match)
    if numbers == []: numbers = [-1]
    numbers.sort()
    return numbers


def process_scans(scans):
    scans = str(scans)
    result = re.findall('\d+', scans)
    result = [int(x) for x in result]
    converted = []
    while len(result)>0:
        angle = result.pop(0)
        distance = result.pop(0)
        strength = result.pop(0)
        converted.append([angle, distance, strength])
    return converted


def read_measurements(input_folder, output_file=None):
    files = get_type_files(input_folder, 'npy')
    files = natsorted(files)
    first_file = os.path.join(input_folder, files[0])
    data = numpy.load(first_file)
    dimensions = data.shape
    all_data = numpy.zeros((*dimensions, len(files)))
    for f in files:
        file_name = os.path.join(input_folder, f)
        data = numpy.load(file_name)
        index = files.index(f)
        all_data[:, :, :, :, index] = data
    if output_file is None: return all_data
    numpy.savez(output_file, all_data)