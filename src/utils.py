import numpy as np, h5py, sys, os, errno, concurrent.futures, argparse, matplotlib.pyplot as plt
from math import sqrt
from functools import partial
from itertools import chain

parent_path = "/asap3/petra3/gpfs/p07/2019/data/11005196"
output_path_data = "../../hdf5/Scan_{0:d}/scan_{0:d}_data.h5"
output_path_scan = "../../hdf5/Scan_{0:d}/scan_{0:d}.h5"
scan_path = "raw/scanFrames/Scan_{0:d}"
log_path = "raw/Scans/Scan_{0:d}.log"

header = '#'
sizeline = '# Points count:'

calib_file = h5py.File(os.path.join(os.path.dirname(__file__), 'lambda_far_up_down_calibration.h5'), 'r')

def make_output_dir(path):
    try:
        os.makedirs(os.path.dirname(path))
    except OSError as e:
        if e.errno != errno.EEXIST: raise

def get_filenames(scan_num, raw_filename):
    path = os.path.join(parent_path, scan_path.format(scan_num))
    filenames = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(raw_filename)]
    filenames.sort()
    return filenames

def get_log(scan_num, verbose):
    if verbose: print('Reading motor coordinates')
    logpath = os.path.join(parent_path, log_path.format(scan_num))
    if verbose: print("Log path: {}".format(logpath))
    lines, sizes = [], []
    for line in open(logpath, 'r'):
        if line.startswith(header):
            if line.startswith(sizeline):
                sizes.append(int(line.strip(sizeline)))
        else:
            lines.append(line)
    return lines, sizes[2], sizes[1]

def get_coords_step(scan_num, verbose):
    lines, fast_size, slow_size = get_log(scan_num, verbose)
    del lines[1::(2 * fast_size + 1)]
    slow_crds, fast_crds = [], []
    for line in lines[1::2]:
        parts = line.split(';')
        slow_crds.append(float(parts[-2].strip('um')))
        fast_crds.append(float(parts[-1].strip('um\n')))
    if verbose: print("Number of coordinates: {:d}".format(len(fast_crds)))
    return np.array(fast_crds), np.array(slow_crds), fast_size, slow_size

def get_coords_fly(scan_num, verbose):
    lines, fast_size, slow_size = get_log(scan_num, verbose)
    slow_crds, fast_crds = [], []
    for line in lines[1::2]:
        parts = line.split(';')
        slow_crds.append(float(parts[-2].strip('um')))
        fast_crds.extend([float(crd) for crd in parts[-1].split(',')][5:-5:2])
    if verbose: print("Number of coordinates: {:d}".format(len(fast_crds)))
    return np.array(fast_crds), np.repeat(np.array(slow_crds), fast_size), fast_size, slow_size - 1

def get_data(filenames, worker, verbose):
    if verbose: 
        print("Scan folder: {}".format(os.path.dirname(filenames[0])))
        print("Number of files: {}".format(len(filenames)))
    data = worker(filenames[0])
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     data = np.concatenate([chunk for chunk in executor.map(worker, filenames)], axis=0)
    if verbose: print("Raw data shape: {}".format(data.shape))
    return data

def pad_stxm(stxm, fast_size, slow_size):
    return np.concatenate(stxm, np.zeros(fast_size * slow_size - stxm.size)).reshape((fast_size, slow_size))

# def create_file(output_path, scan_num, verbose):
#     out_path = os.path.join(os.path.dirname(__file__), output_path.format(scan_num))
#     if verbose: print('Output path: %s' % out_path)
#     make_output_dir(out_path)
#     return h5py.File(out_path, 'w', libver='latest')

# def write_extra_data(out_file, fast_crds, slow_crds, fast_size, slow_size, verbose):
#     if verbose: print("Writing supplementary data")
#     coord_group = out_file.create_group('motor_coordinates')
#     coord_group.create_dataset('fast_coordinates', data=fast_crds)
#     coord_group.create_dataset('slow_coordinates', data=slow_crds)
#     size_group = out_file.create_group('scan_size')
#     size_group.create_dataset('fast_size', data=fast_size)
#     size_group.create_dataset('slow_size', data=slow_size)

# def write_data(scan_num, scan_mode, verbose):
#     out_file = create_file(output_path_data, scan_num, verbose)
#     det_group = out_file.create_group('detectors_data')
#     worker = get_image_step if scan_mode == 'step' else get_image_fly
#     for detector in detectors:
#         det_group.create_dataset(str(detector), data=get_data(scan_num, detector, verbose, worker), compression='gzip')
#     fast_crds, slow_crds, fast_size, slow_size = get_coords_step(scan_num, verbose) if scan_mode == 'step' else get_coords_fly(scan_num, verbose)
#     write_extra_data(out_file, fast_crds, slow_crds, fast_size, slow_size, verbose)
#     out_file.close()
#     if verbose: print('Done!')
    
# def write_stxm(scan_num, scan_mode, verbose):
#     out_file = create_file(output_path_scan, scan_num, verbose)
#     scan_group = out_file.create_group('scans')
#     fast_crds, slow_crds, fast_size, slow_size = get_coords_step(scan_num, verbose) if scan_mode == 'step' else get_coords_fly(scan_num, verbose)
#     if verbose: print('Reading detector data')
#     worker = get_sum_step if scan_mode == 'step' else get_sum_fly
#     stxm_sums = [get_data(scan_num, detector, verbose, worker) for detector in detectors]
#     stxm_sums = [np.concatenate((stix, np.zeros(fast_size * slow_size - stix.size))).reshape((fast_size, slow_size)) for stix in stxm_sums]
#     for counter, detector in enumerate(detectors):
#         scan_group.create_dataset(detector, data=stxm_sums[counter])
#     write_extra_data(out_file, fast_crds, slow_crds, fast_size, slow_size, verbose)
#     out_file.close()
#     if verbose: print('Done!')

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='P07 data processing script')
#     parser.add_argument('snum', type=int, help='scan number')
#     parser.add_argument('smod', type=str, choices=['step', 'fly'], help='scan mode')
#     parser.add_argument('action', type=str, choices=['data', 'scan'], help='choose between show or save data')
#     parser.add_argument('-v', '--verbose', action='store_true', help='increase output verbosity')
#     args = parser.parse_args()

#     if args.action == 'data':
#         write_data(args.snum, args.smod, args.verbose)
#     else:
#         write_stxm(args.snum, args.smod, args.verbose)