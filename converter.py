import numpy as np, h5py, sys, os, errno, concurrent.futures, argparse, matplotlib.pyplot as plt
from math import sqrt

parent_path = "/asap3/petra3/gpfs/p07/2019/data/11005196"
output_path_data = "../hdf5/Scan_{0:d}/scan_{0:d}_data.h5"
output_path_scan = "../hdf5/Scan_{0:d}/scan_{0:d}.h5"
scan_path = "raw/scanFrames/Scan_{0:d}"
log_path = "raw/Scans/Scan_{0:d}.log"
hdf5_data_path = "/entry/instrument/detector/data"

detectors = {"lambda_far": "_LambdaFar.nxs", "lambda_up": "_LambdaUp.nxs", "lambda_down": "_LambdaDown.nxs"}
rois = {"lambda_far": (slice(140, 241), slice(146, 247)), "lambda_up": (slice(0, 301), slice(None)), "lambda_down": (slice(None), slice(None))}
header = '#'
sizeline = '# Points count:'

def make_output_dir(path):
    try:
        os.makedirs(os.path.dirname(path))
    except OSError as e:
        if e.errno != errno.EEXIST: raise

def get_filenames(scan_num, detector):
    path = os.path.join(parent_path, scan_path.format(scan_num))
    return path, [file for file in os.listdir(path) if file.endswith(detector)]

def get_coords(scan_num, verbose):
    logpath = os.path.join(parent_path, log_path.format(scan_num))
    if verbose: print("Log path: {}".format(logpath))
    lines, sizes = [], []
    for line in open(logpath, 'r'):
        if line.startswith(header):
            if line.startswith(sizeline):
                try:
                    sizes.append(int(line.strip(sizeline)))
                except:
                    continue
        else:
            lines.append(line)
    fast_size, slow_size = sizes[2], sizes[1]
    del lines[1::(2 * fast_size + 1)]
    slow_crds, fast_crds = [], []
    for line in lines[1::2]:
        parts = line.split(';')
        try:
            slow_crds.append(float(parts[-2].strip('um')))
            fast_crds.append(float(parts[-1].strip('um\n')))
        except:
            continue
    if verbose: print("Number of coordinates: {:d}".format(len(fast_crds)))
    return np.array(fast_crds), np.array(slow_crds), fast_size, slow_size

def get_point_data(path):
    scanfile = h5py.File(path, 'r')
    point_data = scanfile[hdf5_data_path][:]
    return np.mean(point_data, axis=0)

def get_data(scan_num, detector, verbose):
    dirname, filenames = get_filenames(scan_num, detector)
    if verbose: 
        print("Scan folder: %s" % dirname)
        print("Number of files: %d" % len(filenames))
    filenames.sort()
    filenames = [os.path.join(dirname, filename) for filename in filenames]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        raw_data = np.array([point_data for point_data in executor.map(get_point_data, filenames)])
    if verbose: print("Raw data shape: {}".format(raw_data.shape))
    return raw_data

def get_stix(scan_num, verbose):
    if verbose: print('Reading motor coordinates')
    fast_crds, slow_crds, fast_size, slow_size = get_coords(scan_num, verbose)
    if verbose: print('Reading detector data')
    stix_sums = [get_data(scan_num, detector, verbose)[roi].sum(axis=(-2, -1)) for roi, detector in zip(rois, detectors.values())]
    stix_sums_full = [np.concatenate((stix, np.zeros(fast_size * slow_size - stix.size))).reshape((fast_size, slow_size)) for stix in stix_sums]
    return stix_sums_full, fast_crds, slow_crds, fast_size, slow_size

def show_data(scan_num, verbose):
    stix_sums, fast_crds, slow_crds = get_stix(scan_num, verbose)[0:3]
    for stix, detector in zip(stix_sums, detectors):
        fig, ax = plt.subplots(1, 1)
        ax.imshow(stix, extent=[fast_crds.min(), fast_crds.max(), slow_crds.min(), slow_crds.max()], cmap='gist_gray')
        ax.set_title(detector)
        fig.show()

def create_file(output_path, scan_num, verbose):
    out_path = os.path.join(os.path.dirname(__file__), output_path.format(scan_num))
    if verbose: print('Output path: %s' % out_path)
    make_output_dir(out_path)
    return h5py.File(out_path, 'w', libver='latest')

def write_extra_data(out_file, fast_crds, slow_crds, fast_size, slow_size):
    coord_group = out_file.create_group('motor_coordinates')
    coord_group.create_dataset('fast_coordinates', data=fast_crds)
    coord_group.create_dataset('slow_coordinates', data=slow_crds)
    size_group = out_file.create_group('scan_size')
    size_group.create_dataset('fast_size', data=fast_size)
    size_group.create_dataset('slow_size', data=slow_size)

def write_data(scan_num, verbose):
    out_file = create_file(output_path_data, scan_num, verbose)
    det_group = out_file.create_group('detectors_data')
    for key, item in detectors.items():
        det_group.create_dataset(str(key), data=get_data(scan_num, item, verbose), compression='gzip')
    if verbose: print("writing supplementary data")
    fast_crds, slow_crds, fast_size, slow_size = get_coords(scan_num, verbose)
    write_extra_data(out_file, fast_crds, slow_crds, fast_size, slow_size)
    out_file.close()
    if verbose: print('Done!')
    
def write_stix(scan_num, verbose):
    out_file = create_file(output_path_scan, scan_num, verbose)
    scan_group = out_file.create_group('scans')
    stix_sums, fast_crds, slow_crds, fast_size, slow_size = get_stix(scan_num, verbose)
    for counter, detector in enumerate(detectors):
        scan_group.create_dataset(detector, data=stix_sums[counter])
    if verbose: print("writing supplementary data")
    write_extra_data(out_file, fast_crds, slow_crds, fast_size, slow_size)
    out_file.close()
    if verbose: print('Done!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='P07 data processing script')
    parser.add_argument('snum', type=int, help='scan number')
    parser.add_argument('action', type=str, choices=['show', 'save_data', 'save_scan'], help='choose between show or save data')
    parser.add_argument('-v', '--verbose', action='store_true', help='increase output verbosity')
    args = parser.parse_args()

    if args.action == 'save_data':
        write_data(args.snum, args.verbose)
    elif args.action == 'save_scan':
        write_stix(args.snum, args.verbose)
    else:
        show_data(args.snum, args.verbose)