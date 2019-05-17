import numpy as np, h5py, sys, os, errno, argparse

parent_path = "/asap3/petra3/gpfs/p07/2019/data/11005196"
user_path = "/gpfs/cfel/cxi/scratch/user/nivanov"
output_path = "hdf5/Scan_{0:d}/scan_{0:d}_data.h5"
scan_path = "raw/scanFrames/Scan_{0:d}"
log_path = "raw/Scans/Scan_{0:d}.log"
hdf5_data_path = "/entry/instrument/detector/data"
detectors = {
                "Lambda_Far": "_LambdaFar.nxs",
                "Lambda_Up": "_LambdaUp.nxs",
                "Lambda_Down": "_LambdaDown.nxs"
            }

def make_output_dir(path):
    try:
        os.makedirs(os.path.dirname(path))
    except OSError as e:
        if e.errno != errno.EEXIST: raise

def get_filenames(scan_num, detector):
    path = os.path.join(parent_path, scan_path.format(scan_num))
    return path, [file for file in os.listdir(path) if file.endswith(detector)]

def get_data(scan_num, detector, verbose):
    dirname, filenames = get_filenames(scan_num, detector)
    if verbose: 
        print("Scan folder: %s" % dirname)
        print("Number of files: %d" % len(filenames))
    filenames.sort()
    raw_data = []
    for filename in filenames:
        scanfile = h5py.File(os.path.join(dirname, filename), 'r')
        point_data = scanfile[hdf5_data_path][:]
        raw_data.append(point_data)
    raw_data = np.array(raw_data)
    if verbose: print("Raw data shape: {}".format(raw_data.shape))
    return raw_data

def write_data(scan_num, verbose):
    out_path = os.path.join(user_path, output_path.format(scan_num))
    if verbose: print('Output path: %s' % out_path)
    make_output_dir(out_path)
    out_file = h5py.File(out_path, 'w', libver='latest')
    det_group = out_file.create_group('Detectors')
    for key, item in detectors.items():
        det_group.create_dataset(str(key), data=get_data(scan_num, item, verbose))
    out_file.close()
    if verbose: print('Done!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='P07 data processing script')
    parser.add_argument('snum', type=int, help='scan number')
    parser.add_argument('-v', '--verbose', action='store_true', help='increase output verbosity')
    args = parser.parse_args()
    write_data(args.snum, args.verbose)
