import numpy as np, h5py, sys, os, errno, concurrent.futures, pyqtgraph as pg

try:
    from PyQt5 import QtCore, QtGui
except ImportError:
    from PyQt4 import QtCore, QtGui

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
    with concurrent.futures.ProcessPoolExecutor() as executor:
        data = np.concatenate([chunk for chunk in executor.map(worker, filenames)], axis=0)
    if verbose: print("Raw data shape: {}".format(data.shape))
    return data

def create_file(out_path, verbose):
    if verbose: print('Output path: %s' % out_path)
    make_output_dir(out_path)
    return h5py.File(out_path, 'w', libver='latest')

class Viewer(QtGui.QMainWindow):
    def __init__(self):
        super(Viewer, self).__init__()
        self.setWindowTitle('CM Viewer')
        self.box_layout = QtGui.QHBoxLayout()
        self.central_widget = QtGui.QWidget()
        self.setLayout(self.box_layout)
        self.setCentralWidget(self.central_widget)

    def add_image(self, image, label):
        pass