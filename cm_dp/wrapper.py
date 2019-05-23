import numpy as np, h5py, concurrent.futures, argparse, pyqtgraph as pg
from . import utils
from abc import ABCMeta, abstractmethod, abstractproperty, abstractclassmethod
from functools import partial

class Detector(metaclass=ABCMeta):
    hdf5_data_path = "/entry/instrument/detector/data"

    @abstractproperty
    def raw_filename(self): pass

    @abstractproperty
    def roi(self): pass

    @abstractproperty
    def calib(self): pass
    
    @abstractproperty
    def name(self): pass

    @classmethod
    def apply_mask(cls, data):
        return np.where(cls.calib, data, 0)

    @classmethod
    def filenames(cls, scan_num, verbose):
        return utils.get_filenames(scan_num, cls.raw_filename)

class LambdaFar(Detector):
    raw_filename = "_LambdaFar.nxs"
    roi = (slice(140, 241), slice(146, 247))
    calib = np.invert(utils.calib_file["pixelmask_far"][:].astype(bool))
    name = "lambda_far"

class LambdaDown(Detector):
    raw_filename = "_LambdaDown.nxs"
    roi = (slice(None), slice(None))
    calib = np.invert(utils.calib_file["pixelmask_down"][:].astype(bool))
    name = "lambda_down"

class LambdaUp(Detector):
    raw_filename = "_LambdaUp.nxs"
    roi = (slice(0, 301), slice(None))
    calib = np.invert(utils.calib_file["pixelmask_up"][:].astype(bool))
    name = "lambda_up"

class MotorCoordinates(metaclass=ABCMeta):
    @abstractproperty
    def fast_crds(self): pass

    @abstractproperty
    def slow_crds(self): pass

    @abstractproperty
    def fast_size(self): pass

    @abstractproperty
    def slow_size(self): pass

    @property
    def shape(self):
        return self.fast_size, self.slow_size
    
    def write(self, out_file, verbose):
        if verbose: print("Writing motor coordinates")
        _coord_group = out_file.create_group('motor_coordinates')
        _coord_group.create_dataset('fast_coordinates', data=self.fast_crds)
        _coord_group.create_dataset('slow_coordinates', dat=self.slow_crds)
        _size_group= out_file.create_group('scan_size')
        if verbose: print("Scan size: {}".format(self.shape))
        _size_group.create_dataset('fast_size', data=self.fast_size)
        _size_group.create_dataset('slow_size', data=self.slow_size)

class StepMotorCoordinates(MotorCoordinates):
    fast_crds, slow_crds, fast_size, slow_size = None, None, None, None

    def __init__(self, scan_num, verbose):
        self.fast_crds, self.slow_crds, self.fast_size, self.slow_size = utils.get_coords_step(scan_num, verbose)

class FlyMotorCoordinates(MotorCoordinates):
    fast_crds, slow_crds, fast_size, slow_size = None, None, None, None

    def __init__(self, scan_num, verbose):
        self.fast_crds, self.slow_crds, self.fast_size, self.slow_size = utils.get_coords_fly(scan_num, verbose)

class Scan(object):
    __metaclass__ = ABCMeta

    @abstractproperty
    def scan_num(self): pass
    
    @abstractproperty
    def verbose(self): pass

    @abstractproperty
    def coords(self): pass

    @abstractclassmethod
    def chunk(cls, path, Detector): pass

    @abstractclassmethod
    def chunk_sum(cls, path, Detector): pass

    def data(self, Detector):
        _filenames = Detector.filenames(self.scan_num, self.verbose)
        _worker = partial(self.chunk, Detector=Detector)
        return utils.get_data(_filenames, _worker, self.verbose)

    def stxm(self, Detector):
        _filenames = Detector.filenames(self.scan_num, self.verbose)
        _worker = partial(self.chunk_sum, Detector=Detector)
        return utils.get_data(_filenames, _worker, self.verbose)

    def full_data(self):
        return dict([(_Detector.name, self.data(_Detector)) for _Detector in [LambdaUp, LambdaFar, LambdaDown]])

    def full_stxm(self):
        _det_str = [_Detector.name for _Detector in [LambdaUp, LambdaFar, LambdaDown]]
        _full_stxm = [self.stxm(_Detector) for _Detector in [LambdaUp, LambdaFar, LambdaDown]]
        _full_stxm = [np.concatenate((_stxm / _full_stxm[1], np.zeros(self.coords.fast_size * self.coords.slow_size - _stxm.size))).reshape(self.coords.shape) for _stxm in _full_stxm]
        return dict(zip(_det_str, _full_stxm))

    def write_data(self):
        _out_file = utils.create_file(utils.output_path_data.format(self.scan_num), self.verbose)
        _det_group = _out_file.create_group('detector_data')
        for detector, data in self.full_data().items():
            _det_group.create_dataset(detector, data=data, compression='gzip')
        self.coords.write(_out_file, self.verbose)
        _out_file.close()
        if self.verbose: print("Done!")

    def write_stxm(self):
        _out_file = utils.create_file(utils.output_path_data.format(self.scan_num), self.verbose)
        _det_group = _out_file.create_group('detector_data')
        for detector, stxm in self.full_stxm().items():
            _det_group.create_dataset(detector, data=stxm)
        self.coords.write(_out_file, self.verbose)
        _out_file.close()
        if self.verbose: print("Done!")

    def show_stxm(self):
        app = pg.mkQApp()
        _win = pg.GraphicsWindow(title="STXM viewer")
        _win.resize(640, 480)
        for stxm in self.full_stxm().values():
            _box = _win.addViewBox(lockAspect=True)
            _img = pg.ImageItem(stxm)
            _box.addItem(_img)
        _win.show()
        
class StepScan(Scan):
    scan_num, verbose, coords = None, None, None

    def __init__(self, scan_num, verbose):
        self.scan_num, self.verbose, self.coords = scan_num, verbose, StepMotorCoordinates(scan_num, verbose)

    @classmethod
    def chunk(cls, path, Detector):
        _file = h5py.File(path, 'r')
        _chunk = Detector.apply_mask(np.mean(_file[Detector.hdf5_data_path][:], axis=0))
        return _chunk[np.newaxis, :]

    @classmethod
    def chunk_sum(cls, path, Detector):
        _file = h5py.File(path, 'r')
        _chunk = Detector.apply_mask(np.mean(_file[Detector.hdf5_data_path][:], axis=0))
        return np.array([_chunk.sum()])

class FlyScan(Scan):
    scan_num, verbose, coords = None, None, None

    def __init__(self, scan_num, verbose):
        self.scan_num, self.verbose, self.coords = scan_num, verbose, FlyMotorCoordinates(scan_num, verbose)

    @classmethod
    def chunk(cls, path, Detector):
        _chunk = h5py.File(path, 'r')[Detector.hdf5_data_path][:]
        return np.array([Detector.apply_mask(_frame) for _frame in _chunk])

    @classmethod
    def chunk_sum(cls, path, Detector):
        _chunk = h5py.File(path, 'r')[Detector.hdf5_data_path][:]
        return np.array([Detector.apply_mask(_frame)[Detector.roi].sum() for _frame in _chunk])

def main():
    parser = argparse.ArgumentParser(description='Compton microscopy data processing script')
    parser.add_argument('snum', type=int, help='scan number')
    parser.add_argument('smod', type=str, choices=['step', 'fly'], help='scan mode')
    parser.add_argument('mode', type=str, choices=['save_stxm', 'save_data'], help='choose between saving raw data or stxm')
    parser.add_argument('-v', '--verbose', action='store_true', help='increase output verbosity')
    args = parser.parse_args()

    scan = StepScan(args.snum, args.verbose) if args.smod == 'step' else FlyScan(args.snum, args.verbose)
    if args.action == 'save_data':
        scan.write_data()
    else:
        scan.write_stxm()