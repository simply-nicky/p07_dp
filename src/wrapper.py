import numpy as np, h5py, concurrent.futures
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
    
    @abstractclassmethod
    def __str__(cls): pass

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

    @classmethod
    def __str__(cls):
        return "lambda_far"

class LambdaDown(Detector):
    raw_filename = "_LambdaDown.nxs"
    roi = (slice(None), slice(None))
    calib = np.invert(utils.calib_file["pixelmask_down"][:].astype(bool))

    @classmethod
    def __str__(cls):
        return "lambda_down"

class LambdaUp(Detector):
    raw_filename = "_LambdaUp.nxs"
    roi = (slice(0, 301), slice(None))
    calib = np.invert(utils.calib_file["pixelmask_up"][:].astype(bool))

    @classmethod
    def __str__(cls):
        return "lambda_up"

class StepMotorCoordinates(object):
    def __init__(self, scan_num, verbose):
        self.fast_crds, self.slow_crds, self.fast_size, self.slow_size = utils.get_coords_step(scan_num, verbose)

    @property
    def shape(self):
        return self.fast_size, self.slow_size

class FlyMotorCoordinates(object):
    def __init__(self, scan_num, verbose):
        self.fast_crds, self.slow_crds, self.fast_size, self.slow_size = utils.get_coords_fly(scan_num, verbose)

    @property
    def shape(self):
        return self.fast_size, self.slow_size

class Scan(object):
    __metaclass__ = ABCMeta

    @abstractproperty
    def scan_num(self): pass

    @abstractproperty
    def scan_mode(self): pass
    
    @abstractproperty
    def verbose(self): pass

    @abstractproperty
    def coords(self): pass

    @abstractclassmethod
    def chunk(path, Detector): pass

    @abstractclassmethod
    def chunk_sum(path, Detector): pass

    def data(self, Detector):
        _filenames = Detector.filenames(self.scan_num, self.verbose)
        _worker = partial(Scan.chunk, Detector=Detector)
        return utils.get_data(_filenames, _worker, self.verbose)

    def stxm(self, Detector):
        _filenames = Detector.filenames(self.scan_num, self.verbose)
        _worker = partial(Scan.chunk_sum, Detector=Detector)
        return utils.get_data(_filenames, _worker, self.verbose)

    def full_data(self):
        return dict([(str(_Detector), self.data(_Detector)) for _Detector in [LambdaUp, LambdaFar, LambdaDown]])

    def full_stxm(self):
        _det_str = [str(_Detector) for _Detector in [LambdaUp, LambdaFar, LambdaDown]]
        _full_stxm = [self.stxm(_Detector) for _Detector in [LambdaUp, LambdaFar, LambdaDown]]
        _full_stxm = [utils.pad_stxm(_stxm / _full_stxm[1], self.coords.fast_size, self.coords.slow_size).reshape(self.coords.shape) for _stxm in _full_stxm]
        return dict(zip(_det_str, _full_stxm))

    def write(self):
        pass
        
class StepScan(Scan):
    def __init__(self, scan_num, scan_mode, verbose):
        self.scan_num, self.scan_mode, self.verbose = scan_num, scan_mode, verbose

    @classmethod
    def chunk(path, Detector):
        _file = h5py.File(path, 'r')
        _chunk = Detector.apply_mask(np.mean(_file[Detector.hdf5_data_path][:], axis=0))
        return _chunk[np.newaxis, :]

    @classmethod
    def chunk_sum(path, Detector):
        _file = h5py.File(path, 'r')
        _chunk = Detector.apply_mask(np.mean(_file[Detector.hdf5_data_path][:], axis=0))
        return np.array([_chunk.sum()])

class FlyScan(Scan):
    def __init__(self, scan_num, scan_mode, verbose):
        self.scan_num, self.scan_mode, self.verbose = scan_num, scan_mode, verbose

    @classmethod
    def chunk(path, Detector):
        _chunk = h5py.File(path, 'r')[Detector.hdf5_data_path][:]
        return np.array([Detector.apply_mask(_frame) for _frame in _chunk])

    @classmethod
    def chunk_sum(path, Detector):
        _chunk = h5py.File(path, 'r')[Detector.hdf5_data_path][:]
        return np.array([Detector.apply_mask(_frame)[Detector.roi].sum() for _frame in _chunk])