import os
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from scipy.io import wavfile
import soundfile
import numpy as np


from owai.core.signal_processing_utils import to_fourier


def load_test_data(filename: str) -> dict:
    """
    Load test metadata data from a file into a dictionary.

    Parameters
    ----------
    filename : str
        The name of the file to read from.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the loaded test data.
    """
    test_data = None
    with open(filename, "r") as fid:
        test_data = yaml.load(fid, Loader)
    return test_data


def load_wav(filename: str, frequency_domain: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Load waveform data from a WAV file.

    Parameters
    ----------
    filename : str
        The path to the WAV file to be loaded.
    frequency_domain : bool, optional
        If True, return the frequency domain representation of the signal. Default is False, which returns the time domain representation.

    Returns
    -------
    numpy.ndarray
        The time if `frequency_domain` is False, and the frequency bins if True.
    numpy.ndarray
        The waveform data as a NumPy array. If `frequency_domain` is True, the array will contain the frequency domain representation of the signal, otherwise it will contain the time domain representation.
        If a ".dat" file is also present, the data will be scaled by the value in the .dat file.
    int
        The samplerate

    See Also
    --------
    scipy.io.wavfile.read : Read a WAV file and return the data and sample rate as a tuple.
    """

    # Start with the most challenging tip and then generalize later?
    filename_dat = filename.replace(".wav", ".dat").replace(".WAV", ".dat")

    try:
        samplerate, data = wavfile.read(filename)
        if data.dtype.kind == 'i':
            data = data / (np.iinfo(data.dtype).max + 1)  # To match Matlab
    except ValueError as e:
        print(
            "Couldn't read file using scipy, falling back to soundfile. This was the error ",
            str(e),
        )
        data, samplerate = soundfile.read(filename)

    try:
        scalingdata = np.genfromtxt(filename_dat)
        print("Using scaling data from ", filename_dat)
    except FileNotFoundError:
        scalingdata = np.ones(data.shape[1])

    # Apply the scaling
    try:
        data *= scalingdata
    except np.core._exceptions.UFuncTypeError:
        data = data * scalingdata

    if not frequency_domain:
        # Create the time array
        n = data.shape[0]
        t = np.arange(n) / samplerate
        return t, data, samplerate

    f, d = to_fourier(data, samplerate)
    return f, d, samplerate

def save_calibration_data(path : str, calibration_data : dict, convert_array_to_list : bool=False) -> str:
    """Saves calibration data

    Parameters
    ----------
    path : str
        Folder to save calibration data
    calibration_data : dict
        Dictionary of calibration data
    convert_array_to_list : bool, optional
        If True, all ndarrays will be converted to lists of numbers, by default False

    Returns
    -------
    str
        Actual filename saved to. This is automatically time-stamped.
    """
    date = str(np.datetime64("now")).replace(':',".")
    filename = os.path.join(path, "calibration_" + date + ".yaml")
    # Convert any numpy arrays to lists
    def _arr_to_list(d):
        new_d = d.copy()
        for k, v in d.items():
            if isinstance(v, dict):
                new_d[k] = _arr_to_list(v)
            elif isinstance(v, np.ndarray):
                new_d[k] = v.tolist()
        return new_d
    if convert_array_to_list:
        cd = _arr_to_list(calibration_data)
    else:
        cd = calibration_data
    with open(filename, 'w', encoding='utf-8') as fid:
        yaml.dump(cd, fid, allow_unicode=True)
    return filename

def write_wav(path : str, signal :  np.ndarray, samplerate : float, dtype=None):
    """Writes a wavefile to disk, automatically converting to the required dtype

    Parameters
    ----------
    path : str
        Path where to save file
    signal : np.ndarray
        Signal to save
    samplerate : float
        Samplerate
    dtype : np.dtype, optional
        Datatype for saving the file, by default None
    """
    if dtype is None:
        dtype = signal.dtype
    if np.issubdtype(dtype, np.integer) and not np.issubdtype(signal.dtype, np.integer):
        amplitude = np.iinfo(dtype).max
    else:
        amplitude = 1
    wavfile.write(path, samplerate, (signal * amplitude).astype(dtype))
