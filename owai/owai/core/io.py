import os
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from scipy.io import wavfile
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
    with open(filename, 'r') as fid:
        test_data = yaml.load(fid, Loader)
    return test_data

def load_wav(filename : str, frequency_domain : bool=False) -> tuple[np.ndarray, np.ndarray]:
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

    See Also
    --------
    scipy.io.wavfile.read : Read a WAV file and return the data and sample rate as a tuple.
    """

    # Start with the most challenging tip and then generalize later?
    filename_dat = filename.replace('.wav', '.dat').replace('.WAV', '.dat')

    samplerate, data = wavfile.read(filename)

    try:
        scalingdata = np.genfromtxt(filename_dat)
    except:
        scalingdata = np.ones(data.shape[1])

    # Apply the scaling
    data *= scalingdata

    if not frequency_domain:
        # Create the time array
        n = data.shape[0]
        t = np.arange(n) / samplerate
        return t, data

    return to_fourier(data, samplerate)
