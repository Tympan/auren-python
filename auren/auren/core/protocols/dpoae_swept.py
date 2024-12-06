""" Function related to Distortion Produce Oto-Acoustic Emissions -- swept version """

from typing import List
import numpy as np
from scipy import ndimage

from auren.core.data_models import calibration
from auren.core.data_models import chirp

def make_tone(calibration: calibration.CalibrationData, tones: List[chirp.Chirp], levels: List[float], pad_time: float=0, phi: float=0) -> np.ndarray:
    """Generates a tone wav file for the Auren speakers based on a calibration and desired chirp characteristics

    Parameters
    ----------
    calibration : calibration.CalibrationData
        The Auren calibration data
    tones : List[chirp.Chirp]
        The characteristics of the chirp, one for each speaker channel
    levels : float
        The dB SPL level at which the chirp should be played, for each channel
    pad_time : float, optional
        Amount of padding added to ramp up the chirp signal, in seconds, by default 0
    phi : float, optional
        The phase of the chirp at the start, by default 0

    Returns
    -------
    np.ndarray(shape=(*, len(tones))
        The chirp waveforms
    """
    # Create the full-scaled amplitude functions fs frequency for each speaker
    amp_funcs = [calibration.cal_speaker(level)[i] for i,level in enumerate(levels)]
    # Generate the chirps, where each speaker tries to produce the full-scale sound
    chirps = [
        tone.get(phi, amplitude_func=amp_func, pad_time=pad_time, return_freq=True)
        for amp_func, tone in zip(amp_funcs, tones)
    ]
    chirp_channels = np.stack([c[1] for c in chirps], axis=1)

    return chirp_channels
