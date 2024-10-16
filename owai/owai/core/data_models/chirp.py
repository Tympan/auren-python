"""
Data model for Chirp
"""
import typing as t
from pydantic import BaseModel
import numpy as np

from owai.core.signal_processing_utils import make_chirp, pad_chirp

class Chirp(BaseModel):
    duration : float = 5
    start_freq : float = 80
    end_freq : float = 21000
    samplerate : int = 96000  # Samples per second
    freq_units : str = "Hz"
    time_units : str = "s"


    def get(self, phi:float = 0, amplitude_func : t.Callable = None, return_freq : bool=False) -> t.Tuple[np.ndarray, np.ndarray]:
        """Get the chirp time signal

        Parameters
        ----------
        phi : float
            Phase offset, default is 0
        amplitude_func : t.Callable, optional
            A function describing the amplitude as a function of frequency. E.g. lambda freq: freq**2.
            By default None
        return_freq : bool, optional
            If True, will return times, frequency_at_time, signal. By default False

        Returns
        -------
        t.List[np.ndarray, np.ndarray]
            times, data
        """
        discrete_duration = int(np.round(self.samplerate * self.duration)) / self.samplerate
        times = np.linspace(0, discrete_duration, int(self.samplerate * discrete_duration))
        r = make_chirp(times, self.start_freq, self.end_freq, amplitude_func, phi, return_freq)
        if return_freq:
            return tuple([times] + list(r))
        return (times, r)




