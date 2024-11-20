"""
Data model for Chirp
"""
import typing as t
from pydantic import BaseModel
import numpy as np

from owai.core.utils import IDMixin, GetUnitsMixin
from owai.core.signal_processing_utils import make_multi_chirp


class Chirp(IDMixin, GetUnitsMixin, BaseModel):
    durations: t.List[float] = [5, 5]
    frequencies: t.List[float] = [80, 1000, 21000]
    samplerate: int = 96000  # Samples per second
    freq_units: str = "Hz"
    time_units: str = "s"
    channels: t.Optional[t.List[bool]] = None

    def get(
        self, phi: float = 0, amplitude_func: t.Callable = None, pad_time=0, return_freq: bool = False
    ) -> t.Tuple[np.ndarray, np.ndarray]:
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
        r = make_multi_chirp(
            self.frequencies, self.durations, self.samplerate, amplitude_func, phi, return_freq, pad_time=pad_time
        )
        r = list(r)
        # Expand the chirp to multiple channels
        r[1] = r[1][:, None] * np.array(self.channels)[None, :]
        return r
