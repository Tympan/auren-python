"""
Data models for calibration structures
"""
import typing as t
from pydantic import BaseModel
from numpydantic import NDArray, Shape
import numpy as np

from owai.core import io

class RawData(BaseModel):
    samplerate : t.Optional[int] = None
    data : NDArray[Shape["* tube, 4 channel, * signal"], float] = None
    samplerate_ref : t.Optional[int] = None
    data_ref : NDArray[Shape["* tube, * signal"], float] = None

    @property
    def time(self):
        return np.arange(self.data.shape[-1]) / self.samplerate


    def _load_files(self, files):
        timeseries = []
        minsize = np.inf
        samplerate = None
        for file in files:
            _, data, my_samplerate = io.load_wav(file)
            if samplerate is None:
                samplerate = my_samplerate
            else:
                try:
                    assert samplerate == my_samplerate
                except AssertionError:
                    print ("File {} with samplereate {} does not match expected samplerate of {}".format(file, samplerate, self.samplerate))
            timeseries.append(data.T)
            minsize = min(data.shape[0], minsize)

        # Enforce the same shape
        data = np.stack([t[..., :minsize] for t in timeseries], axis=0)
        return data, samplerate

    def load_wav(self, auren_files : t.List[str], ref_files : t.List[str]):
        self.data, self.samplerate = self._load_files(auren_files)
        self.data_ref, self.samplerate_ref = self._load_files(ref_files)
        print("Done loading data.")

