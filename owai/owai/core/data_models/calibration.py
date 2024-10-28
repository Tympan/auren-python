"""
Data models for calibration structures
"""
import typing as t
from pydantic import BaseModel
from numpydantic import NDArray, Shape
import numpy as np
import matplotlib.pyplot as plt

from owai.core import io
from owai.core.data_models.calibration_geometry import TubeGeometry
from owai.core.data_models.chirp import Chirp

class RawData(BaseModel):
    samplerate : t.Optional[int] = None
    data : t.Optional[NDArray[Shape["* tube, 4 channel, * signal"], float]] = None
    samplerate_ref : t.Optional[int] = None
    data_ref :  t.Optional[NDArray[Shape["* tube, * signal"], float]] = None

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

class FileMetaData(BaseModel):
    name : str
    tube : TubeGeometry
    tone : Chirp

class RawCalibrationData(BaseModel):
    file_meta_data :  t.Optional[t.List[FileMetaData]] = None
    file_meta_data_ref :  t.Optional[t.List[FileMetaData]] = None
    samplerate : t.Optional[int] = None
    # Since tones can be different lengths, the data is a different list for each tone
    data :  t.Optional[t.List[NDArray[Shape["* tube, 4 channel, * signal"], float]]] = None
    samplerate_ref : t.Optional[int] = None
    # Since tones can be different lengths, the data is a different list for each tone
    data_ref :  t.Optional[t.List[NDArray[Shape["* tube, * signal"], float]]] = None

    # Private variables
    _tones_dict :  t.Optional[dict] = None
    _tones_dict_ref :  t.Optional[dict] = None



    @property
    def time(self):
        return np.arange(self.data.shape[-1]) / self.samplerate

    def _load_files(self, files):
        timeseries = []
        # tones = set()
        # tubes = set()
        tones_dict={}
        minsize = {}
        samplerate = None
        for file in files:
            _, data, my_samplerate = io.load_wav(file.name)
            if samplerate is None:
                samplerate = my_samplerate
            else:
                try:
                    assert samplerate == my_samplerate
                except AssertionError:
                    print ("File {} with samplereate {} does not match expected samplerate of {}".format(file, samplerate, self.samplerate))
            timeseries.append(data.T)
            # tones.add(file.tone.id)
            # tubes.add(file.tube.id)
            tones_dict[file.tone.id] = set(list(tones_dict.get(file.tone.id, set())) + [file.tube.id])
            minsize[file.tone.id] = min(data.shape[0], minsize.get(file.tone.id, np.inf))

        # initialize data array
        # tones = list(tones)
        # tubes = list(tubes)
        data = [np.zeros((len(tones_dict[tone]), timeseries[-1].shape[0], minsize[tone])) for tone in tones_dict]

        for i, file in enumerate(files):
            # Enforce the same shape and populate the nice, standard data structure
            # tone_i = tones.index(file.tone.id)
            tone_i = list(tones_dict.keys()).index(file.tone.id)
            # tube_i = tubes.index(file.tube.id)
            tube_i = list(tones_dict[file.tone.id]).index(file.tube.id)
            data[tone_i][tube_i] = timeseries[i][:, :minsize[file.tone.id]]
        return data, samplerate, tones_dict

    def load_wav(self, auren_files : t.List[FileMetaData] = None, ref_files : t.List[FileMetaData] = None):
        if auren_files is None:
            auren_files = self.file_meta_data
        if ref_files is None:
            ref_files = self.file_meta_data_ref
        self.data, self.samplerate, self._tones_dict = self._load_files(auren_files)
        self.data_ref, self.samplerate_ref, self._tones_dict_ref = self._load_files(ref_files)
        print("Done loading data.")

    def plot(self, kwargs=None, figkwargs=None, show=True):
        if kwargs is None:
            kwargs = {}
        if figkwargs is None:
            figkwargs = {}
        kwargs["Fs"] = kwargs.get("Fs", self.samplerate)
        kwargs["NFFT"] = kwargs.get("NFFT", 2048)
        tone_ids = [file.tone.id for file in self.file_meta_data]

        for i in range(len(self._tones_dict)):
            data = self.data[i]
            tone_id = list(self._tones_dict.keys())[i]
            if tone_id in self._tones_dict_ref:
                has_ref = True
                data_ref = self.data_ref[list(self._tones_dict_ref.keys()).index(tone_id)]
            else:
                has_ref = False
                data_ref = None
            cols = data.shape[1] + has_ref
            rows = data.shape[0]
            fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True, **figkwargs)
            axs = np.atleast_2d(axs)
            my_tone = self.file_meta_data[tone_ids.index(list(self._tones_dict.keys())[i])].tone
            ymin = min(my_tone.frequencies)
            ymax = max(my_tone.frequencies)
            i += 1
            for row in range(rows):
                for col in range(cols - has_ref):
                    axs[row, col].specgram(data[row, col], **kwargs)
                    axs[rows-1, col].set_xlabel("Channel {}, time".format(col))
                axs[row, 0].set_ylabel("Tube {}, frequency".format(row))
                if has_ref:
                    axs[row, cols-1].specgram(data_ref[row, 0], **kwargs)
                    axs[rows-1, cols-1].set_xlabel("Ref Mic")
            axs[0, 0].set_ylim(ymin, ymax)

        if show:
            plt.show()





class MicCalibration(BaseModel):
    i2s_channel_index: int = -1
    units: t.List[str] = ["Hz", "Pa", "radians"]
    description: str = "Infineon-IM72D128"
    id: int = -1
    channel : int = -1
    distance: float = -1
    distance_units : str = "mm"
    cal :  t.Optional[NDArray[Shape["* data, 3 freqAmpPhase"], float]] = None

class SpeakerCalibration(BaseModel):
    id: int = -1
    units: t.List[str] = ["Hz", "Fraction of Full Scale"]
    description: str = "RAB-34832-b148"
    frequency_range: t.List[float] = [100, 20000]
    cal :  t.Optional[NDArray[Shape["* data, 3 freqFrac"], float]] = None

class CalibrationData(BaseModel):
    mic :  t.Optional[t.List[MicCalibration]] = None
    speaker :  t.Optional[t.List[SpeakerCalibration]] = None
    description: str = ""

    _mic_cal_cache = None

    @property
    def n_channels(self):
        return len(self.mic)

    def cal_p(self, f, p):
        p_cal = np.zeros_like(p)

        p_cal = p_cal.reshape(-1, self.n_channels, p_cal.shape[-1])

        for i, mic in enumerate(self.mic):
            amp = np.interp(f, mic.cal[:, 0], mic.cal[:, 1])
            phase = np.interp(f, mic.cal[:, 0], mic.cal[:, 2])
            p_cal[..., i, :] = p[..., i, :] * amp * np.exp(1j * phase)

        return p_cal

