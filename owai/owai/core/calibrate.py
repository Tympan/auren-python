import typing as t
from pydantic import BaseModel
from numpydantic import NDArray, Shape
import numpy as np
import os

import owai.core.data_models as dm
from owai.core.signal_processing_utils import pad_chirp
from owai.core import io

class Calibrate(BaseModel):
    raw_data : dm.calibration.RawData = None
    calibration_tone_specs : t.List[dm.Chirp] = [
        dm.Chirp(duration=5,
                 start_freq=80,
                 end_freq=1000,
                 samplerate=96000),
        dm.Chirp(duration=5,
                 start_freq=80,
                 end_freq=11000,
                 samplerate=96000),
        dm.Chirp(duration=5,
                 start_freq=1000,
                 end_freq=21000,
                 samplerate=96000)
    ]
    calibration_channels : t.List[int] = [0, 0, 1]  # 2 == both
    calibration_pad_time : float = 0.25  # seconds
    calibration_tones : t.Optional[t.List[NDArray]] = None

    out_path : str = "."

    def make_cal_chirps(self) -> t.List[NDArray]:
        tones = [ct.get() for ct in self.calibration_tone_specs]
        pad_tones = [pad_chirp(t[0], t[1], ct.start_freq, ct.end_freq, self.calibration_pad_time)[1]
                      for t, ct in zip(tones, self.calibration_tone_specs)]
        pad_signals = []
        for pad_tone, pad_chan in zip(pad_tones, self.calibration_channels):
            signal = pad_tone[:, None].repeat(2, axis=1)
            if pad_chan == 0:
                signal[:, 1] = 0
            elif pad_chan == 1:
                signal[:, 0] = 0
            pad_signals.append(signal)

        self.calibration_tones = pad_signals
        return pad_tones

    def save_cal_tones(self):
        if self.calibration_tones is None:
            self.make_cal_chirps()
        for i, cts in enumerate(zip(self.calibration_tones, self.calibration_tone_specs)):
            path = os.path.join(self.out_path, "PLAY{}.WAV".format(i + 1))
            io.write_wav(path, cts[0], cts[1].samplerate, np.int16)