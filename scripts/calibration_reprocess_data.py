import matplotlib.pyplot as plt
import os

from auren.core.calibrate import Calibrate
import auren.core.data_models as odm
from auren.core.protocols import wai
from auren.core.protocols import dpoae_swept
from auren.core import io
import numpy as np

try:
    os.chdir('jupyter_notebooks')
except:
    pass

cal = Calibrate(
    out_path='./calibrate',
    ref_mic_sensitivity=9.8,  # mv/Pa

)
# cal.save_cal_tones()
import json
with open(os.path.join(cal.out_path, "rawdata.json"), "r", encoding='utf-8') as fid:
    raw_data = json.loads(fid.read())
    raw_data["data"] = []
    raw_data["data_ref"] = []
    raw_data = odm.RawCalibrationData(**raw_data)
cal.raw_data = raw_data

cal.raw_data.load_wav(auren_files=cal.raw_data.file_meta_data, ref_files=cal.raw_data.file_meta_data_ref)
# Modifications to calibration parameters
# cal.calibration_block_size = 2048
# cal.calibration_ref_start_time_unknown = False
cal.calibration_smoothing_sigma = 0.1
# cal.channels = [1, 2, 3]
# cal.probe.mic_positions = cal.probe.mic_positions[2:] + cal.probe.mic_positions[:2]  # Nope
cal.noise_freq_threshold = 1000 # 100
cal.calibration_use_fft = False
cal.calibration_use_global_trim_offset = False
cal.calibrate()

# Plots
# cal.plot_calibration_data(show=True)
# cal.plot_calibration(show=False)
# cal.plot_calibration_checks(show=True)
cal.save_calibration()

# Write out WAI chirp
wai_test = wai.WAI(
    calibration=cal.calibration,
    tone=odm.chirp.Chirp(durations=[10], frequencies=[200, 20000]),
    level=65,
    pad_time=0.25
)
wai_tone, wai_level_not_met, wai_weights = wai_test.make_tone()
wai_path = os.path.join(cal.out_path, 'protocol_wav', "PLAY1.WAV")
os.makedirs(os.path.split(wai_path)[0], exist_ok=True)
io.write_wav(wai_path, wai_tone[:, ::-1].copy(), wai_test.tone.samplerate, np.int16)
# ... and swept OAE chirp
dpoae_ratio = 1.2
dpoae_chirp = [
    odm.chirp.Chirp(durations=[10], frequencies=[200, 16000]),
    odm.chirp.Chirp(durations=[10], frequencies=[200 * dpoae_ratio, 16000 * dpoae_ratio])
]
dpoae_tone = dpoae_swept.make_tone(cal.calibration, dpoae_chirp, levels=[60, 65], pad_time=0.25)
dpoae_path = os.path.join(cal.out_path, 'protocol_wav', "PLAY2.WAV")
os.makedirs(os.path.split(dpoae_path)[0], exist_ok=True)
io.write_wav(dpoae_path, dpoae_tone[:, ::-1].copy(), dpoae_chirp[0].samplerate, np.int16)


# Analyze WAI data
wai_path = os.path.join(cal.out_path, 'calibration_validation', "AUDIO014.WAV")
_, wai_wav, _ = io.load_wav(wai_path)
wai_test.analyze_iteration(wai_wav)
# plt.semilogx(wai_test.f, 20*np.log10(np.abs(wai_test.p_cal[0].T/20e-6)))
# plt.semilogx(wai_test.f, 20*np.log10(np.abs(wai_test.p_cal_at_mic1[0].T.magnitude/20e-6)), 'k--');plt.show()
wai_test.plot_results({}, {})
# Stuff to , _do:
# Analyze data here C:\Repositories\OpenHearing\open-hearing-hardware\jupyter_notebooks\calibrate\calibration_validation
# * FIX SPEAKER calibration ORDER!! (YIKES!)
# * Clean up and document for HARMAN