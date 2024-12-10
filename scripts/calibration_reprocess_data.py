import matplotlib.pyplot as plt
import os

from auren.core.calibrate import Calibrate
import auren.core.data_models as odm
from auren.core.protocols import wai
from auren.core.protocols import dpoae_swept
from auren.core import io
from auren.core.utils import todB
import numpy as np
from scipy.io import loadmat

try:
    os.chdir('jupyter_notebooks')
except:
    pass

cal = Calibrate(
    out_path='./calibrate',
    ref_mic_sensitivity=9.8,  # mv/Pa

)
cal.save_cal_tones()
import json
with open(os.path.join(cal.out_path, "calibration_data", "rawdata.json"), "r", encoding='utf-8') as fid:
    raw_data = json.loads(fid.read())
    raw_data["data"] = []
    raw_data["data_ref"] = []
    raw_data = odm.RawCalibrationData(**raw_data)
cal.raw_data = raw_data

cal.raw_data.load_wav(auren_files=cal.raw_data.file_meta_data, ref_files=cal.raw_data.file_meta_data_ref)
# Modifications to calibration parameters
cal.calibration_block_size = 4096 // 4
# cal.calibration_ref_start_time_unknown = False
cal.calibration_smoothing_sigma = 9
# cal.channels = [1, 2, 3]
# cal.probe.mic_positions = cal.probe.mic_positions[2:] + cal.probe.mic_positions[:2]  # Nope
cal.noise_freq_threshold = 100 # 100
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
imp_z, refl = wai_test.analyze_iteration(wai_wav)
# plt.semilogx(wai_test.f, 20*np.log10(np.abs(wai_test.p_cal[0].T/20e-6)))
# plt.semilogx(wai_test.f, 20*np.log10(np.abs(wai_test.p_cal_at_mic1[0].T.magnitude/20e-6)), 'k--');plt.show()

# Load data from HotProbe
hp_path = os.path.join(cal.out_path, 'calibration_validation', "mpu_2024-12-02T10-16-51_right")
hp_results = loadmat(hp_path)["results"][0, 0]
hp_refl = hp_results["R_model_ear"]
hp_Z = hp_results["Z_model_ear"][:, 0]
hp_abs = hp_results["A_model_ear"][:, 0]
hp_f = hp_results["chirpgen"][0, 0]["Freq"][:, 0]

fig, axs = wai_test.plot_results(show=False)
axs[0, 0].semilogx(hp_f, 1 - hp_refl)
axs[1, 0].semilogx(hp_f, hp_refl)
axs[0, 1].semilogx(hp_f, todB(hp_Z, ref=wai_test.cavern_model.za1.to_base_units().magnitude))
axs[1, 1].semilogx(hp_f, np.rad2deg(np.angle(hp_Z)))
axs[0, 1].legend(["Auren", "Hot Probe"])
axs[0, 0].set_xlim(200, 20000)
plt.show()



# Load
# Stuff to , _do:
# Analyze data here C:\Repositories\OpenHearing\open-hearing-hardware\jupyter_notebooks\calibrate\calibration_validation
