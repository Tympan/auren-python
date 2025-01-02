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
    out_path='./calibrate_2025_01',
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
# cal.calibration_block_size = 4096 // 4
# cal.calibration_ref_start_time_unknown = False
cal.calibration_smoothing_sigma = 9
# cal.channels = [1, 2, 3]
# cal.probe.mic_positions = cal.probe.mic_positions[2:] + cal.probe.mic_positions[:2]  # Nope
cal.noise_freq_threshold = 100 # 100
# cal.calibration_use_fft = False
# cal.calibration_use_global_trim_offset = False
cal.calibrate()


# Plots
# cal.plot_calibration_data(show=True)
cal.plot_calibration(show=False)
# cal.plot_calibration_checks(show=False)
cal.save_calibration()

# Load validation data
with open(os.path.join(cal.out_path, "validation_data", "rawdata.json"), "r", encoding='utf-8') as fid:
    raw_validation_data = json.loads(fid.read())
    raw_validation_data["data"] = []
    raw_validation_data["data_ref"] = []
    raw_validation_data = odm.RawCalibrationData(**raw_validation_data)
cal.raw_validation_data = raw_validation_data
cal.raw_validation_data.load_wav(auren_files=cal.raw_validation_data.file_meta_data, ref_files=cal.raw_validation_data.file_meta_data_ref)


wai_test = wai.WAI(
    calibration=cal.calibration,
    frequencies = [200, 20000],
    duration=10,
    level=65,
    pad_time=0.25
)
dpoae_test = dpoae_swept.DPOAE(
    calibration=cal.calibration,
    levels=[55, 65],
    repeats=11,
    duration=1,
    dpoae_ratio = 1.2,
    frequencies = [200, 20000],
    pad_time=0.25
)

cal.plot_speaker_validation(tones=[wai_test.tone, dpoae_test.tones], show=False)
# cal.plot_speaker_validation(tones=[cal.calibration_mic_tone_specs], show=True)
cal.plot_pressure_validation(show=True)

# Finally, let's look at the WAI for the ear and the B&K

# WAI analysis
# Load data from HotProbe
hp_path = os.path.join(cal.out_path, 'validation_data', "mpu_2024-12-02T10-16-51_right")
hp_results = loadmat(hp_path)["results"][0, 0]
hp_refl = hp_results["R_model_ear"]
hp_Z = hp_results["Z_model_ear"][:, 0]
hp_abs = hp_results["A_model_ear"][:, 0]
hp_f = hp_results["chirpgen"][0, 0]["Freq"][:, 0]

for i, tone in enumerate(cal.raw_validation_data.tones):
    for j, tube in enumerate(cal.raw_validation_data.tubes[i]):
        if not "Ear" in tube.meta:
            continue
        if isinstance(tone[j], (list, tuple)) or tone[j].id != wai_test.tone.id:
            continue
        wai_wav = cal.raw_validation_data.data[i][j]
        imp_z, refl = wai_test.analyze_iteration(wai_wav.T)
        fig, axs = wai_test.plot_results(show=False)
        fig.suptitle(tube.meta)
        axs[0, 0].semilogx(hp_f, 1 - hp_refl)
        axs[1, 0].semilogx(hp_f, hp_refl)
        axs[0, 1].semilogx(hp_f, todB(hp_Z, ref=wai_test.cavern_model.za1.to_base_units().magnitude))
        axs[1, 1].semilogx(hp_f, np.rad2deg(np.angle(hp_Z)))
        axs[0, 1].legend(["Auren", "Hot Probe"])
        axs[0, 0].set_xlim(200, 20000)
plt.show()

# DPOAE
for i, tone in enumerate(cal.raw_validation_data.tones):
    for j, tube in enumerate(cal.raw_validation_data.tubes[i]):
        if not "Ear" in tube.meta:
            continue
        if not isinstance(tone[j], (list, tuple)):
            continue
        dpoae_wav = cal.raw_validation_data.data[i][j]
        dp1, dp2, dpn = dpoae_test.analyze_iteration(dpoae_wav.T)
        fig, axs = dpoae_test.plot_results(show=False)
        fig.suptitle(tube.meta)

plt.show()