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
self=cal

stop

## NEXT STEPS:
# For any 'ear-type' tubes (already implemented)
#   1. Figure the WAI for WAI tones
#   2. Figure the DPOAE for DPOAE tones
# TO DEBUG:
# DPOAE second frequency is NOT at 55dB! it's still at 65 DB! UGH! (check analysis)
# B&K raw data is truncated because of how it gets loaded initially -- manually trim?

wai_auren_path = os.path.join(cal.out_path, 'validation_data', "AUDIO011.WAV")
wai_micW_path = os.path.join(cal.out_path, 'validation_data', "AUDIO013.WAV")
# wai_auren_path = os.path.join(cal.out_path, 'validation_data', "AUDIO012.WAV")
# wai_micW_path = os.path.join(cal.out_path, 'validation_data', "AUDIO014.WAV")
_, wai_auren_wav, _ = io.load_wav(wai_auren_path)
_, wai_micW_wav, _ = io.load_wav(wai_micW_path)

# analyze data
from auren.core.signal_processing_utils import trim_signal_max_correlation_at_f, dft_known_basis
from auren.core import model
import auren
units = auren.units
tone = odm.chirp.Chirp(durations=[10], frequencies=[200, 20000])
_, real_basis, f_at_sample = tone.get(return_freq=True)
_, imag_basis = tone.get(phi=np.pi / 2)

trim_frequency = tone.frequencies[0] * 0.25 + tone.frequencies[-1] * 0.75
block_size = cal.calibration.block_size  # 1024
n_regions = cal.calibration.n_regions  # 256

# Auren to frequency space, calibrated
wai_auren_wav_trim, _ = trim_signal_max_correlation_at_f(
    wai_auren_wav.T[None], f_at_sample, trim_frequency, real_basis, imag_basis, block_size, False
)
wai_f_auren, wai_p_auren_raw = dft_known_basis(
    wai_auren_wav_trim,
    f_at_sample,
    real_basis,
    imag_basis,
    block_size,
    n_regions,
)
wai_p_auren = cal.calibration.cal_p(wai_f_auren, wai_p_auren_raw)

# micW to frequency space, calibrated
wai_micW_wav_trim, _ = trim_signal_max_correlation_at_f(
    wai_micW_wav.T[None], f_at_sample, trim_frequency, real_basis, imag_basis, block_size, False
)
wai_f_micW, wai_p_micW = dft_known_basis(
    wai_micW_wav_trim[0, 0] * cal.tympan_full_scale_voltage / cal.ref_mic_sensitivity * 1000,
    f_at_sample,
    real_basis,
    imag_basis,
    block_size,
    n_regions,
)

wai_f_auren2, wai_p_auren_raw2 = dft_known_basis(
    wai_micW_wav_trim[0, 2:],
    f_at_sample,
    real_basis,
    imag_basis,
    block_size,
    n_regions,
)

# First, check and plot the classical validation where we try to reproduce the micW measurement from the Auren
auren_mic_x = cal.calibration.probe.get_unit("mic_positions")
mic1 = 1
mic2 = 3
cavern_model = model.TwoDiameterTube(
    auren_mic_x[mic1],
    auren_mic_x[mic2],
    cal.calibration.probe.get_unit("length"),
    cal.calibration.probe.get_unit("interior_diameter"),
    cal.raw_data.tubes[2][0].get_unit("interior_diameter"),
)

wai_p_auren_predict = cavern_model.p_measured(
    wai_f_auren * units.Hz,
    cavern_model.xp + cal.raw_data.tubes[2][0].get_unit("probe_locations")[0],
    auren_mic_x[mic1],
    auren_mic_x[mic2],
    wai_p_auren[0, mic1],
    wai_p_auren[0, mic2],
)

fig, axs = plt.subplots(2, 1, sharex=True)
fig.suptitle("Validation Check of the Mic Calibration")
axs[0].semilogx(
    wai_f_auren, todB(wai_p_auren_predict), "C0", lw=4, alpha=0.5, label="Predicted from Mics %s, %s" % (mic1, mic2)
)
axs[0].semilogx(wai_f_micW, todB(wai_p_micW), "C1", lw=2, label="Measured, MicW")
axs[0].semilogx(wai_f_micW, todB(wai_p_auren[0, mic1]), "C2", lw=1, label="Mic %d" % mic1)
axs[0].semilogx(wai_f_micW, todB(wai_p_auren[0, mic2]), "C3", lw=1, label="Mic %d" % mic1)
axs[0].semilogx(wai_f_micW, todB(wai_p_auren_raw[0, mic1]), "C2:", lw=1, label="Mic %d" % mic1)
axs[0].semilogx(wai_f_micW, todB(wai_p_auren_raw[0, mic2]), "C3:", lw=1, label="Mic %d" % mic1)
axs[0].legend()
axs[0].set_ylabel("Amplitude (dB SPL)")
axs[0].set_xlabel("Frequency (Hz)")
axs[1].semilogx(wai_f_auren, todB(wai_p_micW) - todB(wai_p_auren_predict), "C2", lw=3)
axs[1].set_ylabel("Reference - Prediction (Difference of dB)")
axs[1].set_xlabel("Frequency (Hz)")
axs[1].set_ylim(-10, 10)
axs[1].grid()
plt.show()

# Next, check the FPL speaker calibration -- see if that worked out (hoping it did)
k = cavern_model.k(wai_f_auren * units("Hz"))
wai_A0_auren_predict = cavern_model.A0_measured(
    k,
    p0=wai_p_auren[0, mic1],
    p1=wai_p_auren[0, mic2],
)
wai_B0_auren_predict = cavern_model.B0_measured(
    k,
    p0=wai_p_auren[0, mic1],
    p1=wai_p_auren[0, mic2],
)

fig, axs = plt.subplots(2, 1, sharex=True)
fig.suptitle("Validation Check of the Speaker Calibration")
axs[0].semilogx(wai_f_auren, np.abs(wai_B0_auren_predict/wai_A0_auren_predict), "C0", lw=2, label="Reflectance")
# axs[0].semilogx(wai_f_auren, wai_f_auren * 0 + 1, "k", lw=4, alpha=0.5, label="Reflectance")
# axs[0].semilogx(wai_f_auren, wai_f_auren * 0 + 0, "k", lw=4, alpha=0.5, label="Reflectance")
# axs[0].legend()
axs[0].grid()
axs[0].set_ylim(-0.10, 1.10)
axs[0].set_ylabel("Reflectance")
axs[0].set_xlabel("Frequency (Hz)")
axs[1].semilogx(wai_f_auren, wai_f_auren * 0 + cal.speaker_output_db, '--k', label="target")
axs[1].semilogx(wai_f_auren, todB(wai_A0_auren_predict), label="forward")
axs[1].semilogx(wai_f_auren, todB(wai_B0_auren_predict), label="reverse")
axs[1].set_ylabel("Amplitude (dB SPL)")
axs[1].set_xlabel("Frequency (Hz)")
axs[1].grid()
axs[1].legend()
# plt.show()



# Write out WAI chirp
wai_test = wai.WAI(
    calibration=cal.calibration,
    frequencies = [200, 20000],
    duration=10,
    level=65,
    pad_time=0.25
)
wai_tone, wai_level_not_met, wai_weights = wai_test.make_tone()
wai_path = os.path.join(cal.out_path, 'protocol_wav', "PLAY2.WAV")
os.makedirs(os.path.split(wai_path)[0], exist_ok=True)
io.write_wav(wai_path, wai_tone, wai_test.tone.samplerate, np.int16)
# # ... and swept OAE chirp
dpoae_test = dpoae_swept.DPOAE(
    calibration=cal.calibration,
    levels=[55, 65],
    repeats=11,
    duration=1,
    dpoae_ratio = 1.2,
    frequencies = [200, 20000],
    pad_time=0.25
)
dpoae_tone = dpoae_test.make_tone()
dpoae_path = os.path.join(cal.out_path, 'protocol_wav', "PLAY3.WAV")
os.makedirs(os.path.split(dpoae_path)[0], exist_ok=True)
io.write_wav(dpoae_path, dpoae_tone, dpoae_test.tones[0].samplerate, np.int16)


# Analyze WAI data
wai_path = os.path.join(cal.out_path, 'validation_data', "AUDIO015.WAV")
_, wai_wav, _ = io.load_wav(wai_path)
imp_z, refl = wai_test.analyze_iteration(wai_wav)
# plt.semilogx(wai_test.f, 20*np.log10(np.abs(wai_test.p_cal[0].T/20e-6)))
# plt.semilogx(wai_test.f, 20*np.log10(np.abs(wai_test.p_cal_at_mic1[0].T.magnitude/20e-6)), 'k--');plt.show()

# Load data from HotProbe
hp_path = os.path.join(cal.out_path, 'validation_data', "mpu_2024-12-02T10-16-51_right")
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

stop

# Analyze DPOAE data
dpoae_path = os.path.join(cal.out_path, 'calibration_validation', "AUDIO015.WAV")
_, dpoae_wav, _ = io.load_wav(dpoae_path)
dp1, dp2, dpn = dpoae_test.analyze_iteration(dpoae_wav)
dpoae_test.plot_results(show=False)
plt.show()


# Load
# Stuff to , _do:
# Analyze data here C:\Repositories\OpenHearing\open-hearing-hardware\jupyter_notebooks\calibrate\calibration_validation
# -- Check the speaker levels actually achieved based on my calibration
# -- recheck calibration validation -- is it similar to what we found during calibration?
