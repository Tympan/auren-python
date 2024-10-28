import json
import matplotlib.pyplot as plt

from owai.core.calibrate import Calibrate
import numpy as np
import owai.core.data_models as odm

base_path = '/mnt/c/Repositories/OpenHearing/open-hearing-hardware/jupyter_notebooks/calibrate'
with open(base_path + "/calibration_data/rawdata_channels.json") as fid:
    rawdata = odm.RawCalibrationData(**json.loads(fid.read()))
cal = Calibrate()
cal_json = cal.model_dump_json()
cal.raw_data = rawdata

cal.raw_data.load_wav()

# cal.plot_calibration_data(show=True)

self = cal
cal.calibrate()
cal_json = cal.model_dump_json()

# Some notes

# The calibration "works", but needs to be verified. And we need some
# verification plots. Almost there.
# Then we have to write out the play.wav files for
# the WAI and swept DPOAe tests.