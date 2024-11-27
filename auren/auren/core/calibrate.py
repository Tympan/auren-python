import os
import time
import typing as t

import matplotlib.pyplot as plt
import numpy as np
from numpydantic import NDArray, Shape
from pydantic import BaseModel
from scipy import ndimage

import auren
import auren.core.data_models as dm
from auren.core import io, model
from auren.core.signal_processing_utils import dft_known_basis, pad_chirp, to_fourier
from auren.core.tympan_serial import TympanSerial
from auren.core.utils import todB


class Calibrate(BaseModel):

    ### INPUTS section ###

    raw_data: t.Optional[dm.calibration.RawCalibrationData] = None
    calibration_mic_tone_specs: dm.Chirp = dm.Chirp(
        durations=[5, 5], frequencies=[80, 1000, 21000], samplerate=96000, channels=[True, True]
    )
    calibration_speaker_tone_specs: t.List[dm.Chirp] = [
        dm.Chirp(durations=[5], frequencies=[80, 21000], channels=[True, False], samplerate=96000),
        dm.Chirp(durations=[5], frequencies=[80, 21000], samplerate=96000, channels=[False, True]),
    ]
    calibration_pad_time: float = 0.25  # seconds
    calibration_ref_start_time_unknown: bool = False
    calibration_data_path: str = "calibration_data"
    calibration_tone_db: float = -10  # dB FS
    speaker_output_db: float = 65  # dB SPL

    calibration_block_size: int = 4096
    number_of_calibration_frequency_bins: int = 256
    bin_oversampling: int = 5
    calibration_smoothing_sigma: float = 9
    calibration_smoothing_sigma_speaker: float = 0.01

    channels: t.List[int] = [1, 2, 3]  # which channels to use for calibration, need at least 3

    # Below this frequency, we don't calibrate. We just assume the phase and relative mic amplitudes match
    # This is implemented in _relative_calibration
    noise_freq_threshold: float = 100

    probe: dm.Probe = dm.Probe()

    sound_speed: float = 343  # m/s

    ref_mic_sensitivity: float = 3.83  # mv/Pa
    tympan_full_scale_voltage: float = 0.375 * np.sqrt(2)  # Volts FS
    trim_frequency: float = 16000  # Frequency used to figure out the data offset and then trim the signal

    ### OUTPUTS Section ###

    calibration: dm.CalibrationData = None

    # Add ALL the calibration information for debugging and archeology
    trimmed_data: t.Optional[NDArray[Shape["* tubes, * times, * channels"], float]] = None
    trimmed_data_ref: t.Optional[NDArray[Shape["* tubes, * times, * channels"], float]] = None
    trim_slices: t.Optional[t.List] = None
    rel_cal: t.Optional[NDArray[Shape["* frequencies, * channels"], complex]] = None
    abs_cal: t.Optional[NDArray[Shape["* frequencies"], float]] = None
    p_cal: t.Optional[NDArray[Shape["* tubes, * channels, * frequencies"], complex]] = None
    cal: t.Optional[NDArray[Shape["* channels, * frequencies, 3 freqAmpPhase"], float]] = None
    fourier: t.Optional[NDArray[Shape["* tubes, * channels, * frequencies"], complex]] = None
    fourier_ref: t.Optional[NDArray[Shape["* tubes, * frequencies"], complex]] = None
    fourier_freq: t.Optional[NDArray[Shape["* frequencies"], complex]] = None
    real_basis: t.Optional[t.List[NDArray]] = None
    imag_basis: t.Optional[t.List[NDArray]] = None
    f_at_sample: t.Optional[NDArray[Shape["* frequencies"], float]] = None

    # Private/working attributes, not part of the data model but needed for the class to function
    _cavern_model: model.Model = model.StraightTube(0, 0, 0, 0, 0)
    _calibration_tones = None
    _calibration_tones_speaker = None
    _tympan = None
    _start_tone_played = False

    out_path: str = "."

    def k(self, f):
        return np.pi * 2 * f / self.sound_speed

    def connect(self):
        self._tympan = TympanSerial()
        self._tympan.connect()
        self._start_tone_played = False

    def connect_sd(self):
        self._tympan.send_char(">")

    def collect_mic_calibration_data(self, tube):
        if not self._start_tone_played:
            self._tympan.send_char("1")
            self._start_tone_played = True
            print(self._tympan.read_all(5))
        print(self._tympan.read_all(0.5))

        # For this data, we will record the filename, the tube, and the chirp, and add it to a list
        if self.raw_data is None:
            self.raw_data = dm.RawCalibrationData(file_meta_data=[], file_meta_data_ref=[])

        # select the correct channels before playing
        self._tympan.send_char("E")
        print(self._tympan.read_all(1))
        tone = self.calibration_mic_tone_specs
        # Clear the buffer
        print(self._tympan.read_all(1))
        print("playing tone ", tone)
        # send the play/record command
        self._tympan.send_char("4")
        # Get the reply
        reply = ""
        count = 0
        while "Auto-stopping SD recording of AUDIO" not in reply:
            reply += self._tympan.read_all(0.5)
            count += 1
            if count > np.sum(tone.durations) / 0.5 * 2:
                break  # avoid infinite loops, allow twice the time

        try:
            name = "AUDIO" + reply.split("AUDIO")[1].split(".")[0] + ".WAV"
            print("Success: ", reply)
        except IndexError:
            name = ["[error]"]
            print("Error: ", reply)

        print("Results saved in ", name)

        spec = dm.FileMetaData(tube=tube, tone=tone, name=os.path.join(self.out_path, self.calibration_data_path, name))
        self.raw_data.file_meta_data.append(spec)

        # select the correct channels before playing
        self._tympan.send_char("o")
        print(self._tympan.read_all(1))
        # Clear the buffer
        print(self._tympan.read_all(4))
        print("playing tone ", tone)
        # send the play/record command
        self._tympan.send_char("4")
        # Get the reply
        reply = self._tympan.read_all(0.5)
        count = 0
        while "Auto-stopping SD recording of AUDIO" not in reply:
            reply += self._tympan.read_all(0.5)
            count += 1
            if count > np.sum(tone.durations) / 0.5 * 2:
                break  # avoid infinite loops, allow twice the time

        try:
            name = "AUDIO" + reply.split("AUDIO")[1].split(".")[0] + ".WAV"
            print("Success: ", reply)

        except IndexError:
            print("Error: ", reply)

        print("Results saved in ", name)

        spec = dm.FileMetaData(tube=tube, tone=tone, name=os.path.join(self.out_path, self.calibration_data_path, name))
        self.raw_data.file_meta_data_ref.append(spec)

    def collect_speaker_calibration_data(self, tube):
        if not self._start_tone_played:
            self._tympan.send_char("1")
            self._start_tone_played = True
            print(self._tympan.read_all(5))
        print(self._tympan.read_all(0.5))

        # For this data, we will record the filename, the tube, and the chirp, and add it to a list
        if self.raw_data is None:
            self.raw_data = dm.RawCalibrationData(file_meta_data=[], file_meta_data_ref=[])

        # select the correct channels before playing and clear the buffer
        self._tympan.send_char("E")
        print(self._tympan.read_all(1))
        for i, tone in enumerate(self.calibration_speaker_tone_specs):
            # Clear the buffer
            print(self._tympan.read_all(1))
            print("playing tone ", tone)
            # send the play/record command
            self._tympan.send_char(str(5 + i))
            # Get the reply
            reply = self._tympan.read_all(0.5)
            count = 0
            while "Auto-stopping SD recording of AUDIO" not in reply:
                reply += self._tympan.read_all(0.5)
                count += 1
                if count > np.sum(tone.durations) / 0.5 * 2:
                    break  # avoid infinite loops, allow twice the time

            try:
                name = "AUDIO" + reply.split("AUDIO")[1].split(".")[0] + ".WAV"
            except IndexError:
                print("Error: ", reply)

            print("Results saved in ", name)

            spec = dm.FileMetaData(
                tube=tube, tone=tone, name=os.path.join(self.out_path, self.calibration_data_path, name)
            )
            self.raw_data.file_meta_data.append(spec)

    def make_cal_chirps(self) -> t.List[NDArray]:
        amp = 10 ** (self.calibration_tone_db / 20)
        tones = [self.calibration_mic_tone_specs.get(pad_time=self.calibration_pad_time)[1] * amp]
        speaker_tones = [
            csts.get(pad_time=self.calibration_pad_time)[1] * amp for csts in self.calibration_speaker_tone_specs
        ]

        self._calibration_tones = tones
        self._calibration_tones_speaker = speaker_tones
        return tones + speaker_tones

    def save_cal_tones(self, subpath="calibration_tones"):
        if self._calibration_tones is None:
            self.make_cal_chirps()
        os.makedirs(os.path.join(self.out_path, subpath), exist_ok=True)
        for i, cts in enumerate(
            zip(
                self._calibration_tones + self._calibration_tones_speaker,
                [self.calibration_mic_tone_specs] + self.calibration_speaker_tone_specs,
            )
        ):
            path = os.path.join(self.out_path, subpath, "PLAY{}.WAV".format(i + 1))
            io.write_wav(path, cts[0], cts[1].samplerate, np.int16)

    def calibrate(self):
        # Step 1 -- time-domain to frequency domain
        # 1a, make the basis functions
        real_basis = self.calibration_mic_tone_specs.get()
        imag_basis = self.calibration_mic_tone_specs.get(phi=np.pi / 2)
        f_at_sample = self.calibration_mic_tone_specs.get(return_freq=True)[2]

        n_samples = real_basis[0].shape[-1]

        # Step 2: Fourier transform the raw data
        data_id = list(self.raw_data._tones_dict.keys()).index(self.calibration_mic_tone_specs.id)
        # Trim the data
        self.trimmed_data, self.trim_slices = self._trim_data(
            self.raw_data.data[data_id], real_basis, imag_basis, f_at_sample, n_samples, trim_freq=self.trim_frequency
        )
        f_centers, fourier = self._fourier_transform_raw_data(
            self.trimmed_data,
            real_basis,
            imag_basis,
            f_at_sample,
            n_samples,
        )

        # Step 3: Fourier transform the reference data
        data_id = list(self.raw_data._tones_dict_ref.keys()).index(self.calibration_mic_tone_specs.id)
        # FS Wav * V / (mv / Pa) * 1000 mv / V
        data_ref = (
            self.raw_data.data_ref[data_id][:, 0] * self.tympan_full_scale_voltage / self.ref_mic_sensitivity * 1000
        )
        self.trimmed_data_ref = np.zeros((data_ref.shape[0], self.trimmed_data.shape[-1]))
        for i, slc in enumerate(self.trim_slices):
            self.trimmed_data_ref[i] = data_ref[i, slc]
        #
        f_centers_ref, fourier_ref = self._fourier_transform_ref_data(
            self.trimmed_data_ref,
            real_basis,
            imag_basis,
            f_at_sample,
            n_samples,
        )

        # Step 4: Complete the relative calibration
        rel_cal = self._relative_calibration(f_centers, fourier, self.probe, self.channels)

        # Step 5: Complete the absolute calibration
        data_id = list(self.raw_data._tones_dict_ref.keys()).index(self.calibration_mic_tone_specs.id)
        tubes = self.raw_data.tubes_ref[data_id]
        # abs_cal = self._absolute_calibration(f_centers, fourier, fourier_ref, rel_cal, self.probe, tubes, [self.channels[0], self.channels[-1]])
        abs_cal = self._absolute_calibration(f_centers, fourier, fourier_ref, rel_cal, self.probe, tubes, self.channels)

        # Step 6: Combine and smooth the calibration
        cal = self._combine_smooth_cal(f_centers, rel_cal, abs_cal, self.channels, self.calibration_smoothing_sigma)

        # Step 7: Create the summarized calibration objects for the mics and start populating the full calibration object
        mics = [dm.MicCalibration(cal=c, channel=i) for i, c in enumerate(cal)]
        cal_obj = dm.CalibrationData(mic=mics, calibrated_channels=self.channels)

        # Calibrate the pressure for the calibration data (to do self-consistency test)
        p_cal = cal_obj.cal_p(f_centers, fourier) * auren.units("Pa")

        # Save ALL the calibration information up to this point
        self.calibration = cal_obj
        self.rel_cal = rel_cal
        self.abs_cal = abs_cal
        self.cal = cal
        self.p_cal = p_cal
        self.fourier = fourier
        self.fourier_ref = fourier_ref
        self.fourier_freq = f_centers
        self.real_basis = real_basis
        self.imag_basis = imag_basis
        self.f_at_sample = f_at_sample

        # Step 8: Calibrate the speakers
        # Speaker
        speaker_cal = self._calibrate_speakers()
        speakers = [
            dm.SpeakerCalibration(id=i, frequency_range=tone.frequencies, cal=speaker_cal[i], calibrated_level=self.speaker_output_db)
            for i, tone in enumerate(self.calibration_speaker_tone_specs)
        ]
        self.calibration.speaker = speakers

    def calibrateAnalogMic(self):
        TRIAL_DURATION_S = 5
        reply = bytes()

        # Stop reporting level
        self._tympan.send_char("L")

        # Set Tympan input channel to mic jack with bias
        self._tympan.send_char("o")
        # Clear the buffer
        reply = self._tympan.read_all(timeout_s=1)
        print(reply)

        # Start recording
        self._tympan.send_char("r")
        reply = self._tympan.read_line(timeout_s=2, eof_str=".WAV")
        print(reply.decode("utf-8"))

        reply = self._tympan.read_line(timeout_s=2, eof_str=".WAV")
        print(reply.decode("utf-8"))

        # Start reporting level
        self._tympan.send_char("l")

        # Print level until recording has stopped
        curr_time = time.time()
        while time.time() < (curr_time + TRIAL_DURATION_S):
            reply = self._tympan.read_line(timeout_s=1, eof_str="dBFS")
            if "dBFS" in reply.decode("utf-8"):
                print(reply.decode("utf-8"))

        # Stop recording
        self._tympan.send_char("s")
        reply = self._tympan.read_line(timeout_s=2, eof_str=".WAV")
        print(reply.decode("utf-8"))

        # Record wav filename
        try:
            name = "AUDIO" + reply.split("AUDIO")[1].split(".")[0] + ".WAV"
        except:
            name = "error parsing filename"
        # Stop reporting level
        self._tympan.send_char("L")

        # Clear the buffer
        print(self._tympan.read_all(1))

        return name

    def save_calibration(self, path=None):
        if path is None:
            path = os.path.join(self.out_path, "calibration.json")
        self.calibration.save(path)

    ######## PLOT FUNCTIONS
    def plot_calibration_data(self, kwargs={}, figkwargs={}, show=True):
        self.raw_data.plot(kwargs, figkwargs, show)

    def plot_calibration(self, kwargs={}, figkwargs={}, show=True):
        self.calibration.plot(kwargs=kwargs, figkwargs=figkwargs, show=show)

    def plot_calibration_checks(self, kwargs={}, figkwargs={}, show=True):

        # Plot the fourier transforms, to make sure they look reasonable
        classic_bins, classic_fourier = to_fourier(self.trimmed_data, self.raw_data.samplerate)
        classic_bins_ref, classic_fourier_ref = to_fourier(self.trimmed_data_ref, self.raw_data.samplerate)
        tone_freqs = self.calibration_mic_tone_specs.frequencies
        tone_durations = self.calibration_mic_tone_specs.durations

        classic_fourier = classic_fourier[..., (classic_bins >= tone_freqs[0]) & (classic_bins <= tone_freqs[-1])]
        classic_bins = classic_bins[(classic_bins >= tone_freqs[0]) & (classic_bins <= tone_freqs[-1])]

        classic_fourier_ref = classic_fourier_ref[
            ..., (classic_bins_ref >= tone_freqs[0]) & (classic_bins_ref <= tone_freqs[-1])
        ]
        classic_bins_ref = classic_bins_ref[(classic_bins_ref >= tone_freqs[0]) & (classic_bins_ref <= tone_freqs[-1])]

        classic_fourier_scale = (
            np.sqrt(2)
            / np.sqrt(
                tone_durations[0]
                / classic_bins
                / (np.log(tone_freqs[1]) - np.log(tone_freqs[0]))
                * (classic_bins < tone_freqs[1])
                + tone_durations[1]
                / classic_bins
                / (np.log(tone_freqs[2]) - np.log(tone_freqs[1]))
                * (classic_bins > tone_freqs[1])
            )
            * np.sum(tone_durations)
        )

        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, **figkwargs)
        fig.suptitle("Comparison (FFT vs. Block) of Fourier Amplitude")
        for i in range(4):
            ii = i // 2
            jj = i % 2
            axs[ii, jj].semilogx(self.fourier_freq / 1000, todB(self.fourier_ref[i]).T, "k", lw=2, **kwargs)
            axs[ii, jj].semilogx(self.fourier_freq / 1000, todB(self.fourier[i, self.channels]).T, lw=2, **kwargs)
            axs[ii, jj].set_prop_cycle(None)
            axs[ii, jj].semilogx(
                classic_bins / 1000,
                todB(classic_fourier[i, self.channels] * classic_fourier_scale).T,
                alpha=0.5,
                lw=4,
                **kwargs,
            )
            axs[ii, jj].semilogx(
                classic_bins_ref / 1000,
                todB(classic_fourier_ref[i] * classic_fourier_scale).T,
                "k",
                alpha=0.5,
                lw=4,
                **kwargs,
            )
            axs[ii, jj].set_title(f"Tube {i}")
        axs[0, 0].legend(["Ref Mic"] + self.channels)
        axs[0, 0].set_xlabel("Frequency(kHz)")
        axs[0, 0].set_ylabel("Amplitude (dB SPL uncal)")
        # plt.show()
        expected_fourier = dft_known_basis(
            self.real_basis[1][..., 0],
            self.f_at_sample,
            self.real_basis[1][..., 0],
            self.imag_basis[1][..., 0],
            block_size=self.calibration_block_size,
            n_regions=self.number_of_calibration_frequency_bins * self.bin_oversampling,
        )
        expected_phase = np.angle(expected_fourier[1])
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, **figkwargs)
        fig.suptitle("Comparison (FFT vs. Block) of Fourier Phase")
        for i in range(4):
            ii = i // 2
            jj = i % 2
            axs[ii, jj].semilogx(self.fourier_freq / 1000, expected_phase * 0, "k", alpha=0.4, **kwargs)
            axs[ii, jj].semilogx(
                self.fourier_freq / 1000,
                np.rad2deg(np.angle(self.fourier[i, self.channels]) - expected_phase * 0).T,
                **kwargs,
            )
            # axs[ii, jj].semilogx(
            # classic_bins / 1000, np.angle(classic_fourier[i] * classic_fourier_scale).T, alpha=0.5, lw=4, **kwargs)
            # axs[ii, jj].semilogx(self.fourier_freq / 1000, expected_phase, 'k', alpha=0.4, **kwargs)
            axs[ii, jj].set_title(f"Tube {i}")
        axs[0, 0].legend(["ref"] + self.channels)
        axs[0, 0].set_xlabel("Frequency(kHz)")
        axs[0, 0].set_ylabel("Phase (°)")
        # plt.show()
        # Plot the relative calibration -- before smoothing
        fig, axs = plt.subplots(3, 1, sharex=True, sharey=False, **figkwargs)
        fig.suptitle("Relative + Absolute Calibration")
        axs[0].semilogx(self.fourier_freq / 1000, todB(self.rel_cal[..., self.channels]))
        axs[0].set_xlabel("Frequency(kHz)")
        axs[0].set_ylabel("Amplitude (dB SPL uncal)")
        axs[0].set_title("Relative Calibration")
        axs[1].semilogx(
            self.fourier_freq / 1000,
            np.rad2deg(
                np.angle(
                    self.rel_cal[..., self.channels]
                    / self.rel_cal[..., self.channels[:1]]
                    * np.abs(self.rel_cal[..., self.channels[:1]])
                )
            ),
        )
        axs[1].set_xlabel("Frequency(kHz)")
        axs[1].set_ylabel("Phase (°)")
        axs[2].semilogx(self.fourier_freq / 1000, todB(self.abs_cal))
        axs[2].set_xlabel("Frequency(kHz)")
        axs[2].set_ylabel("Amplitude (dB SPL)")
        axs[2].set_title("Absolute Calibration")
        fig.tight_layout()
        # plt.show()

        # Plot the sanity check where we try to recreate one of the mic signals given two others
        mic1 = self.channels[-3]
        mic2 = self.channels[-2]
        mic_ref = self.channels[-1]
        tube = 3
        x_mic = self.probe.get_unit("mic_positions")
        p = self.calibration.cal_p(self.fourier_freq, self.fourier)
        p3_pred = self._cavern_model.p_measured(
            self.fourier_freq * auren.units.Hz, x_mic[mic_ref], x_mic[mic1], x_mic[mic2], p[tube, mic1], p[tube, mic2]
        )

        fig, axs = plt.subplots(2, 1, sharex=True, **figkwargs)
        fig.suptitle("Self-consistency Check of the Calibration")
        axs[0].semilogx(
            self.fourier_freq, todB(p3_pred), "C0", lw=4, alpha=0.5, label="Predicted from Mics %s, %s" % (mic1, mic2)
        )
        axs[0].semilogx(self.fourier_freq, todB(p[tube, mic_ref]), "C1", lw=2, label="Measured, Mic %d" % mic_ref)
        axs[0].legend()
        axs[0].set_ylabel("Amplitude (dB SPL)")
        axs[0].set_xlabel("Frequency (Hz)")
        axs[1].semilogx(self.fourier_freq, todB(p[tube, mic_ref]) - todB(p3_pred), "C2", lw=3)
        axs[1].set_ylabel("Reference - Prediction (Difference of dB)")
        axs[1].set_xlabel("Frequency (Hz)")
        axs[1].set_ylim(-10, 10)
        axs[1].grid()

        # Plot the sanity check where we try to recreate the reference mic from two of the mics
        fig, axs = plt.subplots(2, 2, sharex=True, **figkwargs)
        fig.suptitle("Self-consistency Check of the Calibration with Ref Mic")
        fig2, axs2 = plt.subplots(2, 2, sharex=True, **figkwargs)
        fig2.suptitle("Self-consistency Check of the Calibration with Ref Mic (Diff)")

        fig3, axs3 = plt.subplots(2, 2, sharex=True, **figkwargs)
        fig3.suptitle("Self-consistency Check of the Calibration (Reflectance)")

        fig4, axs4 = plt.subplots(2, 2, sharex=True, **figkwargs)
        fig4.suptitle("Self-consistency Check of the Calibration (Reflected/Fwd Amplitude)")

        for i in range(4):
            mic1 = self.channels[0]
            mic2 = self.channels[-1]
            tube = i
            x_mic = self.probe.get_unit("mic_positions")
            data_id = list(self.raw_data._tones_dict_ref.keys()).index(self.calibration_mic_tone_specs.id)
            tube_obj = self.raw_data.tubes_ref[data_id][tube]
            x_ref_mic = tube_obj.get_unit("probe_locations")[0]
            p = self.calibration.cal_p(self.fourier_freq, self.fourier)
            p3_pred = self._cavern_model.p_measured(
                self.fourier_freq * auren.units.Hz, x_ref_mic, x_mic[mic1], x_mic[mic2], p[tube, mic1], p[tube, mic2]
            )

            k = self._cavern_model.k(self.fourier_freq * auren.units("Hz"))
            a0 = self._cavern_model.A_measured(
                k, x=None, x0=x_mic[mic1], x1=x_mic[mic2], p0=p[tube, mic1, :], p1=p[tube, mic2, :]
            ).magnitude
            b0 = self._cavern_model.B_measured(
                k, x=None, x0=x_mic[mic1], x1=x_mic[mic2], p0=p[tube, mic1, :], p1=p[tube, mic2, :]
            ).magnitude



            ii = i // 2
            jj = i % 2
            axs[ii, jj].semilogx(
                self.fourier_freq, todB(p[tube, mic1]), "C2", lw=1, label=f"Measured, Mic {mic1}", alpha=0.4
            )
            axs[ii, jj].semilogx(
                self.fourier_freq, todB(p[tube, mic2]), "C3", lw=1, label=f"Measured, Mic {mic2}", alpha=0.4
            )
            axs[ii, jj].semilogx(
                self.fourier_freq,
                todB(p3_pred),
                "C0",
                lw=4,
                alpha=0.5,
                label="Predicted from Mics %s, %s" % (mic1, mic2),
            )
            axs[ii, jj].semilogx(self.fourier_freq, todB(self.fourier_ref[tube]), "C1", lw=2, label="Measured, Ref Mic")
            axs[ii, jj].legend()
            axs[ii, jj].set_ylabel("Amplitude (dB SPL)")
            axs[ii, jj].set_xlabel("Frequency (Hz)")
            axs[ii, jj].set_title(f"Tube {i}")
            axs2[ii, jj].semilogx(self.fourier_freq, todB(self.fourier_ref[tube]) - todB(p3_pred), "C2", lw=3)
            axs2[ii, jj].set_ylabel("Reference - Prediction (Difference of dB)")
            axs2[ii, jj].set_xlabel("Frequency (Hz)")
            axs2[ii, jj].set_ylim(-10, 10)
            axs2[ii, jj].grid()
            axs2[ii, jj].set_ylabel("Amplitude (dB SPL)")
            axs2[ii, jj].set_xlabel("Frequency (Hz)")
            axs2[ii, jj].set_title(f"Tube {i}")

            axs3[ii, jj].semilogx(self.fourier_freq, np.abs(b0/a0) *0 + 1, 'k', lw=1)
            axs3[ii, jj].semilogx(self.fourier_freq, np.abs(b0/a0), lw=3)
            axs3[ii, jj].set_ylabel("Reflectance")
            axs3[ii, jj].set_xlabel("Frequency (Hz)")
            axs3[ii, jj].set_ylim(-0.5, 1.5)
            axs3[ii, jj].grid()
            axs3[ii, jj].set_title(f"Tube {i}")

            axs4[ii, jj].semilogx(self.fourier_freq, todB(a0), label='Forward')
            axs4[ii, jj].semilogx(self.fourier_freq, todB(b0), label='Reflected')
            axs4[ii, jj].set_ylabel("Amplitude (dB SPL)")
            axs4[ii, jj].set_xlabel("Frequency (Hz)")
            axs4[ii, jj].legend()
            axs4[ii, jj].grid()
            axs4[ii, jj].set_title(f"Tube {i}")

        if show:
            plt.show()

    ######## PRIVATE METHODS
    def _trim_data(self, data, real_basis, imag_basis, f_at_sample, n_samples, trim_freq):
        # trim_freq = self.calibration_block_size * 8
        i = np.argmin(np.abs(f_at_sample - trim_freq))
        sym_pad = (data.shape[-1] - real_basis[1].shape[0]) // 2

        sub_slice = slice(
            i - int(self.calibration_block_size * 1.5) + sym_pad,
            i + int(self.calibration_block_size * 1.5) + 1 + sym_pad,
        )
        sub_slice2 = slice(i - self.calibration_block_size // 2, i + self.calibration_block_size // 2 + 1)
        subdata = data[..., sub_slice]
        r_base = real_basis[1][sub_slice2, 0]
        i_base = imag_basis[1][sub_slice2, 0]

        real_c = ndimage.convolve1d(subdata, r_base[::-1], axis=-1)
        imag_c = ndimage.convolve1d(subdata, i_base[::-1], axis=-1)
        # tube = 3
        mag_c = np.sqrt(
            real_c ** 2 + imag_c ** 2
        )  # [..., self.calibration_block_size // 2: self.calibration_block_size // 2 + self.calibration_block_size]
        # t = np.arange(mag_c.shape[-1]) - mag_c.shape[-1] // 2
        # plt.plot(real_c[tube, 0])
        # plt.plot(imag_c[tube, 0])
        # plt.plot(mag_c[tube, 0])
        # plt.show()
        # plt.plot(t, mag_c[tube].T)
        # plt.show()
        offset = np.argmax(mag_c, axis=-1) - mag_c.shape[-1] // 2
        offset = np.round(offset.mean(axis=1)).astype(int)  # Average over the channels

        new_shape = list(data.shape)
        new_shape[-1] = n_samples
        new_data = np.zeros(new_shape)
        slices = []
        for i, off in enumerate(offset):
            slices.append(slice(sym_pad + off, sym_pad + off + n_samples))
            new_data[i] = data[i, ..., slices[-1]]

        # Check
        # tube = 0
        # chan = 0
        # print("This: ",
        #     np.sqrt((new_data[tube, chan, sub_slice2] * r_base).sum() ** 2 + (new_data[tube, chan, sub_slice2] * i_base).sum() ** 2),
        #     " should roughly equal: ",
        #     mag_c[tube, chan, offset[tube] + mag_c.shape[-1] // 2]
        # )

        return new_data, slices

    def _fourier_transform_raw_data(self, data_raw, real_basis, imag_basis, f_at_sample, n_samples):
        # The Tympan firmware adds some padding before and after playing the .wav file ON TOP of our
        # padding for the ramp-up period. Thankfully, both are symmetric, so we can just figure out
        # the padding based on the difference in size between the recorded and expected signals

        pad_samples = (data_raw.shape[-1] - real_basis[1].shape[0]) // 2
        raw_clipped = data_raw[..., pad_samples : pad_samples + n_samples]
        f_centers, fourier = dft_known_basis(
            raw_clipped,
            f_at_sample,
            real_basis[1][..., 0],
            imag_basis[1][..., 0],
            block_size=self.calibration_block_size,
            n_regions=self.number_of_calibration_frequency_bins * self.bin_oversampling,
        )
        return f_centers, fourier

    def _fourier_transform_ref_data(self, data_ref, real_basis, imag_basis, f_at_sample, n_samples):
        pad_samples = (data_ref.shape[-1] - real_basis[1].shape[0]) // 2

        if self.calibration_ref_start_time_unknown:
            # Find the max amplitude convolution for the first block
            real_block = real_basis[1][..., 0][-(self.calibration_block_size * 2 + 1) :]
            imag_block = imag_basis[1][..., 0][-(self.calibration_block_size * 2 + 1) :]
            raw_clipped = []
            for i in range(data_ref.shape[0]):
                real = np.convolve(data_ref[i], real_block[::-1], "same")
                imag = np.convolve(data_ref[i], imag_block[::-1], "same")
                # ind = np.argmax(real - np.abs(imag))  # 0 phase mean 0 imaginary part, max amplitude on real
                ind = max([n_samples, np.argmax(real ** 2 + imag ** 2)])  # max amplitude agreement
                raw_clipped.append(data_ref[i, ind - n_samples : ind])
            raw_clipped = np.stack(raw_clipped, axis=0)
        else:
            raw_clipped = data_ref[..., pad_samples : pad_samples + n_samples]

        f_centers_ref, fourier_ref = dft_known_basis(
            raw_clipped,
            f_at_sample,
            real_basis[1][..., 0],
            imag_basis[1][..., 0],
            block_size=self.calibration_block_size,
            n_regions=self.number_of_calibration_frequency_bins * self.bin_oversampling,
        )
        return f_centers_ref, fourier_ref

    def _relative_calibration(self, f, p, probe, channels=None):
        if channels is None:
            n_channels = p.shape[-2]
            channels = np.arange(n_channels)

        n_channels = len(channels)
        k = self.k(f)

        p = p.reshape(-1, p.shape[-2], p.shape[-1])
        n_caverns = p.shape[0]

        x = probe.get_unit("mic_positions").to("m").magnitude

        A = np.zeros((p.shape[-1], n_channels * n_caverns, 2 * n_caverns + n_channels), dtype=np.complex128)
        for j in range(n_caverns):
            A[:, j * n_channels : (j + 1) * n_channels, j * 2 + 0] = np.exp(-1j * k[:, None] * x[None, channels])
            A[:, j * n_channels : (j + 1) * n_channels, j * 2 + 1] = np.exp(1j * k[:, None] * x[None, channels])
            for i in range(n_channels):
                A[:, j * n_channels + i, -n_channels + i] = -p[j, channels[i]]

        u, s, vh = np.linalg.svd(A)
        rel_cal = vh[:, -1, :].conjugate()  # Maybe taking conjugate is important?
        rel_cal_ordered = np.ones((p.shape[-1], p.shape[-2]), rel_cal.dtype)
        rel_cal_ordered[:, channels] = rel_cal[:, -n_channels:]

        # Just accept fate... and realize that phase below 1kHz is too noisy to calibrate
        if self.noise_freq_threshold is not None:
            rel_cal_ordered[f <= self.noise_freq_threshold] = rel_cal_ordered[f > self.noise_freq_threshold][0]

        return rel_cal_ordered

    def _absolute_calibration(self, f, p, p_ref, rel_cal, probe, tubes, channels=None):
        if channels is None:
            n_channels = p.shape[-2]
            channels = np.arange(n_channels)

        n_channels = len(channels)

        k = self.k(f)

        p = p.reshape(-1, p.shape[-2], p.shape[-1])
        p_ref = p_ref.reshape(-1, 1, p_ref.shape[-1])
        n_caverns = p.shape[0]
        # n_caverns = 1

        x = probe.get_unit("mic_positions").to("m").magnitude[channels]
        x_ref_mic = [tube.get_unit("probe_locations")[0].to("m").magnitude for tube in tubes]

        A = np.zeros((p.shape[-1], (n_channels + 1) * n_caverns, 2 * n_caverns + 1), dtype=np.complex128)
        b = np.zeros((p.shape[-1], (n_channels + 1) * n_caverns), dtype=np.complex128)
        for j in range(n_caverns):
            A[:, j * (n_channels + 1) : (j + 1) * (n_channels + 1) - 1, j * 2 + 0] = np.exp(
                -1j * k[:, None] * x[None, :]
            )
            A[:, j * (n_channels + 1) : (j + 1) * (n_channels + 1) - 1, j * 2 + 1] = np.exp(
                1j * k[:, None] * x[None, :]
            )
            A[:, (j + 1) * (n_channels + 1) - 1, j * 2 + 0] = np.exp(-1j * k * x_ref_mic[j])
            A[:, (j + 1) * (n_channels + 1) - 1, j * 2 + 1] = np.exp(1j * k * x_ref_mic[j])
            for i in range(n_channels):
                A[:, j * (n_channels + 1) + i, -1] = -p[j, channels[i]] * rel_cal[:, channels[i]]
            b[:, (j + 1) * (n_channels + 1) - 1] = np.abs(p_ref[j])
            # b[:, (j + 1) * (n_channels + 1) - 1] = (p_ref[j])  # This is worse

        Ainv = np.linalg.pinv(A)

        sol = np.einsum("ijk,ik->ij", Ainv, b)
        # amp_cal = (sol[:, -1])   # This is worse
        amp_cal = np.abs(sol[:, -1])

        return amp_cal

    def _combine_smooth_cal(self, f, rel_cal, abs_cal, channels, smoothing_factor):
        n_channels = rel_cal.shape[1]

        phase = np.angle(rel_cal.T * abs_cal).T
        amp = np.abs(rel_cal)

        mean_phase = np.nanmean(np.angle(rel_cal[..., channels]), -1, keepdims=True)
        phase[..., channels] -= mean_phase  # Phases are relative, so we say that the average phase is 0
        mean_amp = np.nanmean(np.abs(rel_cal[..., channels]), -1, keepdims=True)
        amp[..., channels] -= mean_amp

        high_freq = max(self.calibration_mic_tone_specs.frequencies)
        low_freq = min(self.calibration_mic_tone_specs.frequencies)

        N = self.number_of_calibration_frequency_bins + 1
        f_bins = np.logspace(np.log10(low_freq), np.log10(high_freq), N)
        f_center = (f_bins[1:] + f_bins[:-1]) / 2
        # average phase and amplitude into these bins
        cal_phase_bin = np.zeros((N - 1, n_channels))
        cal_amp_bin = np.zeros((N - 1, n_channels))
        # cal_phase_bin_mean = np.zeros((N - 1, n_channels))
        cal_amp_bin_mean = np.zeros((N - 1, n_channels))
        cal_abs_amp_bin = np.zeros((N - 1, n_channels))
        for i in range(f_center.size):
            I = (f >= f_bins[i]) & (f < f_bins[i + 1])
            cal_phase_bin[i] = np.nanmean(phase[I], axis=0)
            cal_amp_bin[i] = np.nanmean(amp[I], axis=0)
            # cal_phase_bin_mean[i] = np.mean(mean_phase[I], axis=0)
            cal_amp_bin_mean[i] = np.nanmean(mean_amp[I], axis=0)
            cal_abs_amp_bin[i] = np.nanmean(np.abs(abs_cal[I]), axis=0)

        # Add the mean back in
        cal_amp_bin += cal_amp_bin_mean
        # cal_phase_bin += cal_phase_bin_mean

        # Smooth the results
        sigma = smoothing_factor
        cal_phase_bin_smooth = ndimage.gaussian_filter1d(cal_phase_bin, sigma, mode="nearest", axis=0, truncate=2.5)
        cal_amp_bin_smooth = ndimage.gaussian_filter1d(cal_amp_bin, sigma, mode="nearest", axis=0, truncate=2.5)
        sigma = smoothing_factor / 2
        cal_abs_amp_bin_max = ndimage.maximum_filter1d(cal_abs_amp_bin, self.bin_oversampling, axis=0)
        cal_abs_amp_bin_smooth = ndimage.gaussian_filter1d(
            cal_abs_amp_bin_max, sigma / 2, mode="nearest", axis=0, truncate=2.5
        )

        # Just accept fate... and realize that phase below 1kHz is too noisy to calibrate
        # if self.noise_freq_threshold is not None:
        #     cal_phase_bin_smooth[f_center <= self.noise_freq_threshold] = cal_phase_bin_smooth[
        #         f_center > self.noise_freq_threshold
        #     ][0]
        #     cal_amp_bin_smooth[f_center <= self.noise_freq_threshold] = cal_amp_bin_smooth[
        #         f_center > self.noise_freq_threshold
        #     ][0]

        # Finish up the calibration -- combine the results
        cal = np.zeros((n_channels, N - 1, 3))
        cal[..., 0] = f_center
        cal[..., 1] = cal_amp_bin_smooth.T * cal_abs_amp_bin_smooth.T
        cal[..., 2] = cal_phase_bin_smooth.T
        # No smoothing or averaging cal -- for debugging/consistency checks
        # cal = np.zeros((n_channels, f.size, 3))
        # cal[..., 0] = f
        # cal[..., 1] = np.abs(rel_cal.T * abs_cal)
        # cal[..., 2] = np.angle(rel_cal.T * abs_cal)

        return cal

    def _calibrate_speakers(self, calibration=None):
        if calibration is None:
            calibration = self.calibration
        speaker_cal = np.zeros((len(self.calibration_speaker_tone_specs), self.number_of_calibration_frequency_bins, 2))
        for i in range(2):
            real_basis = self.calibration_speaker_tone_specs[i].get()
            imag_basis = self.calibration_speaker_tone_specs[i].get(phi=np.pi / 2)
            f_at_sample = self.calibration_speaker_tone_specs[i].get(return_freq=True)[2]

            chnls = self.calibration_speaker_tone_specs[i].channels
            real_basis[1] = real_basis[1][:, chnls]
            imag_basis[1] = imag_basis[1][:, chnls]

            n_samples = real_basis[0].shape[-1]
            data_id = list(self.raw_data._tones_dict.keys()).index(self.calibration_speaker_tone_specs[i].id)
            # Trim the data
            trimmed_data, _ = self._trim_data(
                self.raw_data.data[data_id],
                real_basis,
                imag_basis,
                f_at_sample,
                n_samples,
                trim_freq=self.trim_frequency,
            )
            f_centers, fourier = self._fourier_transform_raw_data(
                trimmed_data,
                real_basis,
                imag_basis,
                f_at_sample,
                n_samples,
            )

            p_cal = calibration.cal_p(f_centers, fourier) * auren.units("Pa")
            mic1 = self.channels[0]
            mic2 = self.channels[-1]
            k = self._cavern_model.k(f_centers * auren.units("Hz"))
            x_mic = self.probe.get_unit("mic_positions")
            a0 = self._cavern_model.A_measured(
                k, x=None, x0=x_mic[mic1], x1=x_mic[mic2], p0=p_cal[..., mic1, :], p1=p_cal[..., mic2, :]
            ).magnitude
            b0 = self._cavern_model.B_measured(
                k, x=None, x0=x_mic[mic1], x1=x_mic[mic2], p0=p_cal[..., mic1, :], p1=p_cal[..., mic2, :]
            ).magnitude

            # Desired A0
            db_cal_frac = 10 ** (self.calibration_tone_db / 20)
            a0_desired_pa = 10 ** (self.speaker_output_db / 20) * 20e-6
            # pa_desired = cal_frac * pa_actual / db_cal_frac
            # cal_frac = pa_desired / pa_actual * db_cal_frac
            cal_frac = a0_desired_pa / np.abs(a0) * db_cal_frac
            cal_frac_smooth = self._combine_smooth_cal(
                f_centers, cal_frac.T, cal_frac.squeeze() * 0 + 1 + 0j, [0], self.calibration_smoothing_sigma_speaker
            )
            speaker_cal[i] = cal_frac_smooth[0, :, :2]

            # plt.semilogx(f_centers, 20 * np.log10(np.abs(a0) / 20e-6).T, label="A0");
            # plt.semilogx(f_centers, 20 * np.log10(np.abs(b0) / 20e-6).T, '--', label="B0")
            # plt.semilogx(f_centers, 20 * np.log10(np.abs(a0 + b0) / 20e-6).T, ':', label="A0 + B0");
            # plt.legend()
            # plt.show()
            # plt.semilogx(f_centers, np.abs(b0 / a0).T); plt.ylim(0, 2);
            # plt.show()
            # plt.semilogx(f_centers, 20 * np.log10(np.abs(p_cal[0].magnitude) / 20e-6).T);
            # plt.semilogx(f_centers, 20 * np.log10(np.abs(fourier_ref) / 20e-6).T, '--');
            # plt.show()

        return speaker_cal

