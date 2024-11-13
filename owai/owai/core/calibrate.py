import typing as t
from pydantic import BaseModel
from numpydantic import NDArray, Shape
import numpy as np
from scipy import ndimage
import os

from owai.core import model
import owai.core.data_models as dm
from owai.core.tympan_serial import TympanSerial
from owai.core.signal_processing_utils import pad_chirp, dft_known_basis
from owai.core import io
import owai

class Calibrate(BaseModel):

    ### INPUTS section ###

    raw_data :  t.Optional[dm.calibration.RawCalibrationData] = None
    calibration_mic_tone_specs : dm.Chirp = dm.Chirp(
        durations=[5, 5],
        frequencies=[80, 1000, 21000],
        samplerate=96000,
        channels=[True, True])
    calibration_speaker_tone_specs : t.List[dm.Chirp] = [
        dm.Chirp(
            durations=[5],
            frequencies=[80, 21000],
            channels=[True, False],
            samplerate=96000),
        dm.Chirp(
            durations=[5],
            frequencies=[80, 21000],
            samplerate=96000,
            channels=[False, True]),
    ]
    calibration_pad_time : float = 0.25  # seconds
    calibration_ref_start_time_unknown : bool = True
    calibration_data_path : str = "calibration_data"
    calibration_tone_db : float = -10  # dB FS

    calibration_block_size : int = 2048
    number_of_calibration_frequency_bins : int = 256
    bin_oversampling : int = 5
    calibration_smoothing_sigma : float = 9
    # Below this frequency, we don't calibrate. We just assume the phase and relative mic amplitudes match
    # This is implemented in _combine_smooth_cal
    noise_freq_threshold : float = 100

    probe : dm.Probe = dm.Probe()

    sound_speed : float = 343 # m/s

    ref_mic_sensitivity : float = 6.3 # mv/Pa
    tympan_full_scale_voltage : float = 1 # Volts


    ### OUTPUTS Section ###

    calibration : dm.CalibrationData = None

    # Add ALL the calibration information for debugging and archeology
    # rel_cal :  t.Optional[NDArray[Shape["* frequencies, * channels"], complex]] = None
    abs_cal :  t.Optional[NDArray[Shape["* frequencies"], float]] = None
    # p_cal :  t.Optional[NDArray[Shape["* tubes, * channels, * frequencies"], complex]] = None
    cal :  t.Optional[NDArray[Shape["* channels, * frequencies, 3 freqAmpPhase"], float]] = None
    # fourier:  t.Optional[NDArray[Shape["* tubes, * channels, * frequencies"], complex]] = None
    # fourier_ref:  t.Optional[NDArray[Shape["* tubes, * frequencies"], complex]] = None
    # fourier_freq:  t.Optional[NDArray[Shape["* frequencies"], complex]] = None
    real_basis:  t.Optional[t.List[NDArray]] = None
    imag_basis:  t.Optional[t.List[NDArray]] = None
    f_at_sample:  t.Optional[NDArray[Shape["* frequencies"], float]] = None


    # Private/working attributes, not part of the data model but needed for the class to function
    _cavern_model : model.Model = model.StraightTube(0, 0, 0, 0, 0)
    _calibration_tones = None
    _calibration_tones_speaker = None
    _tympan = None
    _start_tone_played = False

    out_path : str = "."

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
            self._tympan.send_char('1')
            self._start_tone_played = True
            print(self._tympan.read_all(5))
        print(self._tympan.read_all(0.5))

        # For this data, we will record the filename, the tube, and the chirp, and add it to a list
        if self.raw_data is None:
            self.raw_data = dm.RawCalibrationData(file_meta_data=[], file_meta_data_ref=[])

        # select the correct channels before playing
        self._tympan.send_char('E')
        print(self._tympan.read_all(1))
        tone = self.calibration_mic_tone_specs
        # Clear the buffer
        print(self._tympan.read_all(1))
        print ("playing tone ", tone)
        # send the play/record command
        self._tympan.send_char('4')
        # Get the reply
        reply = ''
        count = 0
        while "Auto-stopping SD recording of AUDIO" not in reply:
            reply += self._tympan.read_all(0.5)
            count += 1
            if count > np.sum(tone.durations) / 0.5 * 2:
                break  # avoid infinite loops, allow twice the time

        try:
            name = "AUDIO" + reply.split("AUDIO")[1].split('.')[0] + ".WAV"
            print ("Success: ", reply)
        except IndexError:
            name = ["[error]"]
            print ("Error: ", reply)

        print("Results saved in ", name)

        spec = dm.FileMetaData(
            tube=tube,
            tone=tone,
            name=os.path.join(self.out_path, self.calibration_data_path, name)
        )
        self.raw_data.file_meta_data.append(spec)

        # select the correct channels before playing
        self._tympan.send_char('o')
        print(self._tympan.read_all(1))
        # Clear the buffer
        print(self._tympan.read_all(4))
        print ("playing tone ", tone)
        # send the play/record command
        self._tympan.send_char('4')
        # Get the reply
        reply = self._tympan.read_all(0.5)
        count = 0
        while "Auto-stopping SD recording of AUDIO" not in reply:
            reply += self._tympan.read_all(0.5)
            count += 1
            if count > np.sum(tone.durations) / 0.5 * 2:
                break  # avoid infinite loops, allow twice the time

        try:
            name = "AUDIO" + reply.split("AUDIO")[1].split('.')[0] + ".WAV"
            print ("Success: ", reply)

        except IndexError:
            print ("Error: ", reply)

        print("Results saved in ", name)

        spec = dm.FileMetaData(
            tube=tube,
            tone=tone,
            name=os.path.join(self.out_path, self.calibration_data_path, name)
        )
        self.raw_data.file_meta_data_ref.append(spec)

    def collect_speaker_calibration_data(self, tube):
        if not self._start_tone_played:
            self._tympan.send_char('1')
            self._start_tone_played = True
            print(self._tympan.read_all(5))
        print(self._tympan.read_all(0.5))

        # For this data, we will record the filename, the tube, and the chirp, and add it to a list
        if self.raw_data is None:
            self.raw_data = dm.RawCalibrationData(file_meta_data=[], file_meta_data_ref=[])

        # select the correct channels before playing and clear the buffer
        self._tympan.send_char('E')
        print(self._tympan.read_all(1))
        for i, tone in enumerate(self.calibration_speaker_tone_specs):
            # Clear the buffer
            print(self._tympan.read_all(1))
            print ("playing tone ", tone)
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
                name = "AUDIO" + reply.split("AUDIO")[1].split('.')[0] + ".WAV"
            except IndexError:
                print ("Error: ", reply)

            print("Results saved in ", name)

            spec = dm.FileMetaData(
                tube=tube,
                tone=tone,
                name=os.path.join(self.out_path, self.calibration_data_path, name)
            )
            self.raw_data.file_meta_data.append(spec)


    def make_cal_chirps(self) -> t.List[NDArray]:
        amp = 10 ** (self.calibration_tone_db / 20)
        tones = [
            self.calibration_mic_tone_specs.get(pad_time=self.calibration_pad_time)[1][:, None].repeat(2, axis=1) * amp
        ]
        speaker_tones = [csts.get(pad_time=self.calibration_pad_time)[1][:, None].repeat(2, axis=1) * amp for csts in self.calibration_speaker_tone_specs]
        speaker_tones[0][:, 1] = 0
        speaker_tones[1][:, 0] = 0

        self._calibration_tones = tones
        self._calibration_tones_speaker = speaker_tones
        return tones + speaker_tones

    def save_cal_tones(self, subpath='calibration_tones'):
        if self._calibration_tones is None:
            self.make_cal_chirps()
        os.makedirs(os.path.join(self.out_path, subpath), exist_ok=True)
        for i, cts in enumerate(
            zip(
                self._calibration_tones + self._calibration_tones_speaker,
               [self.calibration_mic_tone_specs] + self.calibration_speaker_tone_specs)):
            path = os.path.join(self.out_path, subpath, "PLAY{}.WAV".format(i + 1))
            io.write_wav(path, cts[0], cts[1].samplerate, np.int16)

    def calibrate(self):
        # Step 1 -- time-domain to frequency domain
        # 1a, make the basis functions
        real_basis = self.calibration_mic_tone_specs.get()
        imag_basis = self.calibration_mic_tone_specs.get(phi=np.pi/2)
        f_at_sample = self.calibration_mic_tone_specs.get(return_freq=True)[2]

        n_samples = real_basis[0].shape[-1]

        # Step 2: Fourier transform the raw data
        data_id = list(self.raw_data._tones_dict.keys()).index(self.calibration_mic_tone_specs.id)
        f_centers, fourier = self._fourier_transform_raw_data(
            self.raw_data.data[data_id],
            real_basis,
            imag_basis,
            f_at_sample,
            n_samples,
        )

        # Step 3: Fourier transform the reference data
        data_id = list(self.raw_data._tones_dict_ref.keys()).index(self.calibration_mic_tone_specs.id)
        data_ref = self.raw_data.data_ref[data_id][:, 0] * self.ref_mic_sensitivity * self.tympan_full_scale_voltage
        f_centers_ref, fourier_ref = self._fourier_transform_ref_data(
            data_ref,
            real_basis,
            imag_basis,
            f_at_sample,
            n_samples,
        )

        # Step 4: Complete the relative calibration
        rel_cal = self._relative_calibration(f_centers, fourier, self.probe)

        # Step 5: Complete the absolute calibration
        abs_cal = self._absolute_calibration(f_centers, fourier, fourier_ref, rel_cal, self.probe)

        # Step 6: Combine and smooth the calibration
        cal = self._combine_smooth_cal(f_centers, rel_cal, abs_cal)

        # Step 7: Create the summarized calibration objects for the mics and start populating the full calibration object
        mics = [dm.MicCalibration(cal=c, channel=i) for i, c in enumerate(cal)]
        cal_obj = dm.CalibrationData(mic=mics)

        # Calibrate the pressure for the calibration data (do to self-consistency test)
        p_cal = cal_obj.cal_p(f_centers, fourier) * owai.units('Pa')

        # Save ALL the calibration information
        self.calibration = cal_obj
        # self.rel_cal = rel_cal
        self.abs_cal = abs_cal
        self.cal = cal
        # self.p_cal = p_cal
        # self.fourier = fourier
        # self.fourier_ref = fourier_ref
        # self.fourier_freq = f_centers
        self.imag_basis = imag_basis
        self.f_at_sample = f_at_sample

        # Step 8: Calibrate the speakers

        mic1 = 1
        mic2 = 3
        k = self._cavern_model.k(f_centers * owai.units("Hz"))
        x_mic = self.probe.get_unit("mic_positions")
        a0 = self._cavern_model.A_measured(
            k, x=None,
            x0=x_mic[mic1], x1=x_mic[mic2],
            p0=p_cal[..., mic1, :],
            p1=p_cal[..., mic2, :]).magnitude
        b0 = self._cavern_model.B_measured(
            k, x=None,
            x0=x_mic[mic1], x1=x_mic[mic2],
            p0=p_cal[..., mic1, :],
            p1=p_cal[..., mic2, :]).magnitude

        # stop
        # plt.semilogx(f_centers, 20 * np.log10(np.abs(a0) / 20e-6).T);
        # plt.semilogx(f_centers, 20 * np.log10(np.abs(b0) / 20e-6).T, '--')
        # plt.semilogx(f_centers, 20 * np.log10(np.abs(a0 + b0) / 20e-6).T, ':'); plt.show()
        # plt.semilogx(f_centers, np.abs(b0 / a0).T); plt.ylim(0, 2); plt.show()
        # plt.semilogx(f_centers, 20 * np.log10(np.abs(p_cal[0].magnitude) / 20e-6).T);
        # plt.semilogx(f_centers, 20 * np.log10(np.abs(fourier_ref) / 20e-6).T, '--'); plt.show()





    ######## PLOT FUNCTIONS
    def plot_calibration_data(self, show=False, **kwargs):
        self.raw_data.plot(**kwargs)

    ######## PRIVATE METHODS


    def _fourier_transform_raw_data(self, data_raw, real_basis, imag_basis, f_at_sample, n_samples):
        pad_samples = int(np.round(self.calibration_pad_time * self.raw_data.samplerate))
        raw_clipped = data_raw[..., pad_samples:pad_samples + n_samples]
        f_centers, fourier = dft_known_basis(
            raw_clipped,
            f_at_sample,
            real_basis[1][..., 0],
            imag_basis[1][..., 0],
            block_size=self.calibration_block_size,
            n_regions=self.number_of_calibration_frequency_bins * self.bin_oversampling
        )
        return f_centers, fourier

    def _fourier_transform_ref_data(self, data_ref, real_basis, imag_basis, f_at_sample, n_samples):
        pad_samples = int(np.round(self.calibration_pad_time * self.raw_data.samplerate))

        if self.calibration_ref_start_time_unknown:
            # Find the max amplitude convolution for the first block
            real_block = real_basis[1][..., 0][-(self.calibration_block_size*2 + 1):]
            imag_block = imag_basis[1][..., 0][-(self.calibration_block_size*2 + 1):]
            raw_clipped = []
            for i in range(data_ref.shape[0]):
                real = np.convolve(data_ref[i], real_block[::-1], "same")
                imag = np.convolve(data_ref[i], imag_block[::-1], "same")
                # ind = np.argmax(real - np.abs(imag))  # 0 phase mean 0 imaginary part, max amplitude on real
                ind = max([n_samples, np.argmax(real**2 + imag*2)])  # max amplitude agreement
                raw_clipped.append(data_ref[i, ind - n_samples:ind])
            raw_clipped = np.stack(raw_clipped, axis=0)
        else:
            raw_clipped = data_ref[..., pad_samples:pad_samples + n_samples]

        f_centers_ref, fourier_ref = dft_known_basis(
            raw_clipped,
            f_at_sample,
            real_basis[1][..., 0],
            imag_basis[1][..., 0],
            block_size=self.calibration_block_size,
            n_regions=self.number_of_calibration_frequency_bins * self.bin_oversampling
        )
        return f_centers_ref, fourier_ref

    def _relative_calibration(self, f, p, probe, channels=None):
        if channels is None:
            n_channels = p.shape[-2]
            channels = np.arange(n_channels)

        n_channels = len(channels)
        k = self.k(f)

        p  = p.reshape(-1, p.shape[-2], p.shape[-1])
        n_caverns = p.shape[0]

        x = probe.get_unit("mic_positions").to('m').magnitude

        A = np.zeros((p.shape[-1], n_channels * n_caverns, 2 * n_caverns + n_channels), dtype=np.complex128)
        for j in range(n_caverns):
            A[:, j*n_channels:(j+1) * n_channels, j*2 + 0] = np.exp(-1j * k[:, None] * x[None, channels])
            A[:, j*n_channels:(j+1) * n_channels, j*2 + 1] = np.exp( 1j * k[:, None] * x[None, channels])
            for i in range(n_channels):
                A[:, j*n_channels + i, -n_channels + i] = -p[j, channels[i]]

        u, s, vh = np.linalg.svd(A)
        rel_cal = vh[:, -1, :].conjugate()  # Maybe taking conjugate is important?
        rel_cal_ordered = np.ones((p.shape[-1], p.shape[-2]), rel_cal.dtype)
        rel_cal_ordered[:, channels] = rel_cal[:, -n_channels:]
        return rel_cal_ordered

    def _absolute_calibration(self, f, p, p_ref, rel_cal, probe, channels=None):
        if channels is None:
            n_channels = p.shape[-2]
            channels = np.arange(n_channels)

        n_channels = len(channels)

        k = self.k(f)

        p  = p.reshape(-1, p.shape[-2], p.shape[-1])
        p_ref = p_ref.reshape(-1, 1, p_ref.shape[-1])
        n_caverns = p.shape[0]

        x = x = probe.get_unit("mic_positions").to('m').magnitude

        A = np.zeros((p.shape[-1], (n_channels) * n_caverns, 2 * n_caverns + 1), dtype=np.complex128)
        b = np.zeros((p.shape[-1], (n_channels) * n_caverns), dtype=np.complex128)
        for j in range(n_caverns):
            A[:, j*n_channels:(j+1) * n_channels, j*2 + 0] = np.exp(-1j * k[:, None] * x[None, :])
            A[:, j*n_channels:(j+1) * n_channels, j*2 + 1] = np.exp( 1j * k[:, None] * x[None, :])
            for i in range(n_channels):
                A[:, j*n_channels + i, -1] = -p[j, channels[i]] * rel_cal[:, channels[i]]
            b[:, (j + 1) * n_channels - 1] = p_ref[j]

        Ainv = np.linalg.pinv(A)

        sol = np.einsum('ijk,ik->ij', Ainv, b)
        # amp_cal2 = (sol[:, -1])
        amp_cal = np.abs(sol[:, -1])
        return amp_cal

    def _combine_smooth_cal(self, f, rel_cal, abs_cal):
        n_channels = rel_cal.shape[1]

        phase = np.angle(rel_cal)
        amp = np.abs(rel_cal)

        mean_phase = np.angle(rel_cal.mean(-1, keepdims=True))
        phase -= mean_phase
        mean_amp = np.abs(rel_cal.mean(-1, keepdims=True))
        amp -= mean_amp

        high_freq = max(self.calibration_mic_tone_specs.frequencies)
        low_freq = min(self.calibration_mic_tone_specs.frequencies)

        N = self.number_of_calibration_frequency_bins + 1
        f_bins = np.logspace(np.log10(low_freq), np.log10(high_freq), N)
        f_center = (f_bins[1:] + f_bins[:-1]) / 2
        # average phase and amplitude into these bins
        cal_phase_bin = np.zeros((N-1, n_channels))
        cal_amp_bin = np.zeros((N-1, n_channels))
        cal_phase_bin_mean = np.zeros((N-1, n_channels))
        cal_amp_bin_mean = np.zeros((N-1, n_channels))
        cal_abs_amp_bin = np.zeros((N-1, n_channels))
        for i in range(f_center.size):
            I = (f >= f_bins[i]) & (f < f_bins[i + 1])
            cal_phase_bin[i] = np.mean(phase[I], axis=0)
            cal_amp_bin[i] = np.mean(amp[I], axis=0)
            cal_phase_bin_mean[i] = np.mean(mean_phase[I], axis=0)
            cal_amp_bin_mean[i] = np.mean(mean_amp[I], axis=0)
            cal_abs_amp_bin[i] = np.mean(abs_cal[I], axis=0)


        # Smooth the results
        sigma = self.calibration_smoothing_sigma
        cal_phase_bin_smooth = ndimage.gaussian_filter1d(cal_phase_bin, sigma, mode="nearest", axis=0, truncate=2.5)
        cal_amp_bin_smooth = ndimage.gaussian_filter1d(cal_amp_bin, sigma, mode="nearest", axis=0, truncate=2.5)
        sigma = self.calibration_smoothing_sigma / 2
        cal_abs_amp_bin_max = ndimage.maximum_filter1d(cal_abs_amp_bin, self.bin_oversampling, axis=0)
        cal_abs_amp_bin_smooth = ndimage.gaussian_filter1d(cal_abs_amp_bin_max, sigma / 2, mode="nearest", axis=0, truncate=2.5)

        # Just accept fate... and realize that phase below 1kHz is too noisy to calibrate
        if self.noise_freq_threshold is not None:
            cal_phase_bin_smooth[f_center <= self.noise_freq_threshold] = 0
            cal_amp_bin_smooth[f_center <= self.noise_freq_threshold] = cal_amp_bin_smooth[f_center > self.noise_freq_threshold][0]

        # Finish up the calibration -- combine the results
        cal = np.zeros((n_channels, N-1, 3))
        cal[..., 0] = f_center
        cal[..., 1] = cal_amp_bin_smooth.T * cal_abs_amp_bin_smooth.T
        cal[..., 2] = cal_phase_bin_smooth.T

        return cal





