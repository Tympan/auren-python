""" Function related to Distortion Produce Oto-Acoustic Emissions -- swept version """

from typing import List, Tuple, Any, Optional
import numpy as np
from scipy import ndimage
from pydantic import BaseModel
import matplotlib.pyplot as plt

from auren.core.data_models import calibration
from auren.core.data_models import chirp
from auren.core import model
from auren.core.utils import todB
from auren.core.signal_processing_utils import dft_known_basis, dft_known_basis_noise, trim_signal_max_correlation_at_f

import auren


class DPOAE(BaseModel):
    calibration: calibration.CalibrationData
    tones: Tuple[chirp.Chirp, chirp.Chirp]
    levels: Tuple[float, float] = [65, 60]
    pad_time: float = 0.25
    mic_channels_used: Tuple[int, int] = [1, 3]
    units: str = "mm"
    ear_diameter: float = 3.75 * 2

    power_reflectance: Optional[Any] = None
    impedance: Optional[Any] = None
    f1: Optional[Any] = None
    f2: Optional[Any] = None
    p_F1_cal: Optional[Any] = None
    p_F2_cal: Optional[Any] = None
    p_2F1_F2_cal: Optional[Any] = None
    p_2F2_F1_cal: Optional[Any] = None
    p_noise: Optional[Any] = None

    _cavern_model: model.Model = None

    @property
    def cavern_model(self):
        if self._cavern_model is None:
            mic0 = self.calibration.mic[self.mic_channels_used[0]]
            mic1 = self.calibration.mic[self.mic_channels_used[1]]
            self._cavern_model = model.TwoDiameterTube(
                x0=mic0.distance * auren.units(mic0.distance_units),
                x1=mic1.distance * auren.units(mic1.distance_units),
                xp=self.calibration.probe.get_unit("length_with_tip"),
                D0=self.calibration.probe.get_unit("interior_diameter"),
                D1=self.ear_diameter * auren.units(self.units)
            )
        return self._cavern_model

    def make_tone(self, phi: float=0) -> np.ndarray:
        """Generates a tone wav file for the Auren speakers based on a calibration and desired chirp characteristics

        Parameters
        ----------
        phi : float, optional
            The phase of the chirp at the start, by default 0

        Returns
        -------
        np.ndarray(shape=(*, len(tones))
            The chirp waveforms
        """
        # shorter names
        pad_time = self.pad_time
        tones = self.tones
        levels = self.levels
        calibration = self.calibration

        # Create the full-scaled amplitude functions fs frequency for each speaker
        amp_funcs = [calibration.cal_speaker(level)[i] for i,level in enumerate(levels)]
        # Generate the chirps, where each speaker tries to produce the full-scale sound
        chirps = [
            tone.get(phi, amplitude_func=amp_func, pad_time=pad_time, return_freq=True)
            for amp_func, tone in zip(amp_funcs, tones)
        ]
        chirp_channels = np.stack([c[1] for c in chirps], axis=1)

        return chirp_channels

    def analyze_iteration(self, signal):
        _, real_basis0, f_at_sample0 = self.tones[0].get(return_freq=True)
        _, imag_basis0 = self.tones[0].get(phi=np.pi / 2)
        _, real_basis1, f_at_sample1 = self.tones[1].get(return_freq=True)
        _, imag_basis1 = self.tones[1].get(phi=np.pi / 2)

        # Make expected tone
        ex_tone0 = self.tones[0].model_copy()
        ex_tone1 = self.tones[0].model_copy()
        ex_tone0.frequencies = [(2 * f1 - f2) for f1, f2 in zip(self.tones[0].frequencies, self.tones[1].frequencies)]
        ex_tone1.frequencies = [(2 * f2 - f1) for f1, f2 in zip(self.tones[0].frequencies, self.tones[1].frequencies)]

        # Get the bases from the expected tone
        _, real_basis_ex0, f_at_sample_ex0 = ex_tone0.get(return_freq=True)
        _, imag_basis_ex0 = ex_tone0.get(phi=np.pi / 2)
        _, real_basis_ex1, f_at_sample_ex1 = ex_tone1.get(return_freq=True)
        _, imag_basis_ex1 = ex_tone1.get(phi=np.pi / 2)

        # TO FREQUENCY SPACE!
        trim_frequency = self.tones[0].frequencies[0] * 0.25 + self.tones[0].frequencies[-1] * 0.75
        block_size = self.calibration.block_size
        n_regions = self.calibration.n_regions
        trimmed_signal, _ = trim_signal_max_correlation_at_f(
            signal.T[None], f_at_sample0, trim_frequency, real_basis0, imag_basis0, block_size, False
        )
        normalize_basis = False
        f1, p_F1 = dft_known_basis(
            trimmed_signal,
            f_at_sample0,
            real_basis0,
            imag_basis0,
            block_size,
            n_regions,
            normalize_basis=normalize_basis
        )
        f2, p_F2 = dft_known_basis(
            trimmed_signal,
            f_at_sample1,
            real_basis1,
            imag_basis1,
            block_size,
            n_regions,
            normalize_basis=normalize_basis
        )
        f1t2_f2, p_F1_dp = dft_known_basis(
            trimmed_signal,
            f_at_sample_ex0,
            real_basis_ex0,
            imag_basis_ex0,
            block_size,
            n_regions,
            normalize_basis=normalize_basis
        )
        f2t2_f1, p_F2_dp = dft_known_basis(
            trimmed_signal,
            f_at_sample_ex1,
            real_basis_ex1,
            imag_basis_ex1,
            block_size,
            n_regions,
            normalize_basis=normalize_basis
        )
        # _, p_raw_F2_dpB = dft_known_basis(
        #     trimmed_signal,
        #     f_at_sample_ex1,
        #     real_basis_ex1,
        #     imag_basis_ex1,
        #     block_size,
        #     n_regions,
        #     normalize_basis=False
        # )
        # _, p_raw_F2_dp = dft_known_basis(
        #     trimmed_signal,
        #     f_at_sample_ex1,
        #     real_basis_ex1,
        #     imag_basis_ex1,
        #     block_size,
        #     n_regions,
        #     normalize_basis=True
        # )
        # plt.loglog(f, np.abs(p_raw_F2_dp[0, 0]), label="renormal")
        # plt.loglog(f, np.abs(p_raw_F2_dpB[0, 0]), ":", label="no renormal")
        # plt.legend()
        # plt.show()
        # now the TOTAL signal to frequency space (correlate signal with itself)
        s = 1
        signal_data, signal_noise, p_noise = dft_known_basis_noise(
            trimmed_signal,
            [f_at_sample1, f_at_sample0, f_at_sample_ex0, f_at_sample_ex1],
            [real_basis0, real_basis1, real_basis_ex0, real_basis_ex1],
            [imag_basis0, imag_basis1, imag_basis_ex0, imag_basis_ex1],
            [p_F1 * s, p_F2 * s, p_F1_dp * s, p_F2_dp * s],
            self.tones[0].samplerate,
            block_size,
            n_regions,
            normalize_basis=normalize_basis,
            bandpass_frac=0.01
        )
        # I = signal_data[0, 3] != 0
        # plt.semilogx(f_at_sample1[I], trimmed_signal[0,3, I], label='Original Signal')
        # plt.semilogx(f_at_sample1[I], signal_data[0,3, I], label='Reconstructed Signal')
        # plt.semilogx(f_at_sample1[I], signal_noise[0,3, I], label='Filtered Residual')
        # # plt.semilogx(f_at_sample1[I], real_basis0[I]*0.015, label='base0')
        # # plt.semilogx(f_at_sample1[I], real_basis1[I]*0.015, label='base1')
        # plt.legend()
        # plt.xlabel("F2 Frequency")
        # plt.ylabel("Magnitude (in WAV file, uncalibrated)")
        # plt.grid()
        # # plt.xlim(197000, 197000+1024)
        # plt.show()
        # plt.figure()
        # plt.loglog(f, np.real(p_noise[0, 3]), 'k', label="noise")
        # # plt.loglog(f, np.imag(p_full[0, 3]), 'b', label="imag")
        # # plt.loglog(f, np.abs(p_full[0, 3]), 'g', label="abs")
        # plt.loglog(f, np.abs(p_F1[0, 3]), label="F1")
        # plt.loglog(f, np.abs(p_F2[0, 3]), label="F2")
        # plt.loglog(f, np.abs(p_F1_dp[0, 3]), label="F1_dp")
        # plt.loglog(f, np.abs(p_F2_dp[0, 3]), label="F2_dp")
        # plt.loglog(f, np.abs(p_F2_dp[0, 3] + p_F1_dp[0, 3] + p_F1[0, 3] + p_F2[0, 3]), label="summed contribs")
        # # plt.loglog(f, np.abs(p_F2_dp[0, 3]) + np.abs(p_F1_dp[0, 0]) + np.abs(p_F1[0, 0]) + np.abs(p_F2[0, 0]), label="summed contribs")
        # plt.legend()
        # plt.show()

        # plt.figure()
        # plt.loglog(f, np.real(p_noise.sum(axis=1)[0]), 'k', label="noise")
        # # plt.loglog(f, np.imag(p_full[0, 3]), 'b', label="imag")
        # # plt.loglog(f, np.abs(p_full[0, 3]), 'g', label="abs")
        # plt.loglog(f, np.abs(p_F1.sum(axis=1)[0]), label="F1")
        # plt.loglog(f, np.abs(p_F2.sum(axis=1)[0]), label="F2")
        # plt.loglog(f, np.abs(p_F1_dp.sum(axis=1)[0]), label="F1_dp")
        # plt.loglog(f, np.abs(p_F2_dp.sum(axis=1)[0]), label="F2_dp")
        # plt.loglog(f, np.abs(p_F2_dp.sum(axis=1)[0] + p_F1_dp.sum(axis=1)[0] + p_F1.sum(axis=1)[0] + p_F2.sum(axis=1)[0]), label="summed contribs")
        # # plt.loglog(f, np.abs(p_F2_dp[0, 3]) + np.abs(p_F1_dp[0, 0]) + np.abs(p_F1[0, 0]) + np.abs(p_F2[0, 0]), label="summed contribs")
        # plt.legend()
        # plt.show()

        # f, p_F1 = dft_known_basis(
        #     trimmed_signal.sum(axis=1),
        #     f_at_sample0,
        #     real_basis0,
        #     imag_basis0,
        #     block_size,
        #     n_regions,
        #     normalize_basis=normalize_basis
        # )
        # _, p_F2 = dft_known_basis(
        #     trimmed_signal.sum(axis=1),
        #     f_at_sample1,
        #     real_basis1,
        #     imag_basis1,
        #     block_size,
        #     n_regions,
        #     normalize_basis=normalize_basis
        # )
        # _, p_F1_dp = dft_known_basis(
        #     trimmed_signal.sum(axis=1),
        #     f_at_sample_ex0,
        #     real_basis_ex0,
        #     imag_basis_ex0,
        #     block_size,
        #     n_regions,
        #     normalize_basis=normalize_basis
        # )
        # _, p_F2_dp = dft_known_basis(
        #     trimmed_signal.sum(axis=1),
        #     f_at_sample_ex1,
        #     real_basis_ex1,
        #     imag_basis_ex1,
        #     block_size,
        #     n_regions,
        #     normalize_basis=normalize_basis
        # )

        # s = 1
        # signal_data, signal_noise, p_noise = dft_known_basis_noise(
        #     trimmed_signal.sum(axis=1),
        #     [f_at_sample1, f_at_sample0, f_at_sample_ex0, f_at_sample_ex1],
        #     [real_basis0, real_basis1, real_basis_ex0, real_basis_ex1],
        #     [imag_basis0, imag_basis1, imag_basis_ex0, imag_basis_ex1],
        #     [p_F1 * s, p_F2 * s, p_F1_dp * s, p_F2_dp * s],
        #     self.tones[0].samplerate,
        #     block_size,
        #     n_regions,
        #     normalize_basis=normalize_basis,
        #     bandpass_frac=0.01
        # )

        # plt.figure()
        # plt.loglog(f, np.real(p_noise[0]), 'k', label="noise")
        # # plt.loglog(f, np.imag(p_full[0]), 'b', label="imag")
        # # plt.loglog(f, np.abs(p_full[0]), 'g', label="abs")
        # plt.loglog(f, np.abs(p_F1[0]), label="F1")
        # plt.loglog(f, np.abs(p_F2[0]), label="F2")
        # plt.loglog(f, np.abs(p_F1_dp[0]), label="F1_dp")
        # plt.loglog(f, np.abs(p_F2_dp[0]), label="F2_dp")
        # plt.loglog(f, np.abs(p_F2_dp[0] + p_F1_dp[0] + p_F1[0] + p_F2[0]), label="summed contribs")
        # # plt.loglog(f, np.abs(p_F2_dp[0, 3]) + np.abs(p_F1_dp[0, 0]) + np.abs(p_F1[0, 0]) + np.abs(p_F2[0, 0]), label="summed contribs")
        # plt.legend()
        # plt.show()

        # Store frequency data for later analysis
        self.f1 = f1
        self.f2 = f2

        # Calibrate all the pressures and return information
        self.p_F1_cal = self.calibration.cal_p(f1, p_F1)
        self.p_F2_cal = self.calibration.cal_p(f2, p_F2)
        self.p_2F1_F2_cal = self.calibration.cal_p(f1t2_f2, p_F1_dp)
        self.p_2F2_F1_cal = self.calibration.cal_p(f2t2_f1, p_F2_dp)
        self.p_noise = self.calibration.cal_p((f1 + f2) / 2, p_noise)

        return self.p_2F1_F2_cal, self.p_2F2_F1_cal, p_noise

    def plot_results(self, figkwargs={}, kwargs={}, show=True):
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, **figkwargs)
        for i in range(4):
            ii = i // 2
            jj = i % 2
            # axs[ii, jj].semilogx(0.5*(self.f2 + self.f1), todB(self.p_noise[0, i]), "k", label="Noise", **kwargs)
            # axs[ii, jj].semilogx(self.f1 * 2 - self.f2, todB(self.p_2F1_F2_cal[0, i]), label="2F1 - F2", **kwargs)
            # axs[ii, jj].semilogx(self.f2 * 2 - self.f1, todB(self.p_2F2_F1_cal[0, i]), label="2F2 - F1", **kwargs)
            f = 0.5*(self.f2 + self.f1)
            axs[ii, jj].semilogx(f, todB(self.p_F1_cal[0, i]), label="F1", **kwargs)
            axs[ii, jj].semilogx(f, todB(self.p_F2_cal[0, i]), label="F2", **kwargs)
            axs[ii, jj].semilogx(f, todB(self.p_noise[0, i]), "k", label="Noise", **kwargs)
            axs[ii, jj].semilogx(f, todB(self.p_2F1_F2_cal[0, i]), label="2F1 - F2", **kwargs)
            axs[ii, jj].semilogx(f, todB(self.p_2F2_F1_cal[0, i]), label="2F2 - F1", **kwargs)
            axs[ii, jj].set_xlim(200, 20000)
            axs[ii, jj].set_title("Channel %d" % i)
        axs[0, 0].legend()
        axs[1, 0].set_xlabel("Frequency(Hz)")
        axs[1, 1].set_xlabel("Frequency(Hz)")
        axs[0, 0].set_ylabel("Amplitude (dB SPL)")
        axs[1, 0].set_ylabel("Amplitude (dB SPL)")


        if show:
            plt.show()

        return fig, axs
