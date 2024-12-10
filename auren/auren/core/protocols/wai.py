""" Function related to Wideband Acoustic Immittance """
import os
from typing import List, Tuple, Any, Optional
import numpy as np
from scipy import ndimage
from pydantic import BaseModel
from matplotlib import pyplot as plt

import auren
from auren.core.data_models import calibration
from auren.core.data_models import chirp
from auren.core import model
from auren.core.utils import todB
from auren.core.signal_processing_utils import dft_known_basis, trim_signal_max_correlation_at_f


class WAI(BaseModel):
    calibration: calibration.CalibrationData
    tone: chirp.Chirp
    level: float = 65
    pad_time: float = 0.25
    mic_channels_used: Tuple[int, int] = [1, 3]
    units: str = "mm"
    ear_diameter: float = 3.75 * 2

    power_reflectance: Optional[Any] = None
    impedance: Optional[Any] = None
    f: Optional[Any] = None
    p_cal: Optional[Any] = None
    p_cal_at_mic1: Optional[Any] = None
    u_cal_at_mic1: Optional[Any] = None

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


    def make_tone(self, phi: float=0, weight_relax_time: float=None) -> np.ndarray:
        """Generates a tone wav file for the Auren speakers based on a calibration and desired chirp characteristics

        Parameters
        ----------
        phi : float, optional
            The phase of the chirp at the start, by default 0
        weight_relax_time : float, optional
            The amount of time in seconds to pad between using only 1 speaker vs using 2 in the case where 1 speaker cannot
            produce enough output by itself for a particular frequency, by default uses 1024 samples based on the samplerate

        Returns
        -------
        np.ndarray
            _description_
        """
        #shorter names
        calibration = self.calibration
        pad_time = self.pad_time
        tone = self.tone
        level = self.level

        if weight_relax_time is None:
            weight_relax_time = 1024 / tone.samplerate
        # Create the full-scaled amplitude functions fs frequency for each speaker
        amp_funcs = calibration.cal_speaker(level)
        # Generate the chirps, where each speaker tries to produce the full-scale sound
        chirps = [tone.get(phi, amplitude_func=amp_func, pad_time=pad_time, return_freq=True) for amp_func in amp_funcs]
        # Evaluate the amplitude function based on the actual frequency vs. time function produced
        actual_amps = np.stack([amp_func(c[2]) for amp_func, c in zip(amp_funcs, chirps)], axis=1)

        # Now we figure out how to weight the two speakers so that we get the
        # desired Pa. Assumption is that 0.5 weight == 0.5 Pa (linear in pascals)
        # When 1 of the two speakers is > 1, only use the speaker with < 1
        weights = actual_amps.copy()
        weights[weights >= 1] = 0
        # If we can't make the level even with both speakers, see if we can make it
        # with half the power
        can_t_do_inds = weights.sum(axis=1) == 0
        # Restore the weights we can't do
        weights[can_t_do_inds] = actual_amps[can_t_do_inds]
        # We can produce the maximum level of:
        # frac_of_requested = 1 / a  (where a is > 1, so we have to reduce output by 1/a for the speaker to actually do it (FS))
        # 1 / a + 1 / b > 1 -- then the two speakers combined can do it
        # For example, consider both speakers need output of 2 FS, then:
        #    1/2 + 1/2 = 1 -- so at half the required power for a single speaker, combined can produce the requested level
        # For example, consider both speakers need output of 1.5 FS, then:
        #    1/1.25 + 1/1.25 = 1.6 -- so combined, speakers can produce 1.6x the level required if both play at FS.
        can_produce_frac = (1 / weights[can_t_do_inds]).sum(axis=1, keepdims=True)
        # If we can produce the fraction, then each speaker contributes:
        # 1 / a / can_produce_frac --> for speaker 'a', similarly for 'b'
        # If they can't produce the level, just max out both speakers
        weights[can_t_do_inds] = (
            (can_produce_frac >= 1) * ((1 / weights[can_t_do_inds]) / can_produce_frac)
            + (can_produce_frac < 1) * 1  # Full scale for both speakers, still below requested level :(
        )
        # Keep track of frequencies where we don't hit the requested level
        level_not_met = np.zeros(can_t_do_inds.size, bool)
        level_not_met[can_t_do_inds] = can_produce_frac.squeeze() < 1
        freq_level_not_met = chirps[0][2][level_not_met]


        # Make a smooth ramp between using only one speaker v.s. using both
        remaining_inds = ~can_t_do_inds & ~np.any(weights <= 0, axis=1)
        jump_weights = np.any(weights == 0, axis=1)
        smooth_weight_samples = int(weight_relax_time * tone.samplerate)
        ramp = np.linspace(0, 1, smooth_weight_samples)
        smooth_weights = np.ones_like(weights) * 0.5
        smooth_weights[jump_weights, :] = 1
        smooth_weights[weights == 0] = 0

        ramp_down_starts = np.argwhere((~jump_weights[1:]) & (jump_weights[:-1]))[:, 0]
        for ramp_down_start in ramp_down_starts:
            inactive_channel = int(weights[ramp_down_start, 1] == 0)
            smooth_weights[ramp_down_start: ramp_down_start + smooth_weight_samples, inactive_channel] = ramp / 2
            smooth_weights[ramp_down_start: ramp_down_start + smooth_weight_samples, 1 - inactive_channel] = 1 - ramp / 2
        ramp_up_ends = np.argwhere((jump_weights[1:]) & (~jump_weights[:-1]))[:, 0]
        for ramp_up_end in ramp_up_ends:
            # A bit of clever math to deal with the inactive and active channel separately
            inactive_channel = int(weights[ramp_down_start, 1] == 0)
            smooth_weights[ramp_up_end + 1 - smooth_weight_samples: ramp_up_end + 1, inactive_channel] = 0.5 - ramp / 2
            smooth_weights[ramp_up_end + 1 - smooth_weight_samples: ramp_up_end + 1, 1 - inactive_channel] = 0.5 + ramp / 2
        # smooth_weights[jump_weights == 1] = smooth_weights[weights == 1]
        # smooth_weights[weights == 0] = 0
        # plt.plot(jump_weights); plt.plot(smooth_weights, '.');
        # # plt.xlim(ramp_down_starts[0] - smooth_weight_samples, ramp_down_starts[0] + smooth_weight_samples)
        # plt.show()
        # weights[remaining_inds] = 0.5 * (1 - weights[remaining_inds]) ** 2 / ((1 - weights[remaining_inds]) ** 2).sum(axis=1, keepdims=True)
        weights = smooth_weights

        # TODO: Finally, adjust weights, preferring the speaker that doesn't have to work as hard
        # weights = (1 - weights[remaining_inds]) ** 2 / ((1 - weights[remaining_inds]) ** 2).sum(axis=1, keepdims=True)

        # Because of how the chirp signal is created, we can just multiply these weights through -- but we do have to pad
        # the weights because the frequencies are NOT padded
        pad_samples = int(pad_time * tone.samplerate - 1)

        chirp_channels = np.stack([c[1] for c in chirps], axis=1)
        pad_weights = np.zeros_like(chirp_channels)
        pad_weights[:pad_samples] = weights[0]
        # The following line will fail if the pad dimension is incorrect
        # This might happen in some cases
        pad_weights[pad_samples:-pad_samples] = weights
        pad_weights[-pad_samples:] = weights[-1]

        chirp_channels *= pad_weights

        return chirp_channels, freq_level_not_met, pad_weights

    def analyze_iteration(self, signal):
        _, real_basis, f_at_sample = self.tone.get(return_freq=True)
        _, imag_basis = self.tone.get(phi=np.pi / 2)
        trim_frequency = self.tone.frequencies[0] * 0.25 + self.tone.frequencies[-1] * 0.75
        block_size = self.calibration.block_size  # 1024
        n_regions = self.calibration.n_regions  # 256
        trimmed_signal, _ = trim_signal_max_correlation_at_f(
            signal.T[None], f_at_sample, trim_frequency, real_basis, imag_basis, block_size, False
        )
        f, p_raw = dft_known_basis(
            trimmed_signal,
            f_at_sample,
            real_basis,
            imag_basis,
            block_size,
            n_regions,
        )
        self.p_cal = self.calibration.cal_p(f, p_raw)

        mic0, mic1 = self.mic_channels_used
        units = auren.units(self.calibration.mic[mic0].distance_units)
        x = self.calibration.mic[mic1].distance * units
        self.u_cal_at_mic1 = self.cavern_model.u_measured(
            f * auren.units.Hz,
            x,
            self.p_cal[..., mic0, :] * auren.units.Pa,
            self.p_cal[..., mic1, :] * auren.units.Pa,
            in_D1=True
            ).to_base_units()
        self.p_cal_at_mic1 = self.cavern_model.p_measured(
            f * auren.units.Hz,
            x,
            self.p_cal[..., mic0, :] * auren.units.Pa,
            self.p_cal[..., mic1, :] * auren.units.Pa,
            in_D1=True
            ).to_base_units()
        area_ear = self.cavern_model.D1 ** 2 * np.pi / 4
        self.impedance = (self.p_cal_at_mic1 / self.u_cal_at_mic1 / area_ear).to_base_units()
        # characteristic impedence
        z0 = self.cavern_model.z0
        self.power_reflectance = np.abs((self.p_cal_at_mic1 - self.u_cal_at_mic1 * z0) / (self.p_cal_at_mic1 + self.u_cal_at_mic1 * z0))**2
        self.f = f

        return self.impedance, self.power_reflectance

    def plot_results(self, figkwargs={}, kwargs={}, show=True):
        # Load normative data from rosowski et al
        path = os.path.join(os.path.dirname(__file__), "normative_data", "Rosowski-et-al-2012-Normal-Paper-Data-Power-Reflectance.csv")
        norm_data = np.genfromtxt(path, delimiter=',')
        norm_avg = norm_data[:, -2]
        norm_std = norm_data[:, -1]
        norm_f = norm_data[:, 0]

        fig, axs = plt.subplots(2, 2, sharex=True, **figkwargs)
        f = self.f
        axs[0, 0].semilogx(f, 1 - self.power_reflectance[0], **kwargs)
        axs[0, 0].fill_between(norm_f, 1 - np.clip(norm_avg - norm_std, 0, 1), 1 - np.clip(norm_avg + norm_std, 0, 1), color='k', alpha=0.3)
        axs[0, 0].set_title("Absorbance")
        axs[0, 0].set_ylim(-0.1, 1.1)
        axs[0, 0].grid()
        axs[1, 0].set_ylabel("Fraction")
        axs[1, 0].fill_between(norm_f, np.clip(norm_avg - norm_std, 0, 1), np.clip(norm_avg + norm_std, 0, 1), color='k', alpha=0.3)
        axs[1, 0].semilogx(f, self.power_reflectance[0], **kwargs)
        axs[1, 0].set_title("Power Reflectance")
        axs[1, 0].set_ylabel("Fraction")
        axs[1, 0].set_ylim(-0.1, 1.1)
        axs[1, 0].grid()
        axs[0, 1].semilogx(f, todB(self.impedance[0], ref=self.cavern_model.za1), **kwargs)
        axs[0, 1].set_title("Impedance Magnitude")
        axs[0, 1].grid()
        axs[0, 1].set_ylabel("dB re Z0")
        axs[1, 1].semilogx(f, f*0 - 90, "k")
        axs[1, 1].semilogx(f, np.rad2deg(np.angle(self.impedance[0].magnitude)), **kwargs)
        axs[1, 1].set_title("Impedance Phase (degrees)")
        axs[1, 1].set_ylim(-100, 100)
        axs[1, 1].grid()
        axs[1, 1].set_ylabel("Degrees")
        axs[1, 1].set_xlabel("Frequency (Hz)")
        axs[1, 0].set_xlabel("Frequency (Hz)")

        if show:
            plt.show()

        return fig, axs

