""" Function related to Wideband Acoustic Immittance """

from typing import List
import numpy as np
from scipy import ndimage

from auren.core.data_models import calibration
from auren.core.data_models import chirp

def make_tone(calibration: calibration.CalibrationData, tone: chirp.Chirp, level: float, pad_time: float=0, phi: float=0, weight_relax_time: float=None) -> np.ndarray:
    """Generates a tone wav file for the Auren speakers based on a calibration and desired chirp characteristics

    Parameters
    ----------
    calibration : calibration.CalibrationData
        The Auren calibration data
    tone : chirp.Chirp
        The characteristics of the chirp
    level : float
        The overall dB SPL level at which the chirp should be player
    pad_time : float, optional
        Amount of padding added to ramp up the chirp signal, in seconds, by default 0
    phi : float, optional
        The phase of the chirp at the start, by default 0
    weight_relax_time : _type_, optional
        The amount of time in seconds to pad between using only 1 speaker vs using 2 in the case where 1 speaker cannot
        produce enough output by itself for a particular frequency, by default uses 1024 samples based on the samplerate

    Returns
    -------
    np.ndarray
        _description_
    """
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