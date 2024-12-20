import numpy as np
from scipy import optimize, signal, ndimage
from numpydantic import NDArray
import typing as t


def to_fourier(time_series: np.ndarray, samplerate: float) -> tuple[np.ndarray, np.ndarray]:
    """Convert a time series into its Fourier transform.

    Parameters
    ----------
    time_series : np.ndarray of shape (n_samples,n_channels)
        The input time series data.
    samplerate : float
        The sample rate of the time series data.

    Returns
    -------
    frequency : np.ndarray of shape (n_frequencies)
        The frequencies corresponding to the Fourier transform.
    fourier : np.ndarray of shape (n_frequencies, n_channels)
        The Fourier transform of the input time series data (complex)
    """
    n = time_series.shape[-1]
    frequency = np.fft.fftfreq(n, 1 / samplerate)[: n // 2]  # frequency
    fourier = np.fft.fft(time_series, n, axis=-1)

    # Discard half the power. Power goes as P^2, hence sqrt(2)
    fourier = np.sqrt(2) * fourier[..., : frequency.size] / n

    return frequency, fourier


def from_fourier(fourier: np.ndarray) -> np.ndarray:
    """Generates a time signal from a fourier representation created using `to_fourier`

    Parameters
    ----------
    fourier : np.ndarray
        The complex coefficients of the fourier series

    Returns
    -------
    np.ndarray
        The time signal
    """
    a = fourier / np.sqrt(2) * (fourier.shape[-1] * 4)
    a = np.concatenate([a, a[1::-1]])
    time_series = np.fft.ifft(a, n=fourier.shape[-1] * 2, axis=-1)
    return np.real(time_series)


def find_subsample_alignment_offsets(
    times: np.ndarray,
    data: np.ndarray,
    buffer: int = 1,
    ref_channel: int = -1,
    cost_type: str = "diff",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Function to find subsample alignment offsets.

    Parameters
    ----------
    times : np.ndarray(shape=(..., n_channels, n_data_points))
        Array of times with times[..., i, :] giving the times for channel i.
    data : np.ndarray(shape=(..., n_channels, n_data_points))
        Array of amplitudes for data that will be aligned -- same shape as times.
    buffer : int, optional
        Buffer size, by default 1. The reference signal will has a size `n_data_points - 2 * buffer`
    ref_channel : int, optional
        Reference channel, by default -1. All data is aligned to this channel, which will have an offset of 0
    cost_type : str, optional
        Type of cost function to use. Either correlation function ()"corr") or squared differences ('diff'), by default 'diff'.

    Returns
    -------
    np.ndarray(shape=(..., n_channels))
        The offsets for each event, such that plot(times + offsets, data) are aligned
    np.ndarray(shape=(..., n_channels))
        The value of the cost function for each event.
    np.ndarray(shape=(..., n_channels))
        The value of the cost function with offset == 0.
    """
    # Do a sub-sample alignment of channels
    if cost_type == "diff":
        # print("Diff Type")
        def ss_cost(offset, ref_x, ref, match_x, match):
            diff = ref - np.interp(ref_x, match_x + offset, match, left=0, right=match[-1])
            return np.linalg.norm(diff)

    elif cost_type == "corr":
        # print("Corr Type")
        def ss_cost(offset, ref_x, ref, match_x, match):
            comp = np.interp(ref_x, match_x + offset, match, left=0, right=match[-1])
            corr = ((ref - ref.mean()) * (comp - comp.mean())).mean() / (ref.std() + 1e-16) / (comp.std() + 1e-16)
            return -corr

    else:
        raise ValueError("Unknown cost function types %s. Use either 'diff' or 'corr'." % (cost_type))

    n_channels = data.shape[-2]
    if ref_channel < 0:
        ref_channel = n_channels + ref_channel
    sub_sample_kernel_size = data.shape[-1] - buffer * 2
    buffer_slice = slice(buffer, sub_sample_kernel_size + buffer)
    offsets = np.zeros(times.shape[:-1])
    costs = np.zeros(times.shape[:-1])
    costs_old = np.zeros(times.shape[:-1])

    if len(times.shape) > 2:
        for s in range(times.shape[0]):
            # To deal with multi-dimensional inputs, we have to loop through the
            # starting dimensions recursively
            offsets[s], costs[s], costs_old[s] = find_subsample_alignment_offsets(
                times[s], data[s], buffer, ref_channel, cost_type
            )
        return offsets, costs, costs_old

    ref_signal = data[ref_channel, buffer_slice]
    ref_time = times[ref_channel, buffer_slice]
    for c in range(n_channels):
        if c == ref_channel:
            continue
        match_signal = data[c, :]
        match_time = times[c, :]
        bracket = [
            match_time[0] - match_time[buffer_slice.start],
            match_time[-1] - match_time[buffer_slice.stop - 1],
        ]
        # Do a line search
        cands = np.linspace(bracket[0], bracket[1], 64)
        cand_costs = [ss_cost(cand, ref_time, ref_signal, match_time, match_signal) for cand in cands]
        start_cand = cands[np.argmin(cand_costs)]

        # The 1D line search doesn't seem to work as well. So instead, I give it a good starting value and go from there...
        # res = optimize.minimize_scalar(
        #     ss_cost, bracket=bracket,
        #         args=(ref_time, ref_signal, match_time, match_signal))
        # offsets[c] = res.x
        # costs[c] = res.fun
        # costs_old[c] = ss_cost(start_cand, ref_time, ref_signal, match_time, match_signal)
        # if res.x < bracket[0] or res.x > bracket[1] or res.fun > costs_old[c]:
        res = optimize.minimize(
            ss_cost,
            np.array([start_cand]),
            args=(ref_time, ref_signal, match_time, match_signal),
            method="CG",
        )
        offsets[c] = res.x
        costs[c] = res.fun
        costs_old[c] = ss_cost(0, ref_time, ref_signal, match_time, match_signal)

    return offsets, costs, costs_old


def rolling_window(a, window):
    """ from https://stackoverflow.com/questions/6811183/rolling-window-for-1d-arrays-in-numpy"""
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def trim_signal_sharp_jump(
    data: np.ndarray,
    samplerate: int,
    duration: float,
    kernel_size: int = 65,
    end: bool = True,
    return_index: bool = False,
) -> np.ndarray:
    """
    Trim a signal to a specific duration, with a sharp jump at the beginning or end of the trimmed section.

    Parameters
    ----------
    data : np.ndarray(shape=(..., n))
        The input signal.
    samplerate : int
        The sample rate of the input signal.
    duration : float
        The desired duration in seconds of the trimmed signal.
    kernel_size : int, optional
        The size of the maximum filter kernel to use finding the jump. Default is 65.
    end : bool, optional
        If True (default), the function looks for the largest jump near the end of the signal. Otherwise, it will look for the largest jump at the beginning.
    return_index : bool, optional
        If True, the function will return the slice of the trimmed section instead of the trimmed section itself. Default is False.

    Returns
    -------
    np.ndarray or slice
        If `return_index` is False (default), the trimmed signal is returned as a numpy array with the same dtype as the input signal. Otherwise, the slice needed to trim the signal is returned.
    """
    # Construct the kernel
    trim_size = int(np.round(samplerate * duration))
    data_max = ndimage.maximum_filter1d(np.abs(data), size=kernel_size, axis=-1)
    if end:
        slope = data_max[trim_size + 1 :] - data_max[trim_size:-1]
        ind = np.argmin(slope)
    else:
        slope = data_max[1:-trim_size] - data_max[: -trim_size - 1]
        ind = np.argmax(slope)

    # It's a bit crazy, but the math works out that this is valid for both start and end
    slc = slice(ind, ind + trim_size)

    if return_index:
        return slc
    return data[slc]


def trim_signal_mean_amplitude(
    data: np.ndarray, samplerate: int, duration: float, return_index: bool = False
) -> np.ndarray:
    """
    Trim a signal to a specific duration, ensuring the trimmed section has the largest mean absolute amplitude

    Parameters
    ----------
    data : np.ndarray(shape=(..., n))
        The input signal.
    samplerate : int
        The sample rate of the input signal.
    duration : float
        The desired duration in seconds of the trimmed signal.
    return_index : bool, optional
        If True, the function will return the slice of the trimmed section instead of the trimmed section itself. Default is False.

    Returns
    -------
    np.ndarray or slice
        If `return_index` is False (default), the trimmed signal is returned as a numpy array with the same dtype as the input signal. Otherwise, the slice needed to trim the signal is returned.
    """
    # Construct the kernel
    kernel_size = int(np.round(samplerate * duration))

    ## Find the spot in the data with the maximum amplitude
    data_mean = ndimage.uniform_filter1d(np.abs(data), kernel_size, axis=-1)

    ind = np.argmax(data_mean)
    slc = slice(ind - kernel_size // 2, ind + kernel_size // 2 + 2)

    if return_index:
        return slc
    return data[..., slc]


def trim_signal_max_correlation_at_f(
    data: np.ndarray,
    f_at_sample: np.ndarray,
    trim_freq: float,
    real_basis: np.ndarray,
    imag_basis: np.ndarray,
    block_size: int,
    use_global_trim_offset: bool,
) -> t.Tuple[np.ndarray, t.List[slice]]:
    """Trim data by finding the max correlation of a signal with a reference basis at a specific frequency

    Parameters
    ----------
    data : np.ndarray
        The data to be trimmed
    f_at_sample : np.ndarray
        The frequency of the data for that sample in the basis
    trim_freq : float
        Frequency used to figure out the max correlations
    real_basis : np.ndarray
        The real part of the expected basis function, where the first item in the list is the time
    imag_basis : np.ndarray
        The imaginary part (90 deg out of phase) of the basis function
    block_size : int
        The block size over which to look for a maximum
    use_global_trim_offset : bool
        If True, a single trim offset is returned. Otherwise, if data has multiple dimensions, the offset is averaged
        over axis=1

    Returns
    -------
    np.ndarray, t.List[slice]]
        The trimmed signal
    List[slice]
        The list of slices used to trim the data.
    """
    n_samples = real_basis.shape[0]
    i = np.argmin(np.abs(f_at_sample - trim_freq))
    sym_pad = (data.shape[-1] - n_samples) // 2

    sub_slice = slice(
        i - int(block_size * 1.5) + sym_pad,
        i + int(block_size * 1.5) + 1 + sym_pad,
    )
    sub_slice2 = slice(i - block_size // 2, i + block_size // 2 + 1)
    subdata = data[..., sub_slice]
    r_base = real_basis[sub_slice2]
    i_base = imag_basis[sub_slice2]

    real_c = ndimage.convolve1d(subdata, r_base[::-1], axis=-1)
    imag_c = ndimage.convolve1d(subdata, i_base[::-1], axis=-1)
    # tube = 3
    mag_c = np.sqrt(
            real_c ** 2 + imag_c ** 2
        )  # [..., block_size // 2: block_size // 2 + self.calibration_block_size]
    # t = np.arange(mag_c.shape[-1]) - mag_c.shape[-1] // 2
    # plt.plot(real_c[tube, 0])
    # plt.plot(imag_c[tube, 0])
    # plt.plot(mag_c[tube, 0])
    # plt.show()
    # plt.plot(t, mag_c[tube].T)
    # plt.show()
    offset = np.argmax(mag_c, axis=-1) - mag_c.shape[-1] // 2
    offset = np.round(offset.mean(axis=1)).astype(int)  # Average over the channels
    if use_global_trim_offset:
        offset[:] = np.round(offset.mean()).astype(int)  # Average over all the tube

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


def make_chirp(
    times: np.ndarray,
    f0: float,
    f1: float,
    amplitude_func=None,
    phi: float = 0,
    return_freq: bool = False,
    return_phase: bool = False,
) -> np.ndarray:
    """Generates a logarithmic/exponential chirp in the time domain

    Parameters
    ----------
    times : np.ndarray
        Array of times
    f0 : float
        Starting frequency
    f1 : float
        Ending frequency
    amplitude_func : function, optional
        function of frequency -- used to modulate amplitude of the chirp depending on the current frequency, by default uses amplitude of 1
    phi : int, optional
        Phase offset in radians, by default 0
    return_freq : bool, optional
        Default is False. If True, also returns the frequency as a function of time for the chirp as the SECOND argument

    Returns
    -------
    np.ndarray
        The chirp signal
    """
    if amplitude_func is None:
        amplitude_func = lambda f: 1
    t1 = times.max()
    # See https://en.wikipedia.org/wiki/Chirp#Exponential
    freq = f0 * (pow(f1 / f0, (times / t1)))
    beta = t1 / np.log(f1 / f0)
    phase = 2 * np.pi * beta * (freq - f0) + phi  # -f0 is a phase shift because scipy uses cos and wikipedia uses sin
    signal = amplitude_func(freq) * np.cos(phase)
    if not (return_freq or return_phase):
        return signal
    out = [signal]
    if return_freq:
        out.append(freq)
    if return_phase:
        out.append(phase)
    return tuple(out)


def make_multi_chirp(
    frequencies: t.List[float],
    durations: t.List[float],
    samplerate: float,
    amplitude_func=None,
    phi: float = 0,
    return_freq: bool = False,
    return_phase: bool = False,
    pad_time=0,
) -> np.ndarray:
    """Generates a logarithmic/exponential chirp in the time domain

    Parameters
    ----------
    frequencies : list[float]
        List of frequencies
    durations : list[float]
        List of durations. len(durations) == len(frequencies) - 1, and total_duration is sum(durations)
    samplerate : float
        Samplerate for the time signal
    amplitude_func : function, optional
        function of frequency -- used to modulate amplitude of the chirp depending on the current frequency, by default uses amplitude of 1
    phi : int, optional
        Phase offset in radians, by default 0
    return_freq : bool, optional
        Default is False. If True, also returns the frequency as a function of time for the chirp as the SECOND argument

    Returns
    -------
    np.ndarray
        The time array
    np.ndarray
        The chirp signal
    np.ndarray, optional if return_freq == True
        The frequency at each time sample
    """

    total_duration = int(np.sum(durations) * samplerate) / samplerate
    times = np.linspace(0, total_duration, int(total_duration * samplerate))
    data = np.zeros(times.shape)
    f_at_t = np.zeros(times.shape)
    p_at_t = np.zeros(times.shape)

    durations_sum = 0
    last_phase = phi
    for i, dur in enumerate(durations):
        I = (times >= durations_sum) & (times < (durations_sum + dur + 1 / samplerate))
        durations_sum += dur
        s, fatt, patt = make_chirp(
            times[I] - times[I][0],
            frequencies[i],
            frequencies[i + 1],
            amplitude_func,
            last_phase,
            return_freq=True,
            return_phase=True,
        )
        data[I] = s
        f_at_t[I] = fatt
        p_at_t[I] = patt + last_phase
        last_phase = patt[-1]

    if pad_time > 0:
        times, data = pad_chirp(times, data, frequencies[0], frequencies[-1], pad_time, phi, last_phase=p_at_t[-1])

    out = [times, data]
    if return_freq:
        out.append(f_at_t)
    if return_phase:
        out.append(p_at_t)
    return tuple(out)


def pad_chirp(
    times: np.ndarray, signal: np.ndarray, f0: float, f1: float, pad_time: float, phi: float = 0, last_phase=0
):
    """Pads a time series signal -- specifically a chirp, with an increasing amplitude wave at the start
    and end of the signal.

    Parameters
    ----------
    times : np.ndarray
        The times
    signal : np.ndarray
        The chirp to be padded
    f0 : float
        Starting frequency
    f1 : float
        Ending frequency
    pad_time : float
        Time to add to start/end of signal for padding
    phi : int, optional
        Phase offset in radians, by default 0
    Returns
    -------
    np.ndarray
        New times for padded signal (times go negative)
    np.ndarray
        New padded chirp signal
    """
    dt = times[1] - times[0]
    n = int(pad_time // dt)
    pad_time = n * dt

    if not last_phase:
        t1 = times[-1]
        # See https://en.wikipedia.org/wiki/Chirp#Exponential
        # freq = f0 * (pow(f1 / f0, (times / t1)))
        beta = t1 / np.log(f1 / f0)
        last_phase = (
            2 * np.pi * beta * (f1 - f0)
        )  # -f0 is a last_phase shift because scipy uses cos and wikipedia uses sin

    pad_times = np.linspace(0, pad_time, n)
    new_times = np.concatenate([pad_times - pad_time - dt, times, pad_times + times.max() + dt])
    a0 = signal[0] / np.cos(2 * np.pi * f0 * times[0] + phi) * pad_times / pad_times.max()
    a1 = signal[-1] / np.cos(2 * np.pi * f1 * times[-1] + phi + last_phase) * pad_times[::-1] / pad_times.max()
    new_signal = np.concatenate(
        [np.cos(2 * np.pi * f0 * new_times[:n] + phi) * a0, signal, np.cos(2 * np.pi * f1 * new_times[-n:] + phi) * a1]
    )
    return new_times, new_signal


def dft_known_basis(
    data: NDArray,
    f_at_sample: NDArray,
    basis_real: NDArray,
    basis_imag: NDArray,
    block_size: int = 2048,
    n_regions: int = 256,
) -> t.Tuple[NDArray, NDArray]:

    n_samples = data.shape[-1]
    start_inds = np.round(np.linspace(0, n_samples - block_size, n_regions)).astype(int)

    han_win = signal.windows.hann(block_size, sym=True)
    # Normalize so we can use it with the orthogonal basis
    han_win = han_win / han_win.mean()
    # han_win = han_win * 0 + 1

    # Initialize outputs
    f_centers = f_at_sample[start_inds + block_size // 2]
    fourier = np.zeros(data.shape[:-1] + (n_regions,), complex)

    for i, start_ind in enumerate(start_inds):
        slc = slice(start_ind, start_ind + block_size)
        real = (basis_real[..., slc] * data[..., slc] * han_win).mean(axis=-1)
        imag = (basis_imag[..., slc] * data[..., slc] * han_win).mean(axis=-1)
        fourier[..., i] = 2 * (real + 1j * imag)

    return f_centers, fourier
