import numpy as np
from scipy import optimize, signal, ndimage


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
