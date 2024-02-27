import numpy as np
from scipy import optimize

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
    n = time_series.size
    frequency = np.fft.fftfreq(n, 1/samplerate)[:n//2]   # frequency
    fourier = np.fft.fft(time_series, n)

    # Discard half the power. Power goes as P^2, hence sqrt(2)
    fourier = np.sqrt(2) * fourier[:frequency.size] / n

    return frequency, fourier


def find_subsample_alignment_offsets(times: np.ndarray, data: np.ndarray, buffer: int=1, ref_channel: int=-1, cost_type: str='diff') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        raise ValueError("Unknown cost function types %s. Use either 'diff' or 'corr'." %(cost_type))

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
                offsets[s], costs[s], costs_old[s] = find_subsample_alignment_offsets(times[s], data[s], buffer, ref_channel, cost_type)
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
            ss_cost, np.array([start_cand]),
                args=(ref_time, ref_signal, match_time, match_signal),
                method="CG")
        offsets[c] = res.x
        costs[c] = res.fun
        costs_old[c] = ss_cost(0, ref_time, ref_signal, match_time, match_signal)

    return offsets, costs, costs_old

