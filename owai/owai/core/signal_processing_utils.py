import numpy as np

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
