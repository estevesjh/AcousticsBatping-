import audiosegment
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

## Physical constants
ms = 1e-3
khz = 1e3
# pulse width
pulse_width = 2. # ms

class timeDelay:
    """
        Measure time delay between two signals from batping data

    Example:
    >>> fname = './data/DomeTest109.aiff'
    >>> td = timeDelay(fname)
    >>> tdelay, elapsed_time = td.compute_time_delay_over_dices()

    """
    max_window_size = 5 # ms
    def __init__(self, fname, pulse_distance_ms=250, sample_rate=384000):
        self.fname = fname
        self.sample_rate = sample_rate
        self.fs = 1/sample_rate

        ## load audio data
        self.audio = self.load_data(fname)
        self.duration = self.audio.to_numpy_array().size*self.fs

        ## slice data in smaller periods
        self.dices = self.dice(pulse_distance_ms)
        self.dice_period = pulse_distance_ms*ms
        pass

    def load_data(self, fname):
        return audiosegment.from_file(fname)
    
    def dice(self, period):
        return self.audio.dice(period*ms, zero_pad=False)
    
    def compute_time_delay(self, audio, lo=1, hi=90, min_delay=15, max_delay=250):
        """
        Compute time delay for a given signal
        """
        signal, bands = decompose_signal(audio, lo, hi)
        peaks, pulset0, tdelay = measure_time_delay(signal[0], min_delay, max_delay, fs=self.fs)
        return peaks, pulset0, tdelay
    
    def compute_time_delay_over_dices(self, lo=1, hi=90, min_delay=15, max_delay=250):
        """
        Compute time delay over periods of 250ms in the audio file

        The returned quantities are for each dice

        Return:
            tdelay: array, time delay in ms 
            times: array, elapsed time in ms 
            peaks: array, index location of the peaks
            pulset_0s: array, index location of the initial pulse

        """
        self.init_dice_output()
        for i in range(self.ndices):
            peak, pulset0, td = self.compute_time_delay(self.dices[i], lo, hi, min_delay, max_delay)

            self.times[i] = self.dice_period*(i+0.5)
            self.tdelay[i] = td
            self.peaks.append(peak)
            self.pulse_t0.append(pulset0)
        pass

    def init_dice_output(self):
        ndices = len(self.dices)        
        self.ndices = len(self.dices)
        self.tdelay = np.zeros(ndices,dtype=np.float64)
        self.times  = np.zeros(ndices,dtype=np.float64)
        self.peaks  = []
        self.pulse_t0  =[]
    
    def plot_dice(self, i, ax=None, **kwargs):
        if ax is None: ax = plt.gca()
        audio = self.dices[i].to_numpy_array()
        time = np.linspace(0, audio.size*self.fs, audio.size)/ms
        
        t0 = self.dice_period*(i+0.5)
        ax.plot(time, audio, **kwargs)
        ax.set_xlabel('Time (ms)')
        ax.set_title(f'Dice {i} - Time {t0} ms')
        return ax

    def plot_dice_decompose(self, i, lo=1, hi=90, ax=None, **kwargs):
        if ax is None: ax = plt.gca()
        signal, bands = decompose_signal(self.dices[i], lo, hi)
        print(f'The First channel operates in {bands[0]:0.2f} kHz')
        print(f'The second channel operates in {bands[1]:0.2f} kHz')
        print(f'The Third channel operates in {bands[2]:0.2f} kHz')

        time = np.linspace(0, signal[0].size*self.fs, signal[0].size)/ms
        for i in range(3):
            ax.plot(time, signal[i], label=f'Ch {i+1} ({bands[i]:0.2f} kHz)', **kwargs)
        ax.legend()
        ax.set_xlabel('Time (ms)')
        ax.set_title(f'Decomposed signal in the band 1-90 kHz - Slice {i}')
        return ax
    
    def plot_dice_pulse(self, i, lo=1, hi=90, ax=None, **kwargs):
        fig, ax = plt.subplots(1,2, figsize=(12, 4), sharey=True)
        fig.subplots_adjust(wspace=0.02)

        signals, bands = decompose_signal(self.dices[i], lo, hi)
        signal = signals[0]
        times = np.linspace(0, signal.size*self.fs, signal.size)/ms
        signalF = lambda t: np.interp(t, times, signal)

        # pulse properties
        peaks = self.peaks[i]
        t0 = self.pulse_t0[i]
        tdelay = self.tdelay[i]
        
        ## firt pulse
        tinitial, tend = times[peaks[0]]-pulse_width/2., times[peaks[0]]+pulse_width/2.
        print(tinitial, tend)
        mask = (times>=tinitial) & (times<=tend)

        ax[1].plot(times[mask], signal[mask], color='k', label='Echo')
        ax[1].axvline(times[t0[0]], color='grey', ls='--')
        ax[1].axvline(times[peaks[0]], color='firebrick', ls='--')

        ## Second pulse
        tinitial, tend = tinitial+tdelay, tend+tdelay
        mask = (times>=tinitial) & (times<=tend)
        ax[0].plot(times[mask], signal[mask], color='k', label='Pulse')
        ax[0].axvline(times[t0[1]], color='grey', ls='--', label='Initial Time')
        ax[0].axvline(times[peaks[1]], color='firebrick', ls='--', label='Pulse Peak')

        # ax[1].set_title(f'Slice {i} - Time delay: {tdelay:0.2f} ms')

        ax[0].legend(fontsize=12)
        ax[1].legend(fontsize=12)
        ax[1].set_xlabel('Time (ms)')
        ax[0].set_xlabel('Time (ms)')
        pass

def measure_time_delay(signal, min_delay=15, max_delay=250, fs=1):
    """
    For a given signal, find the time delay between the two pulses
    
    input: 
        signal: numpy array
        min_delay: minimum delay in ms
        max_delay: maximum delay in ms
        fs: sampling frequency in Hz 

    return: 
        peaks: array, index location of the peaks
        pulset0: array, index location of the peaks
        tdelay: array, time delay in ms

    """
    peaks_loc, _ = find_peaks(np.abs(signal), height=np.nanmax(np.abs(signal))/10., distance=(min_delay*ms/fs))

    if peaks_loc.size < 2:
        tdelay = np.nan

    else:
        t0_loc = np.array([find_pulse_initial_time(signal, p, fs=fs) for p in peaks_loc])
        tdelay_t0 = (t0_loc[1]-t0_loc[0])*fs/ms
    
    tdelay = np.where(tdelay_t0>=max_delay, np.nan, tdelay_t0)
    return peaks_loc, t0_loc, tdelay

def find_pulse_initial_time(signal, peak, fs=1):
    idwidth = int(pulse_width/2 * ms / fs)
    # normalize signal to the peak
    y = np.abs(signal)/np.abs(signal)[peak]
    # cut around the pulse
    yn = np.full((y.size,), np.nan)
    yn[peak-idwidth:idwidth+peak] = y[peak-idwidth:idwidth+peak]
    
    # find the inital time  
    # defined as 1/e of the peak value
    id_t0 = get_inital_time(yn, th=1/np.exp(1))
    return id_t0

def get_inital_time(yn, th=0.5):
    mask = yn>th
    if np.count_nonzero(mask)>0:
        return np.where(mask)[0][0]
    else:
        return np.nan
    
def rank_and_filter_spectrum(spec, freq, amp_max = 1e12, amp_min = 1.):
    ## Rank the spectrum by the loudest channels
    amp_values = np.nanmax(np.abs(spec), axis=1)
    
    bad = np.where((amp_values > 1e12) | (amp_values < 1))[0]
    spec[bad] = np.nan
    
    ## Rank the spectrum by the loudest channels
    max_values = np.nanmax(np.abs(spec), axis=1)
    
    # Get the indices that would sort the channels by their maximum values in descending order
    sorted_indices = np.argsort(-1*np.log10(max_values))
    
    # Reorder the array based on the sorted indices
    spec_rank = spec[sorted_indices]
    return spec[sorted_indices], freq[sorted_indices]

def decompose_signal(audio, lo=0.5, hi=90, nfilters=5):
    """
    Decompose the signal in the band lo, hi
    For nfilter passbands
    input: 
        audio (pydub object)
        nfilter float number
        lo float number freq in kHz
        hi float number freq in kHz
    """
    signal0, freq0 = audio.filter_bank(nfilters=7, lower_bound_hz=lo*khz, upper_bound_hz=hi*khz)

    # filter high/low pitch values and rank the signal
    signal, bands = rank_and_filter_spectrum(signal0, freq0, amp_max = 1e12, amp_min = 1.)
    return signal, bands/1000