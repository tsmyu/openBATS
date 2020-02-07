from scipy import signal
from scipy import fftpack as spfft
import matplotlib.pyplot as plt
import numpy as np
from pylab import *

f0 = 2000
fs = int(1 / 5e-7)
sec = 4e-3
N = int(fs*sec)
addnum=5.0

def lowpassfilter(wave):
    sos = signal.iirfilter(N=1,
                        Wn=3000,
                        btype="lowpass",
                        analog=False,
                        ftype="butter",
                        output="sos",
                        fs=fs)
    return signal.sosfiltfilt(sos, wave)

    # return signal.sosfiltfilt(sos, wave, padlen=0)

def fft(X):

    X = spfft.fft(X[0:N])
    freqList = spfft.fftfreq(N, d=1.0/ fs)

    amplitude = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in X]  # 振幅スペクトル

    # 波形を描画
    subplot(211) 
    subplots_adjust(hspace=0.5)
    plot(range(0,N), X[0:N],label = "wave1")
    axis([0, N, -1.0, 1.0])
    xlabel("sample")
    ylabel("amplitude")

    # 振幅スペクトルを描画
    subplot(212)
    plot(freqList, amplitude, marker='.', linestyle='-',label = "fft plot")
    axis([0, fs / 60, 0, 10000])
    xlabel("frequency [Hz]")
    ylabel("amplitude")

    show()
def create_sin_wave(amplitude,f0,fs,sample):
    wave_table = []
    for n in np.arange(sample):
        sine = amplitude * np.sin(2.0 * np.pi * f0 * n / fs)
        wave_table.append(sine)
    return wave_table

wave1 = np.array(create_sin_wave(1.0, f0, fs, N))
wave2 = lowpassfilter(wave1)
# fft(wave1)
fft(wave2)

fs = 1 / 5e-7
sec = 4e-3
f0 = 2000
wave = []
for n in range(int(fs * sec)):
    s = np.sin(2.0 * np.pi * f0 * n / fs)
    wave.append(s)
wave = np.array(wave)
sos = signal.iirfilter(N=1,
                        Wn=3000,
                        btype="lowpass",
                        analog=False,
                        ftype="butter",
                        output="sos",
                        fs=fs)

l_wave = signal.sosfiltfilt(sos, wave, padlen=0)

f_wave = fftpack.fft(wave)[int(len(wave)/2):].real
fl_wave = fftpack.fft(l_wave)[int(len(l_wave) / 2):].real
fft_fre = fftpack.fftfreq(n=f_wave.size, d=1/fs)[int(len(wave)/2):]
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax1.plot(wave)
ax2.plot(l_wave)
ax3.plot( f_wave)
ax4.plot(fl_wave)
plt.show()




max_freq = 100e3
min_freq = 20e3
bin_num = 81
bandwidth = 4e3

fs = 1 / 5.0e-7
sec = 1
f0 = 2e3
a = 1
x = []
for n in np.arange(fs * sec):
    #サイン波を生成
    s = a * np.sin(2.0 * np.pi * f0 * n / fs)
    x.append(s)

x = np.array(x)

for i in range(bin_num):
    hz = i*1e3
    # sos = signal.iirfilter(N=10,
    #                        Wn=[min_freq + hz - bandwidth / 2, min_freq + hz + bandwidth / 2],
    #                        btype="bandpass",
    #                        analog=False,
    #                        ftype="butter",
    #                        output="sos",
    #                        fs=1 / 5.0e-7)
    sos = signal.iirfilter(N=1,
                            Wn=3e3,
                            btype="lowpass",
                            analog=False,
                            ftype="butter",
                            output="sos",
                            fs=1 / 5.0e-7)
    s_f = signal.sosfiltfilt(sos,x)

    plt.plot(s_f)
    plt.show()
    plt.plot(sos)
    plt.show()
    plt.plot(fftpack.fft(x))
    plt.show()
    plt.plot(fftpack.fft(s_f))
    plt.show()
    
    w, h = signal.sosfreqz(sos, 1000000)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    db = 20 * np.log10(np.maximum(np.abs(h), 1e-5))
    plt.plot(w/ np.pi, db)
    # plt.xlim(15e3*5.0e-7, 25e3*5.0e-7)
    plt.show()
