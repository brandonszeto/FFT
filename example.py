import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [16, 12]
plt.rcParams.update({'font.size': 18})

# simple signal with two frequencies
dt = 0.001
t = np.arange(0,1,dt)
f = np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t)
f_clean = f
f = f + 2.5*np.random.randn(len(t))

# compute fast fourier transform (FFT)
n = len(t)
fhat = np.fft.fft(f,n)                       # compute fft
PSD = fhat * np.conj(fhat) / n               # power spectrum (magnitude/n)
freq = (1/(dt*n)) * np.arange(n)             # create x-axis of frequencies
L = np.arange(1, np.floor(n/2), dtype='int') # plot the first half

fig,axs = plt.subplots(3,1)

# filter noise out using PSD
indices = PSD > 100 # find all freq with power > 50
PSDclean = PSD * indices  # zero out all others
fhat = indices * fhat     # zero out unimportant fourier coefficients in Y
ffilt = np.fft.ifft(fhat) # inverse fft for filtered time signal

# plot noisy and clean signal
plt.sca(axs[0])
plt.plot(t,f,color='c', label='Noisy')
plt.plot(t,f_clean,color='k', label='Clean')
plt.xlim(t[0],t[-1])
plt.legend()

# plot clean signal
plt.sca(axs[1])
plt.plot(t, ffilt, color='k', label='Filtered')
plt.xlim(t[0], t[-1])
plt.legend()

# plot clean PSD
plt.sca(axs[2])
# plt.plot(freq[L], PSD[L], color='c', label='Noisy')
plt.plot(freq[L], PSDclean[L], color='k', label='Filtered')
plt.xlim(freq[L[0]], freq[L[-1]])
plt.legend()

plt.show()


