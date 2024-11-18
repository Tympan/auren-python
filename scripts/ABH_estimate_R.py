import numpy as np
from matplotlib import pyplot as plt

alpha_1 = lambda k0L: -0.5 + np.sqrt(0.25 - k0L **2 + 0j)
R = lambda k0L: (1 - alpha_1(k0L) / (1j * k0L)) / (1 + alpha_1(k0L) / (1j * k0L)) * np.exp(-2j*k0L)
R = lambda k0L: (1 + alpha_1(k0L) / (1j * k0L)) / (1 - alpha_1(k0L) / (1j * k0L)) * np.exp(-2j*k0L)

f = np.linspace(100, 20000)
c = 343
k = 2 * np.pi * f / c
L1 = 2 / 1000
L2 = 20 / 1000

# k0L = np.linspace(0, 4, 256)

plt.semilogx(f, np.abs(R(k*L1 + 0.05j)), label=r"2mm, $\alpha=0.05$")
plt.semilogx(f, np.abs(R(k*L1*2 + 0.05j)), label=r"4mm, $\alpha=0.05$")
plt.semilogx(f, np.abs(R(k*L1 + 2*0.05j)), label=r"2mm, $\alpha=0.1$")
plt.semilogx(f, np.abs(R(k*L1 + 0.5j)), label=r"2mm, $\alpha=0.5$")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Reflection coefficient")
plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.show()

