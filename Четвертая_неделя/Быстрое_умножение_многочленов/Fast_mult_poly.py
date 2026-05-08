import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

#Быстрое умножение многочленов с помощью FFT
def fast_multiplication_poly(p, q):
    n = len(p)
    m = len(q)
    size = n + m - 1

    N = 1
    while N < size:
        N *= 2

    p = np.concatenate((p, np.zeros(N - n)))
    q = np.concatenate((q, np.zeros(N - m)))

    P = np.fft.fft(p)
    Q = np.fft.fft(q)

    C = P * Q

    coeff = np.fft.ifft(C)

    coeff = np.real(coeff)

    return coeff[:size]

#Обычное перемножение многочленов
def naive_multiplication_poly(p, q):
    n = len(p)
    m = len(q)

    result = np.zeros(n + m - 1)

    for i in range(n):
        for j in range(m):
            result[i + j] += p[i] * q[j]

    return result

T_fast = []
T_slow = []

sizes_slow = np.unique(np.logspace(1, 3, 40, dtype = int))
sizes_fast = np.unique(np.logspace(1, 7, 40, dtype = int))

for n in tqdm(sizes_fast, desc="FFT"):

    p = np.random.randint(0, 10, n)
    q = np.random.randint(0, 10, n)

    runs = 5

    start = time.perf_counter()

    for _ in range(runs):
        fast_multiplication_poly(p, q)

    end = time.perf_counter()

    T_fast.append((end - start) / runs)

T_fast = np.array(T_fast)

theory_fast = sizes_fast * np.log(sizes_fast)
theory_fast = theory_fast * T_fast[-1] / theory_fast[-1]

for n in tqdm(sizes_slow, desc="Naive"):

    p = np.random.randint(0, 10, n)
    q = np.random.randint(0, 10, n)

    runs = 5

    start = time.perf_counter()

    for _ in range(runs):
        naive_multiplication_poly(p, q)

    end = time.perf_counter()

    T_slow.append((end - start) / runs)

theory_slow = sizes_slow ** 2
theory_slow = theory_slow * T_slow[-1] / theory_slow[-1]


theory_slow = sizes_slow ** 2
theory_slow = theory_slow * T_slow[-1] / theory_slow[-1]

fig, ax = plt.subplots(1, 2, figsize=(12,5))

# FFT
ax[0].loglog(sizes_fast, T_fast, label='FFT', color = 'r')
ax[0].loglog(sizes_fast, theory_fast, '--', label=r'$n \log n$', color = 'b')

ax[0].set_title("FFT")
ax[0].set_xlabel("Степень многочлена")
ax[0].set_ylabel("Время")

ax[0].legend()
ax[0].grid()

# Naive
ax[1].loglog(sizes_slow, T_slow, label='Naive', color = 'r')
ax[1].loglog(sizes_slow, theory_slow, '--', label=r'$n^2$', color = 'b')

ax[1].set_title("Naive multiplication")
ax[1].set_xlabel("Степень многочлена")
ax[1].set_ylabel("Время")

ax[1].legend()
ax[1].grid()

plt.tight_layout()
plt.savefig('D:/Fast_mult_poly/LaTex/Comparison.png')
plt.show()
