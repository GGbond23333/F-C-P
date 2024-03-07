import math


class Vector:
    def __init__(self, values):
        self.values = values

    def length(self):
        return math.sqrt(sum(x ** 2 for x in self.values))

    def add(self, other):
        if len(self.values) != len(other.values):
            raise ValueError("Vectors must have the same dimension")
        return [x + y for x, y in zip(self.values, other.values)]

    def dot_product(self, other):
        if len(self.values) != len(other.values):
            raise ValueError("Vectors must have the same dimension")
        return sum(x * y for x, y in zip(self.values, other.values))


def test_vectors():
    vector1 = Vector([3, 4])
    vector2 = Vector([1, 2])
    assert vector1.length() == 5
    assert vector2.length() == math.sqrt(5)
    assert vector1.add(vector2) == [4, 6]
    assert vector1.dot_product(vector2) == 11


def main():
    test_vectors()


if __name__ == "__main__":
    main()
    print('success')


import numpy as np
from scipy.integrate import odeint 
import matplotlib.pyplot as plt 


def gradient(N, t, alpha, beta):
    return -alpha * N + beta * N


N0 = 100
alpha_values = [0.1, 0.2, 0.3]
beta_values = [0.05, 0.1, 0.15]
t = np.linspace(0, 100)

fig, axs = plt.subplots(3, 3, figsize=(12, 9))

for i, alpha in enumerate(alpha_values):
    for j, beta in enumerate(beta_values):
        N = odeint(gradient, N0, t, (alpha, beta))
        ax = axs[i, j]
        ax.plot(t, N)
        ax.set_title(f"Alpha={alpha}, Beta={beta}")
        ax.set_ylabel("Population size (N people)")
        ax.set_xlabel("Time (months)")

plt.tight_layout()
plt.show()

if __name__ == '__main__':
    main()
