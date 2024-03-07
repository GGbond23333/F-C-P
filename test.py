import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import argparse


# Lotka-Volterra predator-prey model
def lotka_volterra(state, t, alpha, beta, delta, gamma):
    x, y = state
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]


# Command line argument parsing
parser = argparse.ArgumentParser(description="Solve the Lotka-Volterra predator-prey model.")
parser.add_argument('--initial', type=float, nargs=2, default=[10, 5], help='Initial population sizes for prey and predator.')
parser.add_argument('--alpha', type=float, nargs='*', default=[0.1, 0.2], help='Prey birth rate.')
parser.add_argument('--beta', type=float, default=0.2, help='Predation rate coefficient.')
parser.add_argument('--delta', type=float, default=0.01, help='Growth rate of predator per captured prey.')
parser.add_argument('--gamma', type=float, default=0.3, help='Predator death rate.')
parser.add_argument('--save_plot', action='store_true', help='Save the plot instead of displaying it.')

args = parser.parse_args()

# Time vector
t = np.linspace(0, 200, 1000)

# Solving the system of equations
solutions = []
for alpha_val in args.alpha:
    solution = odeint(lotka_volterra, args.initial, t, args=(alpha_val, args.beta, args.delta, args.gamma))
    solutions.append(solution)

# Plotting the results
if not args.save_plot:
    for idx, solution in enumerate(solutions):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(t, solution[:, 0], label=f'Prey (alpha={args.alpha[idx]})')
        plt.title('Prey Population over Time')
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.grid()
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(t, solution[:, 1], label='Predator')
        plt.title('Predator Population over Time')
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.show()
else:
    for idx, solution in enumerate(solutions):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(t, solution[:, 0], label=f'Prey (alpha={args.alpha[idx]})')
        plt.title('Prey Population over Time')
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.grid()
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(t, solution[:, 1], label='Predator')
        plt.title('Predator Population over Time')
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.grid()
        plt.legend()

        plt.tight_layout()
        filename = f'lotka_volterra_alpha_{args.alpha[idx]}.png'
        plt.savefig(filename)
        print(f'Saved plot to {filename}')







