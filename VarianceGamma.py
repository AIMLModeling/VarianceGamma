import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from functools import partial
from scipy.optimize import minimize
from scipy.integrate import quad
import scipy.special as scps

T = 2  # terminal time
N = 1000000  # number of generated random variables

theta = -0.1  # drift of the Brownian motion
sigma = 0.2  # volatility of the Brownian motion
kappa = 0.5  # variance of the Gamma process
np.random.seed(seed=42)
G = ss.gamma(T / kappa, scale=kappa).rvs(N)  # The gamma RV
Norm = ss.norm.rvs(0, 1, N)  # The normal RV
X = theta * G + sigma * np.sqrt(G) * Norm
def VG_density(x, T, c, theta, sigma, kappa):
    return (
        2
        * np.exp(theta * (x - c) / sigma**2)
        / (kappa ** (T / kappa) * np.sqrt(2 * np.pi) * sigma * scps.gamma(T / kappa))
        * ((x - c) ** 2 / (2 * sigma**2 / kappa + theta**2)) ** (T / (2 * kappa) - 1 / 4)
        * scps.kv(T / kappa - 1 / 2, sigma ** (-2) * np.sqrt((x - c) ** 2 * (2 * sigma**2 / kappa + theta**2)))
    )
def cf_VG(u, t=1, mu=0, theta=-0.1, sigma=0.2, kappa=0.1):
    """
    Characteristic function of a Variance Gamma random variable at time t
    mu: additional drift
    theta: Brownian motion drift
    sigma: Brownian motion diffusion
    kappa: Gamma process variance
    """
    return np.exp(t * (1j * mu * u - np.log(1 - 1j * theta * kappa * u + 0.5 * kappa * sigma**2 * u**2) / kappa))
# Gil-Pelaez PDF Inversion
def Gil_Pelaez_pdf(x, cf, right_lim):
    """
    Gil Pelaez formula for the inversion of the characteristic function
    INPUT
    - x: is a number
    - right_lim: is the right extreme of integration
    - cf: is the characteristic function
    OUTPUT
    - the value of the density at x.
    1.	Computes the PDF by inverting the characteristic function using the Gil-Pelaez theorem.
    2.	quad: Performs numerical integration from 00 to right_lim.
    """
    def integrand(u):
        return np.real(np.exp(-u * x * 1j) * cf(u))
    return 1 / np.pi * quad(integrand, 1e-15, right_lim)[0]

cf_VG_b = partial(cf_VG, t=T, mu=0, theta=theta, sigma=sigma, kappa=kappa)
x = np.linspace(X.min(), X.max(), 500)
y = np.linspace(-2, 1, 30)

plt.figure(figsize=(16, 5))
plt.plot(x, VG_density(x, T, 0, theta, sigma, kappa), color="r", label="VG density")
plt.plot(y, [Gil_Pelaez_pdf(i, cf_VG_b, np.inf) for i in y], "p", label="Fourier inversion")
plt.hist(X, density=True, bins=200, facecolor="LightBlue", label="frequencies of X")
plt.legend()
plt.title("Variance Gamma Histogram")
plt.show()
qqplot(X, line="s")
plt.show()
sigma_mm1 = np.std(X) / np.sqrt(T)
kappa_mm1 = T * ss.kurtosis(X) / 3
theta_mm1 = np.sqrt(T) * ss.skew(X) * sigma_mm1 / (3 * kappa_mm1)
c_mm1 = np.mean(X) / T - theta_mm1

print(
    "Estimated parameters: \n\n c={} \n theta={} \n sigma={} \n \
kappa={}\n".format(
        c_mm1, theta_mm1, sigma_mm1, kappa_mm1
    )
)
print("Estimated c + theta = ", c_mm1 + theta_mm1)
def log_likely(x, data, T):
    return (-1) * np.sum(np.log(VG_density(data, T, x[0], x[1], x[2], x[3])))


result_VG = minimize(
    log_likely,
    x0=[c_mm1, theta_mm1, sigma_mm1, kappa_mm1],
    method="L-BFGS-B",
    args=(X, T),
    tol=1e-8,
    bounds=[[-1, 1], [-1, -1e-15], [1e-15, 2], [1e-15, None]],
)

print(result_VG.message)
print("Number of iterations performed by the optimizer: ", result_VG.nit)
print("MLE parameters: ", result_VG.x)
