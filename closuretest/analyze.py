import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Load the central values and uncertainties from the .csv files
alphas_values = []
alphas_uncertainties = []
for i in range(1, 101):
    filename = f"/data/theorie/rstegeman/github/nnpdf40_alphas/closuretest/results/{i}.csv"
    # Open each file and read its contents
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Extract the alphas value and uncertainty from each row
            alphas_values.append(float(row[1]))
            alphas_uncertainties.append(float(row[2]))


# Calculate and print average and standard deviation of alphas values
average_alphas = np.mean(alphas_values)
std_dev_alphas = np.std(alphas_values)
average_uncertainty = np.mean(alphas_uncertainties)
print(f"mean and std of 100 central alphas values: {average_alphas:.6f} ± {std_dev_alphas:.6f}")
print(f"mean of 100 alphas uncertainties: {average_uncertainty:.6f}")

# Plot results + uncs
indices = np.arange(len(alphas_values)) + 1
plt.errorbar(indices, alphas_values, yerr=alphas_uncertainties, fmt='o')
plt.axhline(y=0.118, color='r', linestyle='--')
plt.ylabel(r'$\alpha_s(m_Z)$')
plt.title('TCM')
plt.tight_layout()
plt.savefig('/data/theorie/rstegeman/github/nnpdf40_alphas/closuretest/closuretest_results.pdf')

# Plot the histogram of alphas values
plt.figure(figsize=(8, 5))
# Histogram
bins = [0.1176 + 0.0002 * n for n in range(8)]
plt.hist(
    alphas_values,
    bins=bins,
    color='blue',
    alpha=0.5,
    density=False
)
# Normal distribution fit
mean = np.mean(alphas_values)
std = np.std(alphas_values)
x = np.linspace(0.117, 0.119, 100)
normpdf = norm.pdf(x, mean, std)
normpdf_scaled = normpdf * len(alphas_values) * (bins[1] - bins[0])
plt.plot(x, normpdf_scaled, color='blue')
# Layout
plt.title('TCM')
plt.axvline(x=0.118, color='gray', linestyle='--')
plt.xlabel(r"$\alpha_s(m_Z)$")
plt.xlim(x.min(), x.max())
plt.ylabel("Counts")
plt.tight_layout()
plt.savefig('/data/theorie/rstegeman/github/nnpdf40_alphas/closuretest/closuretest_hist.pdf')


# mean alphas value and the bootstrap uncertainty on it
bootstrap_means = []
for _ in range(1000):
    resampled_values = np.random.choice(alphas_values, size=len(alphas_values), replace=True)
    bootstrap_means.append(np.mean(resampled_values))
lower_bound = np.percentile(bootstrap_means, 16)
upper_bound = np.percentile(bootstrap_means, 84)
print(f"68% C.I. of bootstrap of mean of alphas determinations: {(lower_bound + upper_bound) / 2:.6f} ± {(upper_bound - lower_bound) / 2:.6f}")

# Weighted average calculation
weights = [1 / (uncertainty**2) for uncertainty in alphas_uncertainties]
weighted_average_alphas = sum(value * weight for value, weight in zip(alphas_values, weights)) / sum(weights)
weighted_uncertainty = 1 / (sum(weights)**0.5)
print(f"Weighted average of alphas determinations: {weighted_average_alphas:.6f} ± {weighted_uncertainty:.6f}")

# Rbv
rbv_samples = []
for _ in range(1000):
    resampled_indices = np.random.choice(len(alphas_values), len(alphas_values), replace=True)
    resampled_rbv = 0
    for i in resampled_indices:
        val = alphas_values[i]
        unc = alphas_uncertainties[i]
        resampled_rbv += ((val - 0.118) ** 2 / unc ** 2)
    resampled_rbv = resampled_rbv / len(alphas_values)
    resampled_rbv = resampled_rbv ** 0.5
    rbv_samples.append(resampled_rbv)
ci_lower = np.percentile(rbv_samples, 16)
ci_upper = np.percentile(rbv_samples, 84)
print(f"Rbv: {(ci_lower+ci_upper)/2:.6f} ± {(-ci_lower+ci_upper)/2:.6f}")
