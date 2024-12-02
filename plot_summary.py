import matplotlib.pyplot as plt
import numpy as np

# DATA ********************************************************************************************

# the alphas value and corresponding unceratinties based on partial chi2s for various fit settings
data = {
    # "NNLO": (0.1204, 0.0004),
    # "NNLOxQED": (0.1210, 0.0004),
    # "NNLO, MHOU": (0.1194, 0.0007),
    "NNLOxQED, MHOU": (0.1201, 0.0006),
    # "aN3LO": (0.1190, 0.0006),
    # "aN3LOxQED": (0.1187, 0.0005),
    # "aN3LO, MHOU": (0.1194, 0.0007),
    "aN3LOxQED, MHOU": (0.1194, 0.0007),
    # "NNPDF3.1": (0.1185, 0.0006),
    "NNPDF3.1": (0.1185, 0.0012),
    # "NNPDF3.1, MHOU": (0.1182, 0.0008),
    "PDG 2023": (0.1180, 0.0009),
    "MSHT NNLO": (0.1171, 0.0014),
    "MSHT aN3LO": (0.1170, 0.0016),
}

labels = list(data.keys())[::-1]
central_values = [item[0] for item in data.values()][::-1]
uncertainties = [item[1] for item in data.values()][::-1]

y_positions = np.arange(len(labels))

# Plot setup
plt.figure(figsize=(len(data), 6/10*len(data)))
plt.errorbar(central_values, y_positions, xerr=uncertainties, fmt='o', color='teal', label='1std')
plt.axvline( x=data["PDG 2023"][0], color='teal', linestyle='--', label=r'Reference $\alpha_s(M_Z)$' )
plt.axvspan(data["PDG 2023"][0] - data["PDG 2023"][1], data["PDG 2023"][0] + data["PDG 2023"][1], color='teal', alpha=0.2, label='PDG uncertainty')

plt.yticks(y_positions, labels)
plt.xlabel(r"$\alpha_s(M_Z)$")
plt.grid(axis='x', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("summary.pdf")
