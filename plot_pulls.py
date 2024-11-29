import matplotlib.pyplot as plt
import numpy as np


# DATA ********************************************************************************************

# the alphas value and corresponding unceratinties based on partial chi2s for various fit settings

nnlo = [
    "NNLO",
    {
        "ALL": (0.1202, 0.0003),
        "DIS NC": (0.1200, 0.0004),
        "DIS CC": (0.1228, 0.0010),
        "DY NC": (0.1204, 0.0003),
        "DY CC": (0.1189, 0.0005),
        "TOP": (0.1210, 0.0012),
        "JETS": (0.1204, 0.0007),
        "DIJET": (0.1198, 0.0009),
        "PHOTON": (0.1213, 0.0009),
        "SINGLETOP": (0.1201, 0.0020),
    }
]

nnlo_qed = [
    "NNLOxQED",
    {
        "ALL": (0.1209, 0.0003),
        "DIS NC": (0.1209, 0.0004),
        "DIS CC": (0.1235, 0.0009),
        "DY NC": (0.1209, 0.0003),
        "DY CC": (0.1196, 0.0005),
        "TOP": (0.1222, 0.0012),
        "JETS": (0.1210, 0.0007),
        "DIJET": (0.1205, 0.0008),
        "PHOTON": (0.1215, 0.0008),
        "SINGLETOP": (0.1211, 0.0020),
    }
]

nnlo_mhou = [
    "NNLO, MHOU",
    {
        "ALL": (0.1194, 0.0007),
        "DIS NC": (0.1194, 0.0008),
        "DIS CC": (0.1227, 0.0011),
        "DY NC": (0.1195, 0.0007),
        "DY CC": (0.1183, 0.0008),
        "TOP": (0.1218, 0.0015),
        "JETS": (0.1187, 0.0010),
        "DIJET": (0.1171, 0.0011),
        "PHOTON": (0.1204, 0.0010),
        "SINGLETOP": (0.1201, 0.0020),
    }
]

nnlo_mhou_qed = [
    "NNLOxQED, MHOU",
    {
        "ALL": (0.1202, 0.0006),
        "DIS NC": (0.1203, 0.0006),
        "DIS CC": (0.1235, 0.0011),
        "DY NC": (0.1199, 0.0006),
        "DY CC": (0.1193, 0.0007),
        "TOP": (0.1221, 0.0015),
        "JETS": (0.1195, 0.0009),
        "DIJET": (0.1174, 0.0011),
        "PHOTON": (0.1211, 0.0009),
        "SINGLETOP": (0.1201, 0.0020),
    }
]

n3lo_3pt = [
    "N3LO",
    {
        "ALL": (0.1188, 0.0005),
        "DIS NC": (0.1188, 0.0006),
        "DIS CC": (0.1221, 0.0010),
        "DY NC": (0.1188, 0.0005),
        "DY CC": (0.1177, 0.0006),
        "TOP": (0.1209, 0.0015),
        "JETS": (0.1181, 0.0009),
        "DIJET": (0.1172, 0.0010),
        "PHOTON": (0.1201, 0.0009),
        "SINGLETOP": (0.1191, 0.0020),
    }
]

n3lo_3pt_qed = [
    "N3LOxQED",
    {
        "ALL": (0.1187, 0.0005),
        "DIS NC": (0.1186, 0.0005),
        "DIS CC": (0.1220, 0.0010),
        "DY NC": (0.1187, 0.0005),
        "DY CC": (0.1178, 0.0006),
        "TOP": (0.1208, 0.0015),
        "JETS": (0.1184, 0.0009),
        "DIJET": (0.1168, 0.0010),
        "PHOTON": (0.1202, 0.0009),
        "SINGLETOP": (0.1191, 0.0020),
    }
]

n3lo_mhou = [
    "N3LO, MHOU",
    {
        "ALL": (0.1192, 0.0007),
        "DIS NC": (0.1189, 0.0007),
        "DIS CC": (0.1229, 0.0011),
        "DY NC": (0.1191, 0.0007),
        "DY CC": (0.1181, 0.0008),
        "TOP": (0.1226, 0.0015),
        "JETS": (0.1190, 0.0010),
        "DIJET": (0.1174, 0.0011),
        "PHOTON": (0.1207, 0.0010),
        "SINGLETOP": (0.1201, 0.0020),
    }
]

n3lo_qed_mhou = [
    "N3LOxQED, MHOU",
    {
        "ALL": (0.1194, 0.0007),
        "DIS NC": (0.1191, 0.0007),
        "DIS CC": (0.1231, 0.0011),
        "DY NC": (0.1192, 0.0007),
        "DY CC": (0.1183, 0.0008),
        "TOP": (0.1227, 0.0015),
        "JETS": (0.1192, 0.0010),
        "DIJET": (0.1176, 0.0011),
        "PHOTON": (0.1209, 0.0010),
        "SINGLETOP": (0.1201, 0.0020),
    }
]



# Which to plot ***********************************************************************************
fitdata = nnlo_mhou

# Do the plot *************************************************************************************
plotlabel = fitdata[0]
data = fitdata[1]

labels = list(data.keys())[::-1]
central_values = [item[0] for item in data.values()][::-1]
uncertainties = [item[1] for item in data.values()][::-1]

y_positions = np.arange(len(labels))

plt.figure(figsize=(10, 6))
plt.errorbar( central_values, y_positions, xerr=uncertainties, fmt='o', color='teal', label='1std' )
plt.axvline( x=data["ALL"][0], color='teal', linestyle='--', label=r'Reference $\alpha_s(M_Z)$' )

plt.yticks(y_positions, labels)
plt.xlabel(r"$\alpha_s(M_Z)$")
plt.title(plotlabel)
# plt.legend()
plt.grid(axis='x', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("pulls.pdf")
