# ********************************************************************************
# SetName: {grid_name}
# Author: Roy Stegeman
# Date: January 2025
# CodesUsed: MATRIX
# TheoryInput:
# PDFset: 240517-rs-alphas_01200
# Warnings: top mass variation mt={mtop} GeV
# ********************************************************************************
# 62.8146430047735	9.98013555246471
# 24.5375193467959	3.17461745004482
# 18.8189325704493	2.60267835145856
# 21.6193749733216	4.00251399375547
# 33.811164397667	8.78985824548696
# 67.7702935941926	23.5981889612153

import lhapdf
import pineappl
import numpy as np

pdfset = "240517-rs-alphas_01200"
pdf = lhapdf.mkPDF(pdfset, 0)
gridlist = [
    # ['ATLASTTBARTOT7TEV-TOPDIFF7TEVTOT', 'ATLAS_TTBAR_7TEV_TOT_X-SEC'],
    # ['ATLASTTBARTOT8TEV-TOPDIFF8TEVTOT', 'ATLAS_TTBAR_8TEV_TOT_X-SEC'],
    # ['ATLAS_TTBARTOT_13TEV_FULLLUMI-TOPDIFF13TEVTOT', 'ATLAS_TTBAR_13TEV_TOT_X-SEC'],
    # ['CMSTTBARTOT7TEV-TOPDIFF7TEVTOT', 'CMS_TTBAR_7TEV_TOT_X-SEC'],
    # ['CMSTTBARTOT8TEV-TOPDIFF8TEVTOT', 'CMS_TTBAR_8TEV_TOT_X-SEC'],
    # ['CMSTTBARTOT13TEV-TOPDIFF13TEVTOT', 'CMS_TTBAR_13TEV_TOT_X-SEC'],
    # ['CMS_TTB_5TEV_TOT', 'CMS_TTBAR_5TEV_TOT_X-SEC'],
    ['CMS_TTB_13TEV_2L_TRAP', 'CMS_TTBAR_13TEV_2L_DIF_YT'],

    # RATIOs
    ['ATLAS_TTB_8TEV_LJ_TRAP', 'ATLAS_TTBAR_8TEV_LJ_DIF_YT'],
    ['ATLAS_TTB_8TEV_LJ_TRAP_TOT', 'ATLAS_TTBAR_8TEV_LJ_DIF_YT-INTEGRATED'],

    ['ATLAS_TTB_8TEV_LJ_TTRAP', 'ATLAS_TTBAR_8TEV_LJ_DIF_YTTBAR'],
    ['ATLAS_TTB_8TEV_LJ_TTRAP_TOT', 'ATLAS_TTBAR_8TEV_LJ_DIF_YTTBAR-INTEGRATED'],

    # ['ATLAS_TTB_8TEV_2L_TTRAP', 'ATLAS_TTBAR_8TEV_2L_DIF_YTTBAR'],
    # ['CMSTTBARTOT8TEV-TOPDIFF8TEVTOT', 'ATLAS_TTBAR_8TEV_2L_DIF_YTTBAR-INTEGRATED'],

    # ['CMS_TTB_8TEV_2D_TTM_TRAP', 'CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YT'],
    # ['CMS_TTB_8TEV_2D_TTM_TRAP_TOT', 'CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YT-INTEGRATED'],

    # ['CMSTOPDIFF8TEVTTRAPNORM-TOPDIFF8TEVTTRAP', 'CMS_TTBAR_8TEV_LJ_DIF_YTTBAR'],
    # ['CMSTOPDIFF8TEVTTRAPNORM-TOPDIFF8TEVTOT', 'CMS_TTBAR_8TEV_LJ_DIF_YTTBAR-INTEGRATED'],
]

gridpath = "/Users/roy/github/NNPDF/theories_slim/data/grids"

for old_name, grid_name in gridlist:
    central_grid = pineappl.grid.Grid.read(f"{gridpath}/40009000/{grid_name}.pineappl.lz4")
    plus_grid = pineappl.grid.Grid.read(f"{gridpath}/40009002/{grid_name}.pineappl.lz4")
    minus_grid = pineappl.grid.Grid.read(f"{gridpath}/40009001/{grid_name}.pineappl.lz4")

    central_nlo_pred = central_grid.convolve_with_one(2212, pdf.xfxQ2, pdf.alphasQ2, order_mask=np.array([1,1,0,0,0,0,0,0,0,0,0,0],dtype=bool))
    plus_nlo_pred = plus_grid.convolve_with_one(2212, pdf.xfxQ2, pdf.alphasQ2, order_mask=np.array([1,1,0,0,0,0,0,0,0,0,0,0],dtype=bool))
    minus_nlo_pred = minus_grid.convolve_with_one(2212, pdf.xfxQ2, pdf.alphasQ2, order_mask=np.array([1,1,0,0,0,0,0,0,0,0,0,0],dtype=bool))

    print(f"{grid_name}")
    print(plus_nlo_pred/central_nlo_pred)
    print(minus_nlo_pred/central_nlo_pred)

    def header(old_name, mtop):
        return f"""********************************************************************************
SetName: {old_name}
Author: Roy Stegeman
Date: January 2025
CodesUsed: MATRIX
TheoryInput: NLO
PDFset: 240517-rs-alphas_01200
Warnings: top mass variation mt={mtop} GeV
********************************************************************************
"""

    with open(f"./cfactor/CF_TOPP_{old_name}.dat", "w") as file:
        content = header(old_name, 175.0)
        file.write(content)
        for value in plus_nlo_pred/central_nlo_pred:
            file.write(f"{value:.4f}\t0.0000\n")

    with open(f"./cfactor/CF_TOPM_{old_name}.dat", "w") as file:
        content = header(old_name, 170.0)
        file.write(content)
        for value in minus_nlo_pred/central_nlo_pred:
            file.write(f"{value:.4f}\t0.0000\n")
