import numpy as np
import pandas as pd
import copy

from validphys.api import API

fitname = "240517-rs-alphas_01200"

dataset_inputs = {
    "dataset_inputs": [
        {
            "dataset": "NMC_NC_NOTFIXED_DW_EM-F2",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "NMC_NC_NOTFIXED_P_EM-SIGMARED",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "SLAC_NC_NOTFIXED_P_DW_EM-F2",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "SLAC_NC_NOTFIXED_D_DW_EM-F2",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "BCDMS_NC_NOTFIXED_P_DW_EM-F2",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "BCDMS_NC_NOTFIXED_D_DW_EM-F2",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "CHORUS_CC_NOTFIXED_PB_DW_NU-SIGMARED",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "CHORUS_CC_NOTFIXED_PB_DW_NB-SIGMARED",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "cfac": ["MAS"],
            "dataset": "NUTEV_CC_NOTFIXED_FE_DW_NU-SIGMARED",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "cfac": ["MAS"],
            "dataset": "NUTEV_CC_NOTFIXED_FE_DW_NB-SIGMARED",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "HERA_NC_318GEV_EM-SIGMARED",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "HERA_NC_225GEV_EP-SIGMARED",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "HERA_NC_251GEV_EP-SIGMARED",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "HERA_NC_300GEV_EP-SIGMARED",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "HERA_NC_318GEV_EP-SIGMARED",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "HERA_CC_318GEV_EM-SIGMARED",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "HERA_CC_318GEV_EP-SIGMARED",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "HERA_NC_318GEV_EAVG_CHARM-SIGMARED",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "HERA_NC_318GEV_EAVG_BOTTOM-SIGMARED",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "DYE866_Z0_800GEV_DW_RATIO_PDXSECRATIO",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "DYE866_Z0_800GEV_PXSEC",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "DYE605_Z0_38P8GEV_DW_PXSEC",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "cfac": ["ACC"],
            "dataset": "DYE906_Z0_120GEV_DW_PDXSECRATIO",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "CDF_Z0_1P96TEV_ZRAP",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "D0_Z0_1P96TEV_ZRAP",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "D0_WPWM_1P96TEV_ASY",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "ATLAS_WPWM_7TEV_36PB_ETA",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "ATLAS_Z0_7TEV_36PB_ETA",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "ATLAS_Z0_7TEV_49FB_HIMASS",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "ATLAS_Z0_7TEV_LOMASS_M",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "ATLAS_WPWM_7TEV_46FB_CC-ETA",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "ATLAS_Z0_7TEV_46FB_CC-Y",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "ATLAS_Z0_7TEV_46FB_CF-Y",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "ATLAS_Z0_8TEV_HIMASS_M-Y",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "ATLAS_Z0_8TEV_LOWMASS_M-Y",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "cfac": ["NRM"],
            "dataset": "ATLAS_Z0_13TEV_TOT",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "cfac": ["NRM"],
            "dataset": "ATLAS_WPWM_13TEV_TOT",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "ATLAS_WJ_JET_8TEV_WP-PT",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "ATLAS_WJ_JET_8TEV_WM-PT",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "ATLAS_Z0J_8TEV_PT-M",
            "frac": 0.75,
            "variant": "legacy_10",
        },
        {
            "dataset": "ATLAS_Z0J_8TEV_PT-Y",
            "frac": 0.75,
            "variant": "legacy_10",
        },
        {
            "dataset": "ATLAS_TTBAR_7TEV_TOT_X-SEC",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "ATLAS_TTBAR_8TEV_TOT_X-SEC",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "ATLAS_TTBAR_13TEV_TOT_X-SEC",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "ATLAS_TTBAR_8TEV_LJ_DIF_YT-NORM",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "ATLAS_TTBAR_8TEV_LJ_DIF_YTTBAR-NORM",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "ATLAS_TTBAR_8TEV_2L_DIF_YTTBAR-NORM",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "ATLAS_1JET_8TEV_R06_PTY",
            "frac": 0.75,
            "variant": "legacy_decorrelated",
        },
        {
            "dataset": "ATLAS_2JET_7TEV_R06_M12Y",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "cfac": ["EWK"],
            "dataset": "ATLAS_PH_13TEV_XSEC",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "ATLAS_SINGLETOP_7TEV_TCHANNEL-XSEC",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "ATLAS_SINGLETOP_13TEV_TCHANNEL-XSEC",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "ATLAS_SINGLETOP_7TEV_T-Y-NORM",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "ATLAS_SINGLETOP_7TEV_TBAR-Y-NORM",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "ATLAS_SINGLETOP_8TEV_T-RAP-NORM",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "ATLAS_SINGLETOP_8TEV_TBAR-RAP-NORM",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "CMS_WPWM_7TEV_ELECTRON_ASY",
            "frac": 0.75,
        },
        {
            "dataset": "CMS_WPWM_7TEV_MUON_ASY",
            "frac": 0.75,
            "variant": "legacy",
        },
        {"dataset": "CMS_Z0_7TEV_DIMUON_2D", "frac": 0.75},
        {
            "dataset": "CMS_WPWM_8TEV_MUON_Y",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "cfac": ["NRM"],
            "dataset": "CMS_Z0J_8TEV_PT-Y",
            "frac": 0.75,
            "variant": "legacy_10",
        },
        {"dataset": "CMS_2JET_7TEV_M12Y", "frac": 0.75},
        {
            "dataset": "CMS_1JET_8TEV_PTY",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "CMS_TTBAR_7TEV_TOT_X-SEC",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "CMS_TTBAR_8TEV_TOT_X-SEC",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "CMS_TTBAR_13TEV_TOT_X-SEC",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "CMS_TTBAR_8TEV_LJ_DIF_YTTBAR-NORM",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "CMS_TTBAR_5TEV_TOT_X-SEC",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YT-NORM",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "CMS_TTBAR_13TEV_2L_DIF_YT",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "CMS_TTBAR_13TEV_LJ_2016_DIF_YTTBAR",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "CMS_SINGLETOP_7TEV_TCHANNEL-XSEC",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "CMS_SINGLETOP_8TEV_TCHANNEL-XSEC",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "CMS_SINGLETOP_13TEV_TCHANNEL-XSEC",
            "frac": 0.75,
            "variant": "legacy",
        },
        {
            "dataset": "LHCB_Z0_7TEV_DIELECTRON_Y",
            "frac": 0.75,
        },
        {
            "dataset": "LHCB_Z0_8TEV_DIELECTRON_Y",
            "frac": 0.75,
        },
        {
            "cfac": ["NRM"],
            "dataset": "LHCB_WPWM_7TEV_MUON_Y",
            "frac": 0.75,
        },
        {
            "cfac": ["NRM"],
            "dataset": "LHCB_Z0_7TEV_MUON_Y",
            "frac": 0.75,
        },
        {
            "cfac": ["NRM"],
            "dataset": "LHCB_WPWM_8TEV_MUON_Y",
            "frac": 0.75,
        },
        {
            "cfac": ["NRM"],
            "dataset": "LHCB_Z0_8TEV_MUON_Y",
            "frac": 0.75,
        },
        {"dataset": "LHCB_Z0_13TEV_DIMUON-Y", "frac": 0.75},
        {
            "dataset": "LHCB_Z0_13TEV_DIELECTRON-Y",
            "frac": 0.75,
        },
    ]
}


fit = API.fit(fit=fitname)

common_dict = dict(
    fit=fit.name,
    fits=[fit.name],
    use_cuts="fromfit",
    metadata_group="nnpdf31_process",
    theoryid=809, # NNLO, as=0.120
)

dataset_inputs_plus = copy.deepcopy(dataset_inputs)
for ds in dataset_inputs_plus["dataset_inputs"]:
    if "TTBAR" in ds["dataset"] and ds["dataset"]!="CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YT-NORM" and ds["dataset"]!="ATLAS_TTBAR_8TEV_2L_DIF_YTTBAR-NORM":
        ds["cfac"] = ["TOPP"]
dataset_inputs_minus = copy.deepcopy(dataset_inputs)
for ds in dataset_inputs_minus["dataset_inputs"]:
    if "TTBAR" in ds["dataset"]:
        ds["cfac"] = ["TOPP"]

inps_central = dict(pdf=fitname, dataset_inputs=dataset_inputs, **common_dict)
inps_plus = dict(pdf=fitname, dataset_inputs=dataset_inputs_plus, **common_dict)
inps_minus = dict(pdf=fitname, dataset_inputs=dataset_inputs_minus, **common_dict)

theorypreds_central = API.group_result_central_table_no_table(**inps_central)["theory_central"]
theorypreds_plus = API.group_result_central_table_no_table(**inps_plus)["theory_central"]
theorypreds_minus = API.group_result_central_table_no_table(**inps_minus)["theory_central"]

delta_plus = (theorypreds_plus - theorypreds_central) / np.sqrt(2)
delta_minus = (theorypreds_minus - theorypreds_central) / np.sqrt(2)

S = np.outer(delta_plus, delta_plus) + np.outer(delta_minus, delta_minus)
S = pd.DataFrame(S, index=delta_minus.index, columns=delta_minus.index)
