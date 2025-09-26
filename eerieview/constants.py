location2prefix = dict(cloud="dkrz.cloud", levante="dkrz.disk.model-output")
members_eerie_control = [
    # "hadgem3-gc5-n640-orca12.eerie-picontrol.atmos.gr025.daily.0",
    "icon-esm-er.eerie-control-1950.v20240618.atmos.gr025.2d_monthly_mean",
    "ifs-fesom2-sr.eerie-control-1950.v20240304.atmos.gr025.2D_monthly_avg",
]
members_eerie_hist = [
    "ifs-fesom2-sr.hist-1950.v20240304.atmos.gr025.2D_monthly_avg",
    "icon-esm-er.hist-1950.v20240618.atmos.gr025.2d_monthly_mean",
]

members_eerie_future = [
    "ifs-fesom2-sr.highres-future-ssp245.v20240304.atmos.gr025.2D_monthly_avg",
    "icon-esm-er.highres-future-ssp245.v20240618.atmos.gr025.2d_monthly_mean",
]

futuremember2hist = {
    members_eerie_future[0]: members_eerie_hist[0],
    members_eerie_future[1]: members_eerie_hist[1],
}

members_eerie_hist_amip = [
    "ifs-amip-tco1279.hist.v20240901.atmos.gr025.2D_monthly",
    "ifs-amip-tco1279.hist-c-0-a-lr20.v20240901.atmos.gr025.2D_monthly",
    "ifs-amip-tco399.hist-c-0-a-lr20.v20240901.atmos.gr025.2D_monthly",
    "ifs-amip-tco399.hist-c-lr20-a-0.v20240901.atmos.gr025.2D_monthly",
    "ifs-amip-tco399.hist.v20240901.atmos.gr025.2D_monthly",
]

missing_periods = ["1950-1969"]
AVISO_VARIABLES = ["eke", "zos"]
OCEAN_VARIABLES = ["tos", "sic", "zos", "uo", "vo"]
CMOR2EERIE = {
    "pr": "tprate",
    "tas": "mean2t",
    "tasmax": "mx2t",
    "tasmin": "mn2t",
    "clt": "meantcc",
    "tos": "avg_tos",
    "zos": "avg_zos",
    "uas": "m10u",
    "vas": "m10v",
    "sic": "mci",
    "sfcWind": "mean10ws",
    "eke": "eke",
}
CMOR2EERIEAMIP = CMOR2EERIE.copy()
CMOR2EERIEAMIP.update({"uas": "avg_10u", "vas": "avg_10v", "tos": "avg_sst"})
CMOR2ERA5 = {
    "tasmean": "t2m",
    "pr": "tp",
    "clt": "tcc",
    "tos": "sst",
    "uas": "u10",
    "vas": "v10",
    "sic": "siconc",
}

CMOR2C3SATLAS = {
    "tasmax": "tx",
    "tasmin": "tn",
    "tas": "t",
    "pr": "pr",
    "clt": "clt",
    "tos": "sst",
    "sfcWind": "sfcwind",
    "sic": "siconc",
    "uas": "u10",
    "vas": "v10",
    "zos": "zos",
    "eke": "eke",
}
NUM2MONTH = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}
