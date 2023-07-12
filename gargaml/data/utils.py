import logging, random
import pandas as pd

import numpy as np


def _do_sep_target(df, sep_target, target):
    """ """

    if not sep_target:
        return df
    return df.drop(columns=target), df[target]


def _clean_columns(txt: str) -> str:
    """_clean columns for a df"""

    txt = txt.lower().strip()
    replace = [
        ("(s)", ""),
        ("(", "_"),
        (")", ""),
        ("/", "_"),
        ("__", "_"),
    ]

    for k, v in replace:
        txt = txt.replace(k, v)

    txt = txt.lower().strip()

    return txt


def _preclean_seattle(df):
    # cols
    cols = [
        "primarypropertytype",
        "neighborhood",
        "latitude",
        "longitude",
        "yearbuilt",
        "numberofbuildings",
        "numberoffloors",
        "propertygfaparking",
        "propertygfabuilding",
        "energystarscore",
        "siteenergyuse_kbtu",
        "steamuse_kbtu",
        "electricity_kbtu",
        "naturalgas_kbtu",
    ]

    df = df.loc[:, cols]

    # drop gfa and target negative
    df = df.loc[df.propertygfabuilding.notna()]
    df = df.loc[df.propertygfabuilding > 1, :]
    df = df.loc[df.siteenergyuse_kbtu.notna()]
    df = df.loc[df.siteenergyuse_kbtu > 1, :]

    # fix floor building
    df.numberofbuildings = df.numberofbuildings.fillna(1).astype(int).replace({0: 1})
    df.numberoffloors = df.numberoffloors.fillna(0).astype(int) + 1

    # bool parking as nrj, stema elec etc
    df.propertygfaparking = df.propertygfaparking.apply(lambda i: i > 0).astype(int)
    df["bool_energystarscore"] = df.energystarscore.isna().astype(int)
    df.steamuse_kbtu = df.steamuse_kbtu.apply(lambda i: i > 0).astype(int)
    df.electricity_kbtu = df.electricity_kbtu.apply(lambda i: i > 0).astype(int)
    df.naturalgas_kbtu = df.naturalgas_kbtu.apply(lambda i: i > 0).astype(int)

    # feat eng
    df["gfaperfloor"] = df.propertygfabuilding / df.numberoffloors
    df["gfaperbuilding"] = df.propertygfabuilding / df.numberofbuildings

    # df.primarypropertytype.value_counts()
    df.primarypropertytype = df.primarypropertytype.replace(
        {"office": "small- and mid-sized office"}
    )

    return df


def _add_nan(df, nan_rate=0.0):
    """ """

    if nan_rate:
        N = df.shape[0] * df.shape[1]
        nan_numb = int(nan_rate * N)

        for _ in range(nan_numb):
            x, y = random.randint(0, df.shape[0] - 1), random.randint(
                0, df.shape[1] - 1
            )
            df.iloc[x, y] = np.NaN

    return df
