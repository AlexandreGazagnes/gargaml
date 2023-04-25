from sklearn.pipeline import *
from sklearn.preprocessing import *
from sklearn.compose import *
import numpy as np


passthrough_numb_transformer = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            make_column_selector(dtype_include=np.number),
        )
    ],
    remainder="drop",
)

oneHot_str_transformer = ColumnTransformer(
    transformers=[
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore", dtype=int),
            make_column_selector(dtype_include=object),
        )
    ],
    remainder="drop",
)


simple_numb_str_transformer = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            make_column_selector(dtype_include=np.number),
        ),
        (
            "onehot",
            "onehot",
            OneHotEncoder(handle_unknown="ignore", dtype=int),
            make_column_selector(dtype_include=object),
        ),
    ],
    remainder="drop",
)
