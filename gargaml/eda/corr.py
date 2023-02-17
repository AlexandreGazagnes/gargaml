import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()



def _viz(df, cmap="coolwarm", figsize=12):

    assert cmap in ["coolwarm", "RdBu"]

    corr = df.corr()
    fig = plt.figure(figsize=(figsize, figsize))

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, cmap=cmap, mask=mask, square=True)

    return None


def _filter():
    return


class Corr:

    viz = _viz
    matrix = _viz
    plot = _viz