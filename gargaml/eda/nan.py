def nan_rate(df, threshold=0.0):
    """filter cols by nan rate > threshold """

    tmp = df.isna().mean().sort_values(ascending=False)
    tmp = tmp[tmp >= threshold]
    return tmp.round(3)


def nan_cols(df, threshold):
    """give cols with Nan rate > threshold """

    tmp = nan_rate(df, threshold=threshold)

    return tmp[tmp >= threshold].index.tolist()


def study_distribution(ser, feature=-1, val=True, numeric=True):
    """ """

    # check agrs
    if isinstance(ser, pd.Series):
        ser = ser
    else:
        ser = ser[feature]

    if numeric:
        # fig, axs
        fig, axs = plt.subplots(1, 3, figsize=(12, 6))
        axs = axs.flatten()

        # plot
        ser.plot(kind="box", ax=axs[0])
        ser.plot(kind="hist", ax=axs[1], bins=20)
        axs[2].plot(range(len(ser)), ser.sort_values())

        # info
        axs[0].set_title("box plot")
        axs[1].set_title("dist plot")
        axs[2].set_title("values plot")

    # print value_counts and descibe
    if val:
        print("Value_counts")
        print(ser.value_counts(ascending=False, dropna=False))
        print("\n\n")
    print("Describe")
    print(ser.describe())

    return None


class Nan:
    study_distribution = study_distribution
    nan_cols = nan_cols
    nan_rate = nan_rate
