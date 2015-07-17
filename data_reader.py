import pandas as pd


def read(url, normalize=True):
    names = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"]
    glass = pd.read_csv(url, header=None, names=names)
    glass_types = glass["Type"].unique()
    glass_types.sort()

    labels_one_vs_all = {
        "Type " + str(t): glass["Type"].map(lambda x: 1.0 if x == t else 0.0).values
        for t in glass_types
    }

    original_labels = glass["Type"].values
    del glass["Type"]
    names.remove("Type")

    glass.reset_index(inplace=True)
    glass = pd.concat([glass, pd.DataFrame(labels_one_vs_all)], axis=1)
    type_names = ["Type " + str(t) for t in glass_types]

    X = glass[names]
    Y = glass[type_names]

    if normalize:
        X = X.apply(lambda x: (x - x.mean())/x.std())
        Y = Y.apply(lambda y: (y - y.mean())/y.std())

    return X, Y, original_labels