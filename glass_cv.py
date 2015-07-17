from sklearn.linear_model import enet_path
import matplotlib.pyplot as plot
import numpy as np

from mlutils import sample_indexes, normalize
from glass.data_reader import read


target_url = ("https://archive.ics.uci.edu/ml/machine-learning-"
              "databases/glass/glass.data")


def cross_validation(X, Y, glass_labels, cv=10, n_alphas=200):
    n_rows = len(X)
    mis_class = [0.0] * n_alphas
    means_y, stds_y = Y.mean().values, Y.std().values

    X = normalize(X).values
    Y = normalize(Y).values

    labels_list = list(sorted(set(glass_labels)))
    n_labels = len(labels_list)

    for fold_number in range(cv):
        test_idx, train_idx = sample_indexes(X, fold_number, cv)
        test_x, train_x = X[test_idx], X[train_idx]
        test_y, train_y = Y[test_idx], Y[train_idx]
        labels_test = glass_labels[test_idx]

        # build model for each column in train_y
        models = []
        test_len = n_rows - len(train_y)

        for i_model in range(n_labels):
            current_y = train_y[:, i_model]
            models.append(enet_path(
                train_x, current_y, l1_ratio=1.0, eps=0.5e-3,
                n_alphas=n_alphas, return_models=False))

        for i_step in range(1, n_alphas):
            # assemble the predictions for all the models
            all_predictions = []

            for i_model in range(n_labels):
                _, coefs, _ = models[i_model]
                pred = np.dot(test_x, coefs[:, i_step])
                pred_unnorm = pred*stds_y[i_model] + means_y[i_model]
                all_predictions.append(pred_unnorm)

            for i in range(test_len):
                pred = [all_predictions[j][i] for j in range(n_labels)]
                idx_max = pred.index(max(pred))
                if labels_list[idx_max] != labels_test[i]:
                    mis_class[i_step] += 1.0

    mis_class_plot = [mis_class[i]/n_rows for i in range(1, n_alphas)]
    plot.plot(mis_class_plot)
    plot.xlabel("Penalty Parameter Steps")
    plot.ylabel("Misclassification Error Rate")
    plot.show()


if __name__ == '__main__':
    cross_validation(*read(target_url, normalize=False))