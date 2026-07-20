import numpy as np
from itertools import combinations
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from GB_generation_with_idx import get_GB


def ball_information(GBs, m):
    centers = np.zeros((len(GBs), m))
    radii = np.zeros(len(GBs))
    indices = []
    for i, gb in enumerate(GBs):
        X = gb[:, :m]
        centers[i] = np.mean(X, axis=0)
        radii[i] = np.max(np.linalg.norm(X - centers[i], axis=1))
        indices.append(gb[:, -1].astype(int))
    return centers, radii, indices


def cknn(centers, k):
    n = len(centers)
    k = min(k, n - 1)
    D = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    return np.argsort(D, axis=1)[:, :k]


def angular_variance(centers, radii, neighbors):
    n = len(centers)
    V = np.zeros(n)
    for i in range(n):
        vec = centers[neighbors[i]] - centers[i]
        dist2 = np.sum(vec ** 2, axis=1)
        phi = []
        weight = []
        for j, l in combinations(range(len(neighbors[i])), 2):
            if dist2[j] < 1e-24 or dist2[l] < 1e-24:
                continue
            value = np.dot(vec[j], vec[l]) / (dist2[j] * dist2[l])
            gj = neighbors[i, j]
            gl = neighbors[i, l]
            gamma = ((radii[i] ** 2 + radii[gj] ** 2) / dist2[j]
                     + (radii[i] ** 2 + radii[gl] ** 2) / dist2[l])
            phi.append(value)
            weight.append(1 / (1 + gamma))
        if len(phi) == 0:
            continue
        phi = np.asarray(phi)
        weight = np.asarray(weight)
        mean = np.sum(weight * phi) / np.sum(weight)
        V[i] = np.sum(weight * (phi - mean) ** 2) / np.sum(weight)
    return V


def GABOD(data, k=5):
    data = np.asarray(data, dtype=float)
    n, m = data.shape
    GBs = get_GB(data)
    centers, radii, indices = ball_information(GBs, m)
    neighbors = cknn(centers, k)
    V = angular_variance(centers, radii, neighbors)
    if np.max(V) == np.min(V):
        OS_gb = np.full(len(V), 0.5)
    else:
        OS_gb = 1 - (V - np.min(V)) / (np.max(V) - np.min(V))
    OS = np.zeros(n)
    for i, idx in enumerate(indices):
        OS[idx] = OS_gb[i]
    return OS


if __name__ == "__main__":
    data = loadmat("./Datasets/breast_cancer_variant1.mat")["trandata"]
    X = data[:, :-1]
    labels = data[:, -1]
    X = MinMaxScaler().fit_transform(X)
    scores = GABOD(X, k=5)
    print("AUC: ", roc_auc_score(labels, scores))
