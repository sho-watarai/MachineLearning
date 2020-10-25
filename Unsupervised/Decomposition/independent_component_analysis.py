import matplotlib.pyplot as plt
import numpy as np

from scipy import signal
from sklearn.decomposition import FastICA

np.random.seed(0)


if __name__ == "__main__":
    #
    # generate sample data
    #
    n_samples = 2000
    time = np.linspace(0, 8, n_samples)

    s1 = np.sin(2 * time)  # sinusoidal signal
    s2 = np.sign(np.sin(3 * time))  # square signal
    s3 = signal.sawtooth(2 * np.pi * time)  # saw tooth signal

    S = np.c_[s1, s2, s3]
    S += 0.2 * np.random.normal(size=S.shape)  # add noise

    S /= S.std(axis=0)  # standardization
    M = np.array([[1, 1, 1],  # mixing matrix
                  [0.5, 2, 1.0],
                  [1.5, 1.0, 2.0]])
    X = np.dot(S, M.T)  # generate observations

    #
    # independent component analysis
    #
    ica = FastICA(n_components=3)

    S_ = ica.fit_transform(X)  # reconstruct signals
    A_ = ica.mixing_  # estimated mixing matrix

    #
    # visualization
    #
    plt.figure(figsize=(16, 16))
    for i, (model, label) in enumerate(zip([X, S, S_], ["mixed signal", "truth signals", "ICA recovered signals"])):
        plt.subplot(3, 1, i + 1)
        for s, c in zip(model.T, ["red", "blue", "gold"]):
            plt.plot(s, color=c)
        plt.title(label)
        plt.xlabel("time")
        plt.ylabel("signal")
    plt.show()
    
