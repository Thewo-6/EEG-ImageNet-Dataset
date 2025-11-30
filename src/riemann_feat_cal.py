import os
import numpy as np
import mne
from utilities import *

from pyriemann.utils.covariance import covariances
from pyriemann.tangentspace import TangentSpace


def riemann_feat_cal(
    eeg_data,
    args,
    sfreq=1000,
    bands=None,
    cache_dir="../data/riemann_feat",
):
    """
    Compute multi-band Riemannian features for EEG-ImageNet epochs.

    Parameters
    ----------
    eeg_data : np.ndarray
        Shape (n_epochs, n_channels, n_times).
    args : argparse.Namespace
        Should contain .subject and .granularity.
    sfreq : float
        Sampling frequency (Hz).
    bands : dict or None
        Dict of {band_name: (fmin, fmax)}.
        If None, use default theta/alpha/beta/gamma.
    cache_dir : str
        Directory where features are cached as .npy.

    Returns
    -------
    feats : np.ndarray
        Shape (n_epochs, n_features_total), where
        n_features_total = n_bands * n_features_per_band.
    """

    # ------------- 0. Caching -------------
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(
        cache_dir, f"{args.subject}_{args.granularity}_riemann_multiband.npy"
    )

    if os.path.exists(out_path):
        return np.load(out_path)

    # ------------- 1. Default bands -------------
    if bands is None:
        # With 0.4s epochs, we ignore delta (<4 Hz), which is too slow
        bands = {
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 45.0),
        }

    n_epochs, n_channels, n_times = eeg_data.shape

    # ------------- 2. Create MNE Epochs -------------
    ch_names = [f"EEG{i}" for i in range(1, n_channels + 1)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    # data shape for EpochsArray: (n_epochs, n_channels, n_times)
    epochs = mne.EpochsArray(eeg_data, info, verbose=False)

    # ------------- 3. Multi-band processing -------------
    band_features = []

    # Use a moderate-order IIR filter to avoid super long FIRs
    iir_params = dict(order=4, ftype="butter")

    for band_name, (fmin, fmax) in bands.items():
        # Copy epochs for safety
        band_epochs = epochs.copy().filter(
            fmin,
            fmax,
            method="iir",
            iir_params=iir_params,
            verbose=False,
        )

        # Get data back as numpy
        X_band = band_epochs.get_data()  # shape: (n_epochs, n_channels, n_times)

        # 3a. --- regularized covariances ---
        n_epochs_band, n_channels, _ = X_band.shape
        covs_band = covariances(X_band, estimator="oas")  # (n_epochs, n_channels, n_channels)

        # extra small ridge: 1e-6 * trace(C)/n_channels * I
        eps = 1e-6
        for i in range(n_epochs_band):
            trace = np.trace(covs_band[i])
            covs_band[i] += eps * (trace / n_channels) * np.eye(n_channels)

        # 3b. Map SPD matrices to tangent space
        ts = TangentSpace(metric="riemann")
        ts.fit(covs_band)  # learn reference point
        feats_band = ts.transform(covs_band)  # shape: (n_epochs, n_features_band)

        band_features.append(feats_band)

    # ------------- 4. Concatenate bands -------------
    feats = np.concatenate(band_features, axis=1)  # (n_epochs, n_features_total)

    # ------------- 5. Cache and return -------------
    np.save(out_path, feats)
    return feats