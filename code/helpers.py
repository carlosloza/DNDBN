import math
import numpy as np
from scipy.stats import zscore
from typing import Optional, Dict, List, Tuple
from numpy import logaddexp


def _round_up(x: float) -> int:
    """Custom rounding method for benchmarking with previous versions"""
    if (float(x) % 1) >= 0.5:
        return math.ceil(x)
    else:
        return round(x)


def _logging_messages(verbose: int, msg: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Logging messages when manipulating data from the DREAMS database"""
    output = dict()
    output['extract_rar_msg'] = 'Extracting done!'
    output['file_not_found_msg'] = 'Files not found in provided path. Will proceed with downloading from source.'
    output['rar_found_msg'] = '.rar file found. Extracting files...'
    output['load_pk_msg'] = 'Loading locally cached pickle file.'
    output['save_pk_msg'] = 'Pickle file locally cached'
    output['download_msg'] = 'Downloading data'
    if verbose == 0:
        msg = {key: '' for key in output.keys()}
    # Update messages with inputs
    output.update(msg)
    return output


def eeg_expert_labels(
        eeg: np.ndarray,
        labels: np.ndarray,
        stages: Dict[str, List]
) -> Dict[str, List]:
    """
    Segment single-channel EEG (from DREAMS Patients or DREAMS Subjects datasets) into stages according to expert labels

    Parameters
    ----------
    eeg : numpy array, shape (n_hours*3600*fs, ) where n_hours is the number of hours in the recording

    labels: numpy array, shape (n_hours*3600*fs, ) where n_hours is the number of hours in the recording
        Possible unique values equal to different sleep stages. On-to-one correspondence with keys in stages argument.

    stages: dictionary (n keys) where n is the number of different sleep stages according to specific criterion

    Returns
    -------
    stages: dictionary (n keys) where n is the number of different sleep stages according to specific criterion
        Updated dictionary with segmented eeg according to labels.
    """

    for i, (k, v) in enumerate(stages.items()):
        if v is None:
            stages[k] = list()
        s = 0
        idx = np.where(labels == i + 1)[0]
        if idx.shape[0] > 0:
            idx_diff = np.where(np.diff(idx) > 1)[0]
            if idx_diff.shape[0] > 0:
                for j in range(idx_diff.shape[0]):
                    if j == 0:
                        stages[k].append(eeg[idx[:idx_diff[0] + 1], :])
                    else:
                        stages[k].append(eeg[idx[idx_diff[j - 1] + 1:idx_diff[j] + 1], :])
                    s += stages[k][-1].shape[0]
                stages[k].append(eeg[idx[idx_diff[-1] + 1:], :])
            else:
                stages[k].append(eeg[idx, :])
            s += stages[k][-1].shape[0]
    return stages


def time_embedding(eeg: np.ndarray, ar_order: int) -> Tuple[np.ndarray, int]:
    """Time embedding for an eeg recording, single or multi channel"""
    n, n_dims = eeg.shape
    eeg_embed = np.zeros((n - ar_order, ar_order, n_dims))
    for j in range(n - ar_order):
        eeg_embed[j, :, :] = -np.flip(eeg[j:j + ar_order, :], axis=0)
    return eeg_embed, n


def preprocess(
        eeg: List[np.ndarray],
        labels: Optional[List],
        ar_order: int = 0,
        normalize: bool = True
        ) -> Tuple[List[np.ndarray], List[np.ndarray], List[List], List[int]]:
    """
    Pre-processing pipeline (scaling, time embedding, labeling) of a batch of eeg recordings

    Parameters
    ----------
    eeg: list
        Each list element is a single-channel eeg trace.

    labels: list
        Each list element is a label array corresponding to the eeg traces in eeg argument.

    ar_order: int, default=0
        Autoregressive order of the model.

    normalize: bool, default=True
        Normalization flag, z-score approach

    Returns
    -------
    list_eeg_embed: list
        Each list element is a time-embedded version of the elements in eeg argument.

    list_eeg_target: list
        Each list element contains the targets corresponding to the elements of list_eeg_embed output.

    list_all_labels: list
        Each list element contains the labels corresponding to the elements of list_eeg_embed output.

    n: list
        Each list element contains the length of the original eeg data (i.e., the elements in eeg argument)
    """

    n_seq = len(eeg)
    n = list()
    # Assume all recordings are the same type, i.e. same number of channels/dimensions
    if normalize is True:
        eeg = [zscore(x, axis=0) for x in eeg]
    list_eeg_embed, list_all_labels, list_eeg_target = list(), list(), list()
    if ar_order > 0:
        for i in range(n_seq):
            eeg_embed, n_i = time_embedding(eeg[i], ar_order)
            list_eeg_embed.append(eeg_embed)
            n.append(n_i)
            if labels is not None:
                list_all_labels.append(labels[i][ar_order:].astype(np.float32))
            list_eeg_target.append(eeg[i][ar_order:, ...])
    else:
        pass
        # TODO: add non-autoregressive case
    return list_eeg_embed, list_eeg_target, list_all_labels, n


def hard2soft_labels(
        x: np.ndarray,
        low: float = 0.8,
        high: float = 0.99
) -> np.ndarray:
    """
    Add random perturbations to hard labels that are 100% confident (i.e., no uncertainty)

    Parameters
    ----------
    x: numpy array, shape (n_samples, n_labels)
        One-hot encoded labels

    low: float, default=0.8
        Minimum value for soft labels (i.e., minimum degree of certainty)

    high: float, default=0.99
        Maximum value for soft labels (i.e., maximum degree of certainty)

    Returns
    -------
    x: numpy array, shape (n_samples, n_labels)
        Updated (soft) labels with uncertainty
    """

    num_states = x.shape[1]
    num_states_arr = range(num_states)
    for k in range(num_states):
        idx = np.where(np.argmax(x, axis=1) == k)
        s_labels = np.random.uniform(low, high, idx[0].size)
        x[idx[0], k] = s_labels
        x[idx[0], np.setdiff1d(num_states_arr, k)] = (1 - x[idx[0], k]) / (num_states - 1)
    return x


def log_sum_exp(x: np.ndarray, axis=None) -> np.ndarray:
    """ Fast implementation of log-sum-exp (when compared to scipy's implementation)"""
    return logaddexp.reduce(x, axis=axis)
