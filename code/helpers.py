import math
import numpy as np
from scipy.stats import zscore
from typing import Optional, Dict
from numpy import logaddexp


def _round_up(x):
    """Custom rounding method for benchmarking with previous versions"""
    if (float(x) % 1) >= 0.5:
        return math.ceil(x)
    else:
        return round(x)


def _logging_messages(verbose: int, msg: Optional[Dict[str, str]]):
    output = dict()
    output['extract_rar_msg'] = 'Extracting done!'
    output['file_not_found_msg'] = 'Files not found in provided path. Will proceed with downloading from source.'
    output['rar_found_msg'] = '.rar file found. Extracting files...'
    output['load_pk_msg'] = 'Loading locally cached pickle file.'
    output['save_pk_msg'] = 'Pickle file locally cached'
    output['download_msg'] = 'Downloading data'
    if verbose == 0:
        msg = {key: '' for key in output.keys()}
    output.update(msg)
    return output


def eeg_expert_labels(eeg, labels, stages):
    """Segment single-channel EEG into stages according to expert labels"""
    for i, (k, v) in enumerate(stages.items()):
        s = 0
        idx = np.where(labels == i + 1)[0]
        if idx.shape[0] > 0:
            idxdiff = np.where(np.diff(idx) > 1)[0]
            if idxdiff.shape[0] > 0:
                for j in range(idxdiff.shape[0]):
                    if j == 0:
                        stages[k].append(eeg[idx[:idxdiff[0] + 1], :])
                    else:
                        stages[k].append(eeg[idx[idxdiff[j - 1] + 1:idxdiff[j] + 1], :])
                    s += stages[k][-1].shape[0]
                stages[k].append(eeg[idx[idxdiff[-1] + 1:], :])
            else:
                stages[k].append(eeg[idx, :])
            s += stages[k][-1].shape[0]
    return stages


def time_embedding(eeg, ar_order):
    # Embedding for a single recording - can be single or multi channel
    n, n_dims = eeg.shape
    xp_seq = np.zeros((n - ar_order, ar_order, n_dims))
    for j in range(n - ar_order):
        xp_seq[j, :, :] = -np.flip(eeg[j:j + ar_order, :], axis=0)
    return xp_seq, n


def preprocess(eeg, ar_order=0, labels=None, normalize=True):
    # EEG is a list of multidimensional numpy arrays, labels is a list too
    n_seq = len(eeg)
    n = list()
    # Assume all recordings are the same type, i.e. same number of channels/dimensions
    if normalize is True:
        eeg = [zscore(x, axis=0) for x in eeg]
    xp, labels_all, y = list(), list(), list()
    if ar_order > 0:
        for i in range(n_seq):
            xp_seq, n_i = time_embedding(eeg[i], ar_order)
            xp.append(xp_seq)
            n.append(n_i)
            if labels is not None:
                labels_all.append(labels[i][ar_order:].astype(np.float32))
            y.append(eeg[i][ar_order:, ...])
    # TODO: add non-autoregressive case
    return xp, y, labels_all, n


def hard2soft_labels(x, low=0.8, high=0.99):
    num_states = x.shape[1]
    num_states_arr = range(num_states)
    for k in range(num_states):
        idx = np.where(np.argmax(x, axis=1) == k)
        s_labels = np.random.uniform(low, high, idx[0].size)
        x[idx[0], k] = s_labels
        x[idx[0], np.setdiff1d(num_states_arr, k)] = (1 - x[idx[0], k]) / (num_states - 1)
    return x


def log_sum_exp(x, axis=None):
    """ Fast implementation of log-sum-exp (in comparison to scipy's implementation)"""
    return logaddexp.reduce(x, axis=axis)
