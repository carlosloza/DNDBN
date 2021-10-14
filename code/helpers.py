import math
import numpy as np
from typing import Optional, Dict


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



def EEG_expert_labels(eeg, labels, stages):
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
