import os
import pickle
import patoolib
import urllib.request
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
from scipy.signal import resample_poly as resample
from fractions import Fraction
from helpers import _round_up, _logging_messages

# KEY: brew install rar for macOS
# TODO: add more methods to read multi-channel EEG


def load_data_single_channel(
        fs: Optional[float] = None,
        path: Optional[str] = None,
        verbose: Optional[int] = 1
) -> Tuple[List, Dict[str, Dict], List[float]]:
    """
    Load single EEG channel (CZ-A1 or C3-A1) from DREAMS Sleep Spindles dataset
    Data hosted at https://zenodo.org/record/2650142/files/DatabaseSpindles.rar
    PS: mne methods do not work with the edf files provided by the dataset authors

    Parameters
    ----------
    fs : float, default=50
        Sampling rate, in Hz.
        If None, the data is not resampled.

    path : str, default=None
        Path to directory where data will be saved.
        If 'None', the default path is './../datasets'.

    verbose: int, default=1
        Verbosity mode. 0 = silent, 1 = messages.

    Returns
    -------
    eeg : list (length 8)
        Each element (1 per subject) contains a 1-D numpy array with shape (30*60*Fs, )

    expert_labels : dictionary (8 keys)
        Each key-value (1 per subject) contains one dictionary with 3 keys-values: Expert1, Expert2, ExpertsUnion
        where expert labels and their union are stored, respectively.
        According to the datasets authors:
        Only subjects 1 through 6 have all 3 possible labels, subjects 7 and 8 only have Expert1 and ExpertsUnion
        Expert1 labels were cut off after 1000 seconds

    new_fs : list (length 8)
         Each element (1 per subject) contains the sampling frequency (float) of the corresponding eeg entries
    """

    # Prefix of processed files
    fname_pkl = 'DREAMS_SleepSpindles'
    # Directory where to cache the dataset locally (relative to /datasets)
    folder = 'DREAMS_Sleep_Spindles'
    if path is None:
        path = os.path.join(os.path.relpath(os.getcwd()), '..', 'datasets')
    # Check if folder exists
    dirname = os.path.join(path, folder)
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    # Name of .rar file from source
    fname_rar = os.path.join(dirname, 'DatabaseSpindles.rar')
    ftest = os.path.join(dirname, 'excerpt1.txt')
    # Check if data has already been downloaded and processed
    tmp = '.pkl' if fs is None else '_Fs' + str(fs) + 'Hz.pkl'

    regenerate = False
    processing_flag = False

    # Messages according to verbosity mode
    msg = _logging_messages(verbose, {'download_msg': 'Downloading DREAMS Sleep Spindles dataset... (45 MBytes)'})

    if not regenerate:
        # Check for pickle file first
        if not os.path.isfile(os.path.join(dirname, fname_pkl + tmp)):
            # Check for .rar file then
            if os.path.isfile(fname_rar):
                if not os.path.isfile(ftest):
                    print(msg['rar_found_msg'])
                    patoolib.extract_archive(fname_rar, outdir=dirname, verbosity=-1)
                    print(msg['extract_rar_msg'])
                else:
                    processing_flag = True
            else:
                print(msg['file_not_found_msg'])
                regenerate = True
                processing_flag = True
        else:
            print(msg['load_pk_msg'])
            with open(os.path.join(dirname, fname_pkl + tmp), 'rb') as f:
                eeg, expert_labels, new_fs = pickle.load(f)
                return eeg, expert_labels, new_fs

    if regenerate:
        # Download dataset from https://zenodo.org/record/2650142/files/DatabaseSpindles.rar
        print(msg['download_msg'])
        # TODO: add download bar
        url = 'https://zenodo.org/record/2650142/files/DatabaseSpindles.rar'
        urllib.request.urlretrieve(url, fname_rar)
        if os.path.isfile(fname_rar):
            print(msg['rar_found_msg'])
            patoolib.extract_archive(fname_rar, outdir=dirname, verbosity=-1)
        print(msg['extract_rar_msg'])

    if processing_flag:
        # Number of subjects
        n_sub = 8
        # Duration of recordings - 30 minutes
        len_record_sec = 30*60
        # Initialize variables
        eeg = [[] for _ in range(n_sub)]
        sub_keys = ['Subject' + str(i) for i in range(1, n_sub + 1)]
        expert_labels = dict(zip(sub_keys, (dict() for _ in sub_keys)))
        expert_names = ['Expert1', 'Expert2']

        # Resampling if fs is provided
        fs_sub = np.zeros(n_sub)
        if fs is not None:
            for i in range(1, n_sub + 1):
                df = pd.read_csv(os.path.join(dirname, "excerpt" + str(i) + ".txt"))
                # Special case
                if i == 6:
                    df = df[:360000]
                # Original sampling frequency
                fs_sub[i-1] = int(len(df) / len_record_sec)
                # Resampling
                f = Fraction(fs_sub[i-1]/fs).limit_denominator()
                eeg[i-1] = resample(df.iloc[:, 0].to_numpy(), f.denominator, f.numerator)[..., None]
            new_fs = [fs] * n_sub
        else:
            for i in range(1, n_sub + 1):
                df = pd.read_csv(os.path.join(dirname, "excerpt" + str(i) + ".txt"))
                # Special case
                if i == 6:
                    df = df[:360000]
                # Original sampling frequency
                fs_sub[i-1] = int(len(df) / len_record_sec)
                # No resampling
                eeg[i-1] = df.iloc[:, 0].to_numpy()[..., None]
            new_fs = fs_sub.tolist()

        # Expert labels
        for sub_i in range(1, n_sub + 1):
            if sub_i <= 6:
                aux = np.zeros((len_record_sec * round(new_fs[sub_i-1]), 2), dtype=np.uint8)
                for i in range(1, 3):
                    df = pd.read_csv(os.path.join(dirname,
                                                  "Visual_scoring" + str(i) + "_excerpt" + str(sub_i) + ".txt"),
                                     header=0)
                    df['SSloc'], df['SSdur'] = zip(*df[df.columns[0]].str.split())
                    df['SSloc'] = df['SSloc'].astype(float)
                    df['SSdur'] = df['SSdur'].astype(float)
                    for j in range(len(df)):
                        aux[_round_up(new_fs[sub_i - 1] * df['SSloc'][j]) - 1:
                            _round_up(new_fs[sub_i - 1] * df['SSloc'][j]) +
                            _round_up(new_fs[sub_i - 1] * df['SSdur'][j]), i - 1] = 1
                    expert_labels[sub_keys[sub_i - 1]][expert_names[i - 1]] = 1 + aux[:, i - 1]
                labels = 1 + np.sum(aux, axis=1)
                labels[labels > 1] = 2
                expert_labels[sub_keys[sub_i - 1]]["ExpertsUnion"] = labels
            else:
                aux = np.zeros((len_record_sec * round(new_fs[sub_i-1])), dtype=np.uint8)
                df = pd.read_csv(os.path.join(dirname,
                                              "Visual_scoring1_excerpt" + str(sub_i) + ".txt"),
                                 header=0)
                df['SSloc'], df['SSdur'] = zip(*df[df.columns[0]].str.split())
                df['SSloc'] = df['SSloc'].astype(float)
                df['SSdur'] = df['SSdur'].astype(float)
                for j in range(len(df)):
                    aux[_round_up(new_fs[sub_i - 1] * df['SSloc'][j]) - 1:
                        _round_up(new_fs[sub_i - 1] * df['SSloc'][j]) +
                        _round_up(new_fs[sub_i - 1] * df['SSdur'][j])] = 1
                labels = 1 + aux
                expert_labels[sub_keys[sub_i - 1]][expert_names[0]] = labels
                expert_labels[sub_keys[sub_i - 1]]["ExpertsUnion"] = labels
        # Save EEG, labels, sampling rates
        with open(os.path.join(dirname, fname_pkl + tmp), 'wb') as f:
            pickle.dump([eeg, expert_labels, new_fs], f)
        print(msg['save_pk_msg'])
        return eeg, expert_labels, new_fs
