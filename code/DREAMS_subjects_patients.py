import os
import pickle
import patoolib
import mne
import urllib.request
import numpy as np
from typing import Optional, Tuple, Dict, List
from helpers import EEG_expert_labels, _logging_messages

# TODO: Add functionalities to handle other sampling frequencies besides 50 Hz


def load_subject(
        dataset: str,
        subj: int,
        path: Optional[str] = None,
        verbose: Optional[int] = 1
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, List], Dict[str, List]]:
    """
    Load and process single EEG channel (C3-A1, CZ-A1, C3-A2) from DREAMS Patients dataset or load single
    EEG channel (C3-A1, CZ-A1, C3-A2) from DREAMS Subjects dataset for a given subject
    Data hosted at https://zenodo.org/record/2650142/files/DatabasePatients.rar
    and https://zenodo.org/record/2650142/files/DatabaseSubjects.rar

    Parameters
    ----------
    dataset : str
        Dataset to load. 'Patients' or 'Subjects'.

    subj : int
        Subject identifier (1 through 27).

    path : str, default=None
        Path to directory where data will be saved.
        If 'None', the default path is './../datasets'.

    verbose: int, default=1
        Verbosity mode. 0 = silent, 1 = messages.

    Returns
    -------
    eeg : Numpy array, shape (n_hours*3600*fs, ) where n_hours is the number of hours in the recording

    labels_score : dictionary (2 keys)
        Each key-value ('AASM', 'RK') contains per-time-sample labeling according to
        https://zenodo.org/record/2650142/files/The%20DREAMS%20Databases%20and%20Assessment%20Algorithm.pdf:
        AASM:
            - 5 = wake
            - 4 = REM stage
            - 3 = sleep stage N1
            - 2 = sleep stage N2
            - 1 = sleep stage N3
            - 0, -1, -2, or -3 = unknown sleep stage
        R&K:
            - 5 = wake
            - 4 = REM stage
            - 3 = sleep stage S1
            - 2 = sleep stage S2
            - 1 = sleep stage S3
            - 0 = sleep stage S4
            - -1 = sleep stage movement
            - -2, or -3 = unknown sleep stage

    aasm : dictionary (5 keys)
        Each key-value contains a list with EEG segments from a particular stage according to the AASM scoring criterion

    rk : dictionary (6 keys)
        Each key-value contains a list with EEG segments from a particular stage according to the R&K scoring criterion
    """

    original_dataset = dataset
    dataset = dataset.capitalize()
    if dataset not in {'Patients', 'Subjects'}:
        raise Exception('"{}" not recognized, only two dataset options are available: Patients or Subjects'.
                        format(original_dataset))

    # Directory where to cache the dataset locally (relative to /datasets)
    folder_dict = {'Patients': 'DREAMS_Patients', 'Subjects': 'DREAMS_Subjects'}
    folder = folder_dict[dataset]
    # For now, only implemented for 50 Hz sampling rate
    fs = 50
    if path is None:
        path = os.path.join(os.path.relpath(os.getcwd()), '..', 'datasets')
    # Check if folder exists
    dirname = os.path.join(path, folder)
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    # Different files to use
    fname_rar = os.path.join(dirname, 'Database' + dataset + '.rar')
    url = 'https://zenodo.org/record/2650142/files/Database{}.rar'.format(dataset)
    file_dataset = 'patient' if dataset == 'Patients' else 'subject'
    ftest = os.path.join(dirname, file_dataset + str(subj) + '.edf')
    pickle_file = os.path.join(dirname, file_dataset + str(subj) + '_' + str(fs) + 'Hz.pkl')
    edf_file = os.path.join(dirname, file_dataset + str(subj) + '.edf')
    labels_file = {'AASM': os.path.join(dirname, 'HypnogramAASM_{}'.format(file_dataset) + str(subj) + '.txt'),
                   'RK': os.path.join(dirname, 'HypnogramR&K_{}'.format(file_dataset) + str(subj) + '.txt')}

    regenerate = False
    processing_flag = False

    # Messages according to verbosity mode
    msg = _logging_messages(verbose, {'download_msg': 'Downloading DREAMS {} dataset, subject {}...'.
                            format(dataset, subj)})

    if not regenerate:
        # Check for pickle file first
        if not os.path.isfile(pickle_file):
            # Check for .rar file then
            if os.path.isfile(fname_rar):
                if not os.path.isfile(ftest):
                    print(msg['rar_found_msg'])
                    patoolib.extract_archive(fname_rar, outdir=dirname)
                    print(msg['extract_rar_msg'])
                else:
                    processing_flag = True
            else:
                print(msg['file_not_found_msg'])
                regenerate = True
                processing_flag = True
        else:
            print(msg['load_pk_msg'])
            with open(pickle_file, 'rb') as f:
                eeg, labels_score, aasm, rk = pickle.load(f)
                return eeg, labels_score, aasm, rk
    if regenerate:
        # Download dataset from https://zenodo.org/record/2650142/files/DatabasePatients.rar
        # or https://zenodo.org/record/2650142/files/DatabaseSubjects.rar
        print(msg['download_msg'])
        # TODO: add download bar
        urllib.request.urlretrieve(url, fname_rar)
        if os.path.isfile(fname_rar):
            print(msg['rar_found_msg'])
            patoolib.extract_archive(fname_rar, outdir=dirname)
        print(msg['extract_rar_msg'])

    if processing_flag:
        # New sampling frequency
        new_fs = fs
        labels_score = dict()
        data = mne.io.read_raw_edf(edf_file, verbose=0)
        eeg_allch = data.get_data()
        # get one channel
        idx = None
        if any('C3-A1' in s for s in data.ch_names):
            idx = data.ch_names.index('C3-A1')
        elif any('CZ-A1' in s for s in data.ch_names):
            idx = data.ch_names.index('CZ-A1')
        elif any('C3-A2' in s for s in data.ch_names) and dataset == 'Patients':
            idx = data.ch_names.index('C3-A2')
        assert idx is not None, 'Channel not found'

        # Resample EEG at 50 Hz
        eeg = eeg_allch[idx:idx+1, ::4].T

        # AASM labels
        with open(labels_file['AASM']) as f:
            lines = f.read().splitlines()
        temp = np.array(list(map(int, lines[1:])))
        labels = np.kron(temp, np.ones(5 * new_fs))
        aasm = {'N3': list(), 'N2': list(), 'N1': list(), 'REM': list(), 'wake': list()}
        aasm = EEG_expert_labels(eeg, labels, aasm)
        labels_score['AASM'] = labels

        # R&K labels
        with open(labels_file['RK']) as f:
            lines = f.read().splitlines()
        temp = np.array(list(map(int, lines[1:])))
        labels = np.kron(temp, np.ones(5 * new_fs))
        rk = {'S4': list(), 'S3': list(), 'S2': list(), 'S1': list(), 'REM': list(), 'wake': list()}
        rk = EEG_expert_labels(eeg, labels, rk)
        labels_score['RK'] = labels

        with open(pickle_file, 'wb') as f:
            pickle.dump([eeg, labels_score, aasm, rk], f)
            print(msg['save_pk_msg'])
        return eeg, labels_score, aasm, rk


def load_all_subjects(
        dataset: str,
        path: Optional[str] = None,
        verbose: Optional[int] = 1
) -> Dict:
    """
    Load and process single-channel EEG data from DREAMS Patients dataset or
    DREAMS Subjects dataset for all subjects in the dataset

    Parameters
    ----------
    dataset : str
        Dataset to load. 'Patients' or 'Subjects'

    path : str, default=None
        Path to directory where data will be saved.
        If 'None', the default path is './../datasets'.

    verbose: int, default=1
        Verbosity mode. 0 = silent, 1 = messages.

    Returns
    -------
    subjects: dictionary (n_subjects keys) where n_subjects is the number of subjects in the dataset
        Each key-value contains a tuple of arrays and dictionaries according to the output of the "load_subject" method
    """

    subjects = dict()
    original_dataset = dataset
    dataset = dataset.capitalize()
    if dataset == 'Patients':
        subjects_id = range(1, 28)
        for subj in subjects_id:
            subjects['patient_{}'.format(subj)] = load_subject('Patients', int(subj), path, verbose=0)
            if verbose == 1:
                print('Patient {} successfully loaded, processed, and saved'.format(subj))
    elif dataset == 'Subjects':
        subjects_id = range(1, 21)
        for subj in subjects_id:
            subjects['subject_{}'.format(subj)] = load_subject('Subjects', int(subj), path, verbose=0)
            if verbose == 1:
                print('Subject {} successfully loaded, processed, and saved'.format(subj))
    else:
        raise Exception('"{}" not recognized, only two dataset options are available: Patients or Subjects'.
                        format(original_dataset))
    return subjects
