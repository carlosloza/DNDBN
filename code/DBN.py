import numpy as np
import sklearn.linear_model as sk_linear
from scipy.stats import t
from scipy.special import digamma
import scipy.optimize as optimize
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from typing import Optional
tf.keras.backend.set_floatx('float64')

CALLBACK = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3,
                                             verbose=1, mode='min', min_delta=0)]


class DynamicBayesianNetwork:
    
    def __init__(self, K: int, fs: float, ar_order: Optional[int] = 0, normalize: Optional[bool] = True):
        self.K = K
        self.fs = fs
        self.ar_order = ar_order
        if ar_order > 1:
            self.idx_ini = ar_order
            self.ar_flag = True
        else:
            self.idx_ini = 0
            self.ar_flag = False
        self.n_seq = None
        self.n = None
        self.n_dims = None
        self.x = None
        self.y = None
        self.labels = None
        self.normalize = normalize
        self.observation_parameters = dict()
        self.state_parameters = dict()
        self.duration_parameters = dict()
        self.em = dict()
        self.converged = False
        self.nll = list()

    def obs_max_linear(self, df_prev=None, complete_data=False, idx_complete_data=None):
        # M-step
        if df_prev is None:
            df_prev = [-1] * self.K
        nll = np.zeros(self.K)
        for k in range(self.K):
            loc_model = sk_linear.LinearRegression(fit_intercept=True)
            if complete_data is True:
                # TODO: Only works for one-dimensional output (apparently) Must double check this
                idx = np.concatenate(self.em['gamma'], axis=0)[:, k] == 1
            else:
                idx = np.asarray([True] * len(np.concatenate(self.y, axis=0)))
            if idx_complete_data is not None:
                # Semi-Supervised case
                for i_seq in range(self.n_seq):
                    idx_complete = self.em['gamma'][i_seq][idx_complete_data[i_seq] is True, k] == 1
                    idx_incomplete = idx_complete_data[i_seq] is False
                    idx = np.logical_or(idx_complete, idx_incomplete)

            weight = np.concatenate(self.em['gamma'], axis=0)[:, k] * np.concatenate(self.em['w'], axis=0)[:, k]
            loc_model.fit((np.sqrt(weight)[..., None] * np.squeeze(np.concatenate(self.x, axis=0)))[idx],
                          (np.sqrt(weight) * np.squeeze(np.concatenate(self.y, axis=0)))[idx])
            self.observation_parameters['error_parameters'][k]['loc_model'] = loc_model
            self.observation_parameters['error'][k] = np.squeeze(np.concatenate(self.y, axis=0))[idx] - \
                                                      loc_model.predict(np.squeeze(np.concatenate(self.x, axis=0))[idx])
            self.observation_parameters['error_parameters'][k]['scale'] = np.sqrt(
                np.sum(weight[idx] * (self.observation_parameters['error'][k]) ** 2) /
                np.sum(np.concatenate(self.em['gamma'], axis=0)[:, k]))
            if df_prev[k] == -1:
                t_param = t.fit(self.observation_parameters['error'][k])
                df_prev[k] = t_param[0]
                self.observation_parameters['error_parameters'][k]['df'] = t_param[0]
                self.observation_parameters['error_parameters'][k]['scale'] = t_param[2]
            else:
                aux_nu = 1 + \
                         np.sum((np.concatenate(self.em['gamma'], axis=0)[:, k] *
                                (np.log(np.concatenate(self.em['w'], axis=0)[:, k]) -
                                 np.concatenate(self.em['w'], axis=0)[:, k]))[idx]) / \
                         np.sum((np.concatenate(self.em['gamma'], axis=0)[:, k])[idx]) + \
                         digamma((df_prev[k] + 1) / 2) - np.log((df_prev[k] + 1) / 2)

                def df_func(df):
                    return aux_nu - digamma(df / 2) + np.log(df / 2)

                self.observation_parameters['error_parameters'][k]['df'] = optimize.brentq(df_func, 0.1, 100)
            # TODO: Add bias to location parameter
            t_dist = tfd.StudentT(df=self.observation_parameters['error_parameters'][k]['df'],
                                  loc=0,
                                  scale=self.observation_parameters['error_parameters'][k]['scale'])
            nll[k] = -tf.math.reduce_mean(t_dist.log_prob(self.observation_parameters['error'][k]))
        self.observation_parameters['nll'].append(np.mean(nll))
        return self

    def obs_exp_t(self, complete_data=False):
        # E-step: Expectations of tau under posterior conditional, only valid for linear AR model of observations
        for i_seq in range(self.n_seq):
            for k in range(self.K):
                df = self.observation_parameters['error_parameters'][k]['df']
                scale = self.observation_parameters['error_parameters'][k]['scale']
                loc_model = self.observation_parameters['error_parameters'][k]['loc_model']
                idx = self.em['gamma'][i_seq][:, k] == 1 if complete_data is True else np.ones(self.x[i_seq].shape[0],
                                                                                               dtype=bool)
                self.em['w'][i_seq][idx, k] = (df + 1) / \
                                             (df + ((np.squeeze(self.y[i_seq]) -
                                                     loc_model.predict(np.squeeze(self.x[i_seq])))[
                                                        idx] ** 2) / scale ** 2)
                self.em['w'][i_seq][~idx, k] = np.nan  # this only makes sense for complete data, i.e. no EM
        return self

    def obs_max_non_linear(self, observation_model, complete_data=False, random_state=42, verbose=0):
        if verbose == 1:
            verbose = 2
        # Non-linear, deep models for distribution parameters
        assert isinstance(observation_model, dict), \
            "Observation model has to be either a string ('Linear') or a dictionary"

        # Check inputs and set defaults
        self.observation_parameters['epochs'] = observation_model.get('epochs', 10)
        self.observation_parameters['batch_size'] = observation_model.get('batch_size', 32)
        self.observation_parameters['callbacks'] = observation_model.get('callbacks',
                                                                         [tf.keras.callbacks.EarlyStopping(
                                                                             monitor='val_loss', patience=3, verbose=1,
                                                                             mode='min', min_delta=0)])
        self.observation_parameters['validation_split'] = observation_model.get('validation_split', 0.25)
        self.observation_parameters['name'] = observation_model.get('name', 'Deep_Observation_Model')

        for k in range(self.K):
            if complete_data is True:
                idx = np.concatenate(self.em['gamma'], axis=0)[:, k] == 1
            else:
                idx = np.asarray([True] * len(np.concatenate(self.y, axis=0)))
            model = observation_model['model'][k]
            x_all = np.squeeze(np.concatenate(self.x, axis=0)[idx])
            y_all = np.concatenate(self.y, axis=0)[idx]
            indices = range(x_all.shape[0])
            x_train, x_val, y_train, y_val, indices_train, indices_val = train_test_split(
                x_all, y_all, indices, test_size=self.observation_parameters['validation_split'],
                random_state=random_state)
            if complete_data:
                # TODO: Change this next line
                verbosetemp = 0
                model.fit(x_train, y_train,
                          epochs=self.observation_parameters['epochs'],
                          batch_size=self.observation_parameters['batch_size'],
                          callbacks=self.observation_parameters['callbacks'],
                          validation_data=(x_val, y_val),
                          verbose=verbosetemp)
            else:
                # EM case
                # TODO: Change this next line
                verbosetemp = 0
                gamma_em = np.concatenate(self.em['gamma'], axis=0)
                sample_weight_train = gamma_em[indices_train, k]
                sample_weight_val = gamma_em[indices_val, k]
                model.fit(x_train, y_train,
                          sample_weight=sample_weight_train,
                          epochs=self.observation_parameters['epochs'],
                          batch_size=self.observation_parameters['batch_size'],
                          callbacks=self.observation_parameters['callbacks'],
                          validation_data=(x_val, y_val, sample_weight_val),
                          verbose=verbosetemp)
            self.observation_parameters['models'][k] = model
            del model
        return self
