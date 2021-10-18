import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from typing import Optional, List, Dict
from helpers import preprocess, hard2soft_labels, log_sum_exp
from utils import tfp_linear_t
from DBN import DynamicBayesianNetwork
tf.keras.backend.set_floatx('float64')


class DeepHiddenSemiMarkovModel(DynamicBayesianNetwork):

    def __init__(self, K: int, fs: float, ar_order: Optional[int] = 0, normalize: Optional[bool] = True):
        super(DeepHiddenSemiMarkovModel, self).__init__(K, fs, ar_order, normalize)

    def fit_complete_data(self, eeg: List, labels: List, observation_model: Optional[Dict], df_ini: Optional[float],
                          min_duration: Optional[int], max_duration: Optional[int], th_em: Optional[float] = 0.01,
                          max_iterations: Optional[int] = 20, verbose: Optional[int] = 1,
                          random_state: Optional[int] = 42):
        # pre-processing
        self.n_seq = len(eeg)
        self.x, self.y, self.labels, self.n = \
            preprocess(eeg, ar_order=self.ar_order, labels=labels, normalize=self.normalize)
        # Observation model
        self.observation_parameters['name'] = observation_model['name'] if observation_model is not None else 'linear'
        self.observation_parameters['models'] = [None for _ in range(self.K)]
        self.observation_parameters['error_parameters'] = [dict() for _ in range(self.K)]
        self.observation_parameters['error'] = [None for _ in range(self.K)]
        self.observation_parameters['nll'] = list()
        # Posterior conditionals based on experts (i.e, 100% confidence)
        self.em['gamma'] = [np.zeros((self.x[i].shape[0], self.K)) for i in range(len(self.x))]
        for i in range(self.n_seq):
            for k in range(self.K):
                self.em['gamma'][i][:, k] = (1 * (self.labels[i] == k + 1))
        if observation_model is None:
            # Default linear model
            # Linear likelihood is a special case. It does not need an instantiated model
            # Initial expectations
            self.em['w'] = [np.ones((self.x[i].shape[0], self.K)) for i in range(len(self.x))]
            flag_em = True
            nll_prev = 1e10
            df_prev = df_ini
            it = 0
            while flag_em:
                # M-step
                self.obs_max_linear(df_prev=df_prev, complete_data=True)
                df_prev = [self.observation_parameters['error_parameters'][k]['df'] for k in range(self.K)]
                # check for convergence
                if abs(self.observation_parameters['nll'][it] - nll_prev)/abs(nll_prev) <= th_em:
                    flag_em = False
                if it == max_iterations:
                    flag_em = False
                    # TODO: add warning about non-convergence
                # E step
                self.obs_exp_t(complete_data=True)
                nll_prev = self.observation_parameters['nll'][it]
                it += 1
            # either EM converged or maximum iterations reached
            if ~flag_em:
                # Instantiate TFP model
                nll_check = np.zeros(self.K)
                for k in range(self.K):
                    model = tfp_linear_t(self.observation_parameters['error_parameters'][k]['df'],
                                         self.observation_parameters['error_parameters'][k]['scale'],
                                         self.ar_order)
                    # Load weights (kernel and bias)
                    tfp_weights = list()
                    tfp_weights.append(self.observation_parameters['error_parameters'][k]['loc_model'].coef_[..., None])
                    tfp_weights.append(
                        np.asarray(self.observation_parameters['error_parameters'][k]['loc_model'].intercept_)[None, ...])
                    model.get_layer('linear_location').set_weights(tfp_weights)
                    self.observation_parameters['models'][k] = model
                    del model
                    # TODO: when everything works, this code should be deleted
                    idx = np.concatenate(self.em['gamma'], axis=0)[:, k] == 1
                    nll_check[k] = -tf.math.reduce_mean(self.observation_parameters['models'][k](
                        np.concatenate(self.x, axis=0)[idx]).log_prob(
                        np.concatenate(self.y, axis=0)[idx]))
                fl = np.isclose(np.mean(nll_check), self.observation_parameters['nll'][-1])
        else:
            # Non-linear, deep models for distribution parameters
            self.obs_max_non_linear(observation_model, complete_data=True, random_state=random_state, verbose=verbose)
        # Hidden semi-Markov chain model
        ini_state = [int(x[0] - 1) for x in self.labels]
        sum_pi = np.sum(to_categorical(ini_state, num_classes=self.K), axis=0)
        self.state_parameters['pi'] = (sum_pi / self.n_seq).astype(np.float64)
        if min_duration is None or min_duration <= self.ar_order:
            min_duration = self.ar_order
        self.duration_parameters['d_min'] = min_duration
        if max_duration is None:
            # TODO: There's a bug here...
            self.state_parameters['A'] = np.eye(self.K, dtype=np.float64)
            self.state_parameters['p_non_parametric'] = np.zeros((self.K, np.max(self.n)))
            for i in range(self.n_seq):
                idx = np.concatenate((np.asarray(np.nonzero(np.diff(self.labels[i]))).flatten(),
                                      np.array(len(self.labels[i]) - 1)[..., None]))
                idx_diff = np.concatenate((idx[0][..., None], np.diff(idx)))
                for k in range(self.K):
                    counts, _ = np.histogram(idx_diff[self.labels[i][idx] == k + 1],
                                             bins=range(1, np.max(self.n) + 2), density=False)
                    self.duration_parameters['p_non_parametric'][k, :] += counts
            self.duration_parameters['p_non_parametric'] /= \
                np.sum(self.duration_parameters['p_non_parametric'], axis=1)[..., None]
            self.duration_parameters['d_max'] = np.max(np.nonzero(self.duration_parameters['p_non_parametric'])[1])
            self.duration_parameters['p_non_parametric'] = \
                self.duration_parameters['p_non_parametric'][:, 0:self.duration_parameters['d_max'] + 1]
        else:
            self.duration_parameters['d_max'] = int(max_duration)
            self.state_parameters['A'] = np.zeros((self.K, self.K), dtype=np.float64)
            self.duration_parameters['p_non_parametric'] = np.zeros((self.K, max_duration), dtype=np.float64)
            for i in range(self.n_seq):
                idx = np.concatenate((np.asarray(np.nonzero(np.diff(self.labels[i]))).flatten(),
                                      np.array(len(self.labels[i]) - 1)[..., None]))
                idx_diff = np.concatenate((idx[0][..., None] + 1, np.diff(idx)))
                for j in range(len(idx_diff)):
                    self_trans, trans = np.divmod(idx_diff[j], max_duration)
                    if trans == 0:
                        trans = max_duration
                        self_trans -= 1
                    k = int(self.labels[i][idx][j] - 1)
                    # Durations and transitions
                    self.duration_parameters['p_non_parametric'][k, max_duration - 1] += self_trans
                    self.duration_parameters['p_non_parametric'][k, trans - 1] += 1
                    self.state_parameters['A'][k, k] += self_trans
                    if j != len(idx_diff) - 1:
                        k_trans = int(self.labels[i][idx][j + 1] - 1)
                        self.state_parameters['A'][k, k_trans] += 1
            self.state_parameters['A'] /= np.sum(self.state_parameters['A'], axis=1)[..., None]
            # Correct durations less then min_duration
            self.duration_parameters['p_non_parametric'][..., min_duration] += \
                np.sum(self.duration_parameters['p_non_parametric'][..., :min_duration], axis=1)
            self.duration_parameters['p_non_parametric'][..., :min_duration] = np.zeros((self.K, min_duration))
            self.duration_parameters['p_non_parametric'] /= \
                np.sum(self.duration_parameters['p_non_parametric'], axis=1)[..., None]
        return self

    def fit_em(self, eeg, hard_labels=None, soft_labels=None, normalize=True, min_duration=None, max_duration=None,
               initial_state=None, initial_transitions=None, initial_durations=None, prob_band=(0.8, 0.99),
               observation_model=None, df_ini=None, th_em=0.01, max_iterations=20, random_state=42, verbose=1):
        """
        NOTES:
        Observation=None -> additive, homoscedastic (scale and degrees of freedom), linear location model on Observations
        Observations=Model(s) -> parameterizes one or more observation likelihood parameters with a DNN. In the extreme
        case this can yield a heteroscedastic (scale and degrees of freedom), non-linear location model
        """
        self.n_seq = len(eeg)
        self.normalize = normalize
        # Initialize EM variables
        ini_state_max = True
        self.em['gamma'] = [np.zeros((eeg[i].shape[0] - self.ar_order, self.K)) for i in range(self.n_seq)]
        self.em['sum_xi'] = np.zeros((self.K, self.K))

        if hard_labels is not None:
            self.fit_complete_data(eeg, hard_labels, observation_model=None,
                                   min_duration=min_duration, max_duration=max_duration, df_ini=df_ini)
            for i_seq in range(self.n_seq):
                self.em['gamma'][i_seq] = hard2soft_labels(self.em['gamma'][i_seq], low=prob_band[0], high=prob_band[1])
        if soft_labels is not None:
            self.x, self.y, self.labels, self.n = \
                preprocess(eeg, ar_order=self.ar_order, labels=None, normalize=self.normalize)
            self.duration_parameters['d_max'] = max_duration
            if min_duration is None or min_duration <= self.ar_order:
                min_duration = self.ar_order
            self.duration_parameters['d_min'] = min_duration
            for i_seq in range(self.n_seq):
                self.em['gamma'][i_seq] = soft_labels[i_seq][self.idx_ini:, :]
            self.observation_parameters['models'] = [None for _ in range(self.K)]

        # Initialize xi
        max_duration = self.duration_parameters['d_max']
        self.em['xi'] = [np.zeros((eeg[i].shape[0] - self.ar_order, self.K, max_duration)) for i in range(self.n_seq)]
        # Initialize sum_xi
        self.em['sum_xi'] = initial_transitions
        self.state_parameters['A'] = initial_transitions
        # Initialize eta
        self.em['eta_ini'] = [np.zeros((self.K, max_duration)) for _ in range(self.n_seq)]
        df_prev = df_ini
        # Delete learned weights/models
        self.observation_parameters['name'] = 'linear' if observation_model is None else observation_model['name']
        self.duration_parameters['p_non_parametric'] = np.zeros((self.K, max_duration))
        self.observation_parameters['nll'] = list()
        if observation_model is None:
            self.observation_parameters['error_parameters'] = [dict() for _ in range(self.K)]
            self.observation_parameters['error'] = [None for _ in range(self.K)]
            # Initialize posterior conditionals of t dist to ones
            self.em['w'] = [np.ones((self.x[i].shape[0], self.K)) for i in range(len(self.x))]

        # Initial state
        self.state_parameters['pi'] = initial_state
        for i_seq in range(self.n_seq):
            if initial_state is not None:
                self.em['gamma'][i_seq][0, :] = initial_state

        # Initialize durations
        if initial_durations is None:
            self.duration_parameters['p_non_parametric'][:, self.idx_ini:].fill(1)
            self.duration_parameters['p_non_parametric'] /= np.sum(self.duration_parameters['p_non_parametric'], axis=1)[..., None]
        else:
            self.duration_parameters['p_non_parametric'] = initial_durations

        flag_em = True
        nll_prev = 1e10
        it = 0
        while flag_em:
            if observation_model is None:
                """
                # E step for hidden variables of markov chain
                self._posterior_marginals()
                """
                # M step for Observation likelihood - Linear location
                self.obs_max_linear(df_prev=df_prev)                   # Observations
                df_prev = [self.observation_parameters['error_parameters'][k]['df'] for k in range(self.K)]
                for k in range(self.K):
                    model = tfp_linear_t(self.observation_parameters['error_parameters'][k]['df'],
                                         self.observation_parameters['error_parameters'][k]['scale'],
                                         self.ar_order)
                    # Load weights (kernel and bias)
                    tfp_weights = list()
                    tfp_weights.append(self.observation_parameters['error_parameters'][k]['loc_model'].coef_[..., None])
                    tfp_weights.append(np.asarray(
                        self.observation_parameters['error_parameters'][k]['loc_model'].intercept_)[np.newaxis, ...])
                    model.get_layer('linear_location').set_weights(tfp_weights)
                    self.observation_parameters['models'][k] = model
            else:
                # "local" M step for Observation likelihood - Non linear case
                self.obs_max_non_linear(observation_model, complete_data=False, random_state=random_state,
                                        verbose=verbose)
            # M-step for States likelihood
            self.states_max(ini=ini_state_max)
            ini_state_max = False

            # E step for hidden variables of markov chain
            self._posterior_marginals()

            if verbose == 1:
                print('EM iteration {:2d}, negative log-likelihood : {:4.3f}'.format(it+1, self.nll[-1]))
            if observation_model is None:
                # E step for hidden variables of Observations likelihood - Linear case
                # NOTE: non-linear case is implicitly handled by minimizing the nll of the Observations likelihood
                self.obs_exp_t()
            if abs(self.nll[it] - nll_prev) / abs(nll_prev) <= th_em:
                flag_em = False
                self.converged = True
                if verbose == 1:
                    print('EM converged')
            if it == max_iterations:
                flag_em = False
            nll_prev = self.nll[it]
            it += 1
            if flag_em is False:
                # clean up
                self.em = dict()
                self.x = None
                self.y = None
        return self

    def posterior_mode(self, eeg):
        logp_z_ini, logp_dz, logA = self._probs_inference()
        max_duration = self.duration_parameters['d_max']
        n_seq = len(eeg)
        labels = list()
        log_like = list()
        duration_left = list()
        shp = np.array([self.K, self.K, max_duration])
        dim_idx = list(np.ix_(*[np.arange(i) for i in shp[1:]]))
        dim_idx.insert(0, np.inf)
        for i_seq in range(n_seq):
            x, y, _, n = preprocess([eeg[i_seq]], ar_order=self.ar_order, normalize=self.normalize)
            logp_yz = -np.inf * np.ones((n[0], self.K, 1))
            for k in range(self.K):
                model = self.observation_parameters['models'][k]
                logp_yz[self.idx_ini:, k, :] = model(x[0]).log_prob(y[0]).numpy()
            # Forward algorithm - max-product algorithm
            aux_idx_z = np.repeat(np.arange(self.K)[..., None], max_duration-1, axis=1)
            aux = np.repeat(np.arange(1, max_duration)[None, ...], self.K, axis=0)
            psi_z = np.zeros((n[0], self.K, max_duration), dtype=np.uint16)
            psi_d = np.zeros((n[0], self.K, max_duration), dtype=np.uint16)
            delta_prev = np.squeeze(logp_yz[self.idx_ini, ...] + logp_dz + logp_z_ini)
            for i in range(self.idx_ini + 1, n[0]):
                pre_max = (delta_prev[..., 0:1] + logA)
                aux_idx = np.argmax(pre_max, axis=0)
                psi_z[i, ...] = np.concatenate((aux_idx[..., None], aux_idx_z), axis=1).astype(np.uint16)
                pre_max = np.stack((np.squeeze(logp_dz + pre_max[aux_idx, np.arange(self.K)][..., None]),
                                   np.concatenate((delta_prev[:, 1:], -np.inf * (np.ones((self.K, 1)))), axis=1)),
                                   axis=0)
                idx_max = np.argmax(pre_max, axis=0)
                dim_idx[0] = idx_max
                idx_temp = idx_max.astype(np.float64)
                idx_dmax = idx_temp[:, -1]
                idx_temp = idx_temp[:, :-1]
                idx_nan = idx_temp == 0
                idx_temp[idx_nan] = np.nan
                idx_temp[~idx_nan] = 0
                idx_temp = idx_temp + aux
                idx_temp[np.isnan(idx_temp)] = 0
                psi_d[i, ...] = np.concatenate((idx_temp, idx_dmax[..., None]), axis=-1).astype(np.uint16)
                delta_prev = logp_yz[i, ...] + pre_max[tuple(dim_idx)]
            log_like_idx = np.argmax(delta_prev)
            log_like_idx = np.unravel_index(log_like_idx, delta_prev.shape)
            log_like.append(delta_prev[log_like_idx])
            # Back-tracking
            z = -1*np.ones(n[0], dtype=np.uint16)
            d_rem = -1*np.ones(n[0], dtype=np.uint16)
            z[-1] = log_like_idx[0]
            d_rem[-1] = log_like_idx[1]
            for i in range(n[0] - 2, self.idx_ini - 1, -1):
                d_rem[i] = psi_d[i+1, z[i+1], d_rem[i+1]]
                z[i] = psi_z[i+1, z[i+1], d_rem[i]]
            z += 1
            # "Extend" initial label to first samples
            z[:self.idx_ini] = z[self.idx_ini]
            labels.append(z)
            duration_left.append(d_rem)
        return labels, log_like, duration_left

    def log_prob(self, eeg):
        logp_z_ini, logp_dz, logA = self._probs_inference()
        max_duration = self.duration_parameters['d_max']
        n_seq = len(eeg)
        log_like = list()
        for i_seq in range(n_seq):
            x, y, _, n = preprocess([eeg[i_seq]], ar_order=self.ar_order, normalize=self.normalize)
            logp_yz = -np.inf * np.ones((n[0], self.K, 1))
            for k in range(self.K):
                model = self.observation_parameters['models'][k]
                logp_yz[self.idx_ini:, k, :] = model(x[0]).log_prob(y[0]).numpy()
                del model
            # Forward algorithm - sum-product algorithm
            logalpha_em_prev = logp_yz[self.idx_ini, ...] + logp_dz + logp_z_ini
            aux = logp_yz + logp_dz[..., -1:]
            for i in range(self.idx_ini+1, n[0]):
                logalpha_aux_dmin = log_sum_exp(logalpha_em_prev[..., 0:1] + logA, axis=1)[..., None]
                logalpha_em_update_dmax = aux[i: i + 1, ...] + logalpha_aux_dmin
                logalpha_em_update_d = logp_yz[i, ...] + \
                                       log_sum_exp(np.concatenate((logp_dz[..., :max_duration - 1] + logalpha_aux_dmin,
                                                                  logalpha_em_prev[..., 1:]), axis=0), axis=0)
                logalpha_em_prev = np.concatenate((logalpha_em_update_d[None, ...], logalpha_em_update_dmax), axis=-1)
            log_like.append(log_sum_exp(logalpha_em_prev))
        return log_like

    def posterior_marginals(self, eeg):
        logp_z_ini, logp_dz, logA = self._probs_inference()
        logAext = logA[..., None]
        logAT = np.log(self.state_parameters['A'].T)
        max_duration = self.duration_parameters['d_max']
        n_seq = len(eeg)
        gamma = list()
        for i_seq in range(n_seq):
            x, y, _, n = preprocess([eeg[i_seq]], ar_order=self.ar_order, normalize=self.normalize)
            logp_yz = -np.inf * np.ones((n[0], self.K, 1))
            for k in range(self.K):
                model = self.observation_parameters['models'][k]
                logp_yz[self.idx_ini:, k, :] = model(x[0]).log_prob(y[0]).numpy()
                del model
            logalpha_em = np.zeros((n[0], self.K, max_duration))
            logalpha_em = self._log_prob(logalpha_em, logp_z_ini, logA, logp_yz, logp_dz, self.idx_ini, max_duration)
            log_like_seq = log_sum_exp(logalpha_em[-1, :])
            gamma_seq = np.zeros((n[0] - self.idx_ini, self.K))
            # Backwards
            logalphamin_duration = np.array(logalpha_em[..., 0].T)[..., None]
            logalpha_em -= log_like_seq                # in place substraction of loglikelihood for computations
            # First iteration
            logbeta_prev = np.zeros((self.K, max_duration))
            gamma_seq[-1, :] = np.sum(np.exp(logalpha_em[-1, ...] + logbeta_prev), axis=1)
            for i in range(n[0] - 2, self.idx_ini - 1, -1):
                logbeta_aux = log_sum_exp(logbeta_prev + logp_dz, axis=-1).T
                logbeta_update_dmin = log_sum_exp(logp_yz[i + 1, ...] + logAT + logbeta_aux, axis=0)  #
                logbeta_update_rest = logp_yz[i + 1, ...] + logbeta_prev[:, 0:max_duration - 1]  #
                logbeta_prev = np.concatenate((logbeta_update_dmin[..., None], logbeta_update_rest), axis=1)
                gamma_seq[i - self.ar_order, :] = (np.exp(logalpha_em[i, ...] + logbeta_prev)).sum(axis=1)
            gamma.append(np.concatenate((np.zeros((self.idx_ini, self.K)), gamma_seq), axis=0))
        return gamma

    def posterior_gamma(self, eeg):
        # Computes the expected value (mean) of the posterior conditional gamma variable
        n_seq = len(eeg)
        w = list()
        df = list()
        for i_seq in range(n_seq):
            x, y, _, n = preprocess([eeg[i_seq]], ar_order=self.ar_order, normalize=self.normalize)
            w_seq = np.zeros((n[0], self.K))
            df_seq = np.zeros((n[0] - self.ar_order, self.K))
            for k in range(self.K):
                model = self.observation_parameters['models'][k]
                t_dist = model(x[0])
                df_seq[:, k] = t_dist.df.numpy().flatten()
                w_seq[self.idx_ini:, k] = \
                    ((t_dist.df + 1) / (t_dist.df + tf.math.square((y[0] - t_dist.loc) / t_dist.scale))).numpy().flatten()
                del model
            w.append(w_seq)
            df.append(df_seq)
        return w, df

    def states_max(self, ini):
        # pi - initial state
        sum_pi = np.zeros(self.K)
        for i_seq in range(self.n_seq):
            sum_pi += self.em['gamma'][i_seq][0, :]
        self.state_parameters['pi'] = sum_pi/np.sum(sum_pi)
        # Transition matrix
        self.state_parameters['A'] = self.em['sum_xi']/np.sum(self.em['sum_xi'], axis=1)[..., None]
        if ini is False:
            if 'p_non_parametric' in self.duration_parameters.keys():
                for k in range(self.K):
                    temp = np.zeros(self.duration_parameters['d_max'])
                    for x, y in zip(self.em['xi'], self.em['eta_ini']):
                        temp += x[:, k, :].sum(axis=0) + y[k, :]
                    self.duration_parameters['p_non_parametric'][k, :] = temp/np.sum(temp)
        return self

    def _probs_inference(self):
        with np.errstate(divide='ignore'):
            logp_z_ini = np.expand_dims(np.log(self.state_parameters['pi']), axis=(0, 2))
            logp_dz = np.expand_dims(np.log(self.duration_parameters['p_non_parametric']), axis=0)
            logA = np.log(self.state_parameters['A'])
        return logp_z_ini, logp_dz, logA

    def _posterior_marginals(self):
        logp_z_ini, logp_dz, logA = self._probs_inference()
        logA_ext = logA[..., None]
        logAT = np.log(self.state_parameters['A'].T)
        max_duration = self.duration_parameters['d_max']
        log_like = list()
        for i_seq in range(self.n_seq):
            n = self.n[i_seq]
            logp_yz = -np.inf * np.ones((n, self.K, 1))
            for k in range(self.K):
                model = self.observation_parameters['models'][k]
                logp_yz[self.idx_ini:, k, :] = model(self.x[i_seq]).log_prob(self.y[i_seq]).numpy()
            logalpha_em = np.zeros((self.n[i_seq], self.K, max_duration))
            logalpha_em = self._log_prob(logalpha_em, logp_z_ini, logA, logp_yz, logp_dz, self.idx_ini, max_duration)
            log_like_seq = log_sum_exp(logalpha_em[-1, :])
            log_like.append(log_like_seq)

            # Backwards
            logalpha_min_duration = np.array(logalpha_em[..., 0].T)[..., None]
            logalpha_em -= log_like_seq                # in place subtraction of log-likelihood for computations
            logbeta_up = -np.inf * np.ones((self.K, max_duration))
            # First iteration
            logbeta_prev = np.zeros((self.K, max_duration))
            self.em['gamma'][i_seq][-1, :] = np.sum(np.exp(logalpha_em[-1, ...] + logbeta_prev), axis=1)
            aux = np.exp(logalpha_min_duration[:, -2:-1, :] + logp_yz[-1:, ...] +
                         logp_dz + logA_ext + logbeta_prev[None, ...] - log_like_seq)
            self.em['sum_xi'] += np.sum(aux, axis=-1)
            self.em['xi'][i_seq][-1, ...] = np.sum(aux, axis=0)
            temp = logp_dz + logA_ext - log_like_seq

            for i in range(self.n[i_seq] - 2, self.idx_ini - 1, -1):
                logbeta_aux = log_sum_exp(logbeta_prev + logp_dz, axis=-1).T
                logbeta_update_d_min = log_sum_exp(logp_yz[i + 1, ...] + logAT + logbeta_aux, axis=0)
                logbeta_update_rest = logp_yz[i + 1, ...] + logbeta_prev[:, 0:max_duration - 1]
                logbeta_prev = np.concatenate((logbeta_update_d_min[..., None], logbeta_update_rest), axis=1)
                self.em['gamma'][i_seq][i - self.ar_order, :] = (np.exp(logalpha_em[i, ...] + logbeta_prev)).sum(axis=1)
                if i > self.idx_ini:
                    aux = np.exp(logalpha_min_duration[:, i-1:i, :] + logp_yz[i:i+1, ...] +
                                 temp + logbeta_prev[None, ...])
                    self.em['sum_xi'] += aux.sum(axis=-1)
                    self.em['xi'][i_seq][i - self.ar_order, ...] = aux.sum(axis=0)
            self.em['eta_ini'][i_seq] = np.exp(logalpha_em[self.idx_ini, ...] + logbeta_prev)
        self.nll.append(-np.mean(np.array(log_like) / np.array(self.n)))     # normalize by eeg duration
        return self

    @staticmethod
    def _log_prob(logalphaEM, logp_z_ini, logA, logp_yz, logp_dz, idx_ini, max_duration):
        logalphaEMprev = logp_yz[idx_ini, ...] + logp_dz + logp_z_ini
        logalphaEM[idx_ini:idx_ini + 1, ...] = logalphaEMprev
        aux = logp_yz + logp_dz[..., -1:]
        for i in range(idx_ini + 1, logalphaEM.shape[0]):
            logalphaauxdmin = log_sum_exp(logalphaEMprev[..., 0:1] + logA, axis=1)[..., None]
            logalphaEMposdmax = aux[i: i + 1, ...] + logalphaauxdmin
            logalphaEMposd = logp_yz[i, ...] + \
                             log_sum_exp(np.concatenate((logp_dz[..., :max_duration - 1] + logalphaauxdmin,
                                                         logalphaEMprev[..., 1:]), axis=0), axis=0)
            logalphaEMprev = np.concatenate((logalphaEMposd[None, ...], logalphaEMposdmax), axis=-1)
            logalphaEM[i:i + 1, ...] = logalphaEMprev
        return logalphaEM
