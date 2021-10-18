import tensorflow as tf
from tensorflow.keras import Sequential as Sequential
from tensorflow.keras.layers import Dense, InputLayer, Flatten
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
tf.keras.backend.set_floatx('float64')


def tfp_linear_t(df, scale, ar_order):
    """
    TFP model for the observation likelihood - t distribution
    The location parameter is parameterized by a shallow neural network,
    while the scale and degrees of freedom are kept constant: homogeneity of scale and degrees of freedom
    , i.e. homoscedasticity (at least for the scale)
    The resulting observation likelihood is an additive linear generalized t model
    """
    model = Sequential([
        InputLayer(input_shape=(ar_order,)),
        Flatten(),
        Dense(1, name='linear_location'),
        tfp.layers.DistributionLambda(
            lambda t: tfd.StudentT(df=df,
                                   loc=t,
                                   scale=scale),
            convert_to_tensor_fn=tfp.distributions.Distribution.sample)
        ])
    return model

