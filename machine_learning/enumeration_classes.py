from enum import Enum


class Optimizer(Enum):
    SGD = "SGD"
    RMSprop = "RMSprop"
    Adam = "Adam"
    Adadelta = "Adadelta"
    Adagrad = "Adagrad"
    Adamax = "Adamax"
    Nadam = "Nadam"
    Ftrl = "Ftrl"


class ActivationFunctions(Enum):
    relu = "relu"
    sigmoid = "sigmoid"
    softmax = "softmax"
    softplus = "softplus"
    softsign = "softsign"
    tanh = "tanh"
    selu = "selu"
    elu = "elu"


class ActivationFunctionSVM(Enum):
    rbf = "rbf"
    sigmoid = "sigmoid"
    linear = "linear"
    poly = "poly"


class GammaList(Enum):
    scale = "scale"
    auto = "auto"


class LossFunctions(Enum):
    binary_crossentropy = "binary_crossentropy"
    categorical_crossentropy = "categorical_crossentropy"
    sparse_categorical_crossentropy = "sparse_categorical_crossentropy"
    mean_squared_error = "mean_squared_error"
    mean_absolute_error = "mean_absolute_error"
    mean_squared_logarithmic_error = "mean_squared_logarithmic_error"


class WeightList(Enum):
    NONE = None
    Auto = "Auto"
