"""
Sleep Stage Classification - Models Package

Available models:
- CNN1D: Basic 1D CNN for single-epoch classification
- CNN1DPaper: Paper replication of CNN architecture
- DeepSleepNet: CNN + BiLSTM hybrid for sequence learning (best for N1 detection)
- SleepTransformer: CNN + Transformer hybrid for sequence learning
"""

from .cnn1d import CNN1D, CNN1DPaper, get_model

# DeepSleepNet (CNN + BiLSTM)
from .deepsleepnet import (
    DeepSleepNet,
    get_deepsleepnet,
    DEEPSLEEPNET_CONFIG_4GB,
    DEEPSLEEPNET_CONFIG_8GB,
    DEEPSLEEPNET_CONFIG_FULL,
    DEEPSLEEPNET_CONFIG_100HZ
)

# SleepTransformer (CNN + Transformer)
from .sleep_transformer import (
    SleepTransformer,
    get_sleep_transformer,
    SLEEPTRANSFORMER_CONFIG_DEFAULT
)

__all__ = [
    'CNN1D',
    'CNN1DPaper', 
    'DeepSleepNet',
    'SleepTransformer',
    'get_model',
    'get_deepsleepnet',
    'get_sleep_transformer',
    'DEEPSLEEPNET_CONFIG_4GB',
    'DEEPSLEEPNET_CONFIG_8GB',
    'DEEPSLEEPNET_CONFIG_FULL',
    'DEEPSLEEPNET_CONFIG_100HZ',
    'SLEEPTRANSFORMER_CONFIG_DEFAULT'
]
