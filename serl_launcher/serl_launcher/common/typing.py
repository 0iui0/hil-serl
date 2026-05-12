from typing import Any, Callable, Dict, Optional, Sequence, Union

import flax
import jax.numpy as jnp
import numpy as np

try:
    import tensorflow as tf
    Array = Union[np.ndarray, jnp.ndarray, tf.Tensor]
except ImportError:
    Array = Union[np.ndarray, jnp.ndarray]

PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]
Data = Union[Array, Dict[str, "Data"]]
Batch = Dict[str, Data]
# A method to be passed into TrainState.__call__
ModuleMethod = Union[str, Callable, None]
