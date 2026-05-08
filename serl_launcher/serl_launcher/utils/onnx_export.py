"""
Export Flax policy network to ONNX format for TensorRT inference on Jetson.

Usage:
    from serl_launcher.utils.onnx_export import export_policy_to_onnx, policy_to_numpy

    # Export the deterministic mode of a Flax policy
    onnx_bytes = export_policy_to_onnx(agent, example_obs)

    # Or extract params as numpy dict for manual inference
    numpy_params = policy_to_numpy(agent.state.params)
"""
import io
from typing import Dict, Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto
    _onnx_available = True
except ImportError:
    _onnx_available = False


def extract_mlp_weights(params: Dict, prefix: str = "network") -> Dict[str, np.ndarray]:
    """Extract MLP Dense layer weights from Flax params as numpy arrays.

    Args:
        params: Flax params dict (nested FrozenDict or dict).
        prefix: Key prefix for the MLP network in the params dict.

    Returns:
        Dict mapping layer names to {"kernel": np.array, "bias": np.array}.
    """
    weights = {}
    network = params.get(prefix, params)
    for key, val in network.items():
        if isinstance(val, dict) and "kernel" in val:
            weights[key] = {
                "kernel": np.asarray(val["kernel"]),
                "bias": np.asarray(val.get("bias", np.zeros(val["kernel"].shape[-1]))),
            }
        elif isinstance(val, dict):
            # Recursively search for Dense layers
            sub = extract_mlp_weights(val, prefix="")
            weights.update(sub)
    return weights


def _require_onnx():
    if not _onnx_available:
        raise ImportError("The 'onnx' package is required for ONNX export. Install with: pip install onnx")


def build_mlp_onnx_graph(
    input_dim: int,
    output_dim: int,
    weights: Dict[str, Dict[str, np.ndarray]],
    hidden_dims: list,
) -> bytes:
    """Build an ONNX model for an MLP with ReLU activations.

    Args:
        input_dim: Input feature dimension.
        output_dim: Output action dimension.
        weights: Dict of {"kernel": ..., "bias": ...} per layer.
        hidden_dims: List of hidden layer dimensions.

    Returns:
        Serialized ONNX model bytes.
    """
    _require_onnx()
    nodes = []
    inputs = [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [None, input_dim])]
    outputs = [helper.make_tensor_value_info("action", TensorProto.FLOAT, [None, output_dim])]

    layer_names = [f"Dense_{i}" for i in range(len(hidden_dims) + 1)]
    prev_output = "obs"

    for i, name in enumerate(layer_names):
        if name not in weights:
            raise KeyError(f"Layer {name} not found in weights. Available: {list(weights.keys())}")

        w = weights[name]
        is_last = i == len(layer_names) - 1

        # Dense weight matrix (output_dim, input_dim) in ONNX format
        w_name = f"{name}_W"
        b_name = f"{name}_b"
        matmul_out = f"{name}_mm"
        add_out = f"{name}_out" if is_last else f"{name}_relu"

        # Transpose kernel from Flax (input_dim, output_dim) to ONNX (output_dim, input_dim)
        kernel_t = np.ascontiguousarray(w["kernel"].T, dtype=np.float32)
        bias = np.ascontiguousarray(w["bias"], dtype=np.float32)

        nodes.append(
            helper.make_node(
                "MatMul",
                inputs=[prev_output, w_name],
                outputs=[matmul_out],
                name=f"{name}_MatMul",
            )
        )
        nodes.append(
            helper.make_node(
                "Add",
                inputs=[matmul_out, b_name],
                outputs=[add_out],
                name=f"{name}_Add",
            )
        )

        # Initializers
        inputs.append(
            helper.make_tensor_value_info(w_name, TensorProto.FLOAT, kernel_t.shape)
        )
        inputs.append(
            helper.make_tensor_value_info(b_name, TensorProto.FLOAT, bias.shape)
        )

        if not is_last:
            relu_out = f"{name}_relu_out"
            nodes.append(
                helper.make_node("Relu", inputs=[add_out], outputs=[relu_out], name=f"{name}_Relu")
            )
            prev_output = relu_out
        else:
            prev_output = add_out

    # Create the graph
    graph = helper.make_graph(
        nodes,
        "policy_network",
        inputs[:1],  # only "obs" is real input, weights are initializers
        outputs,
        initializer=[
            helper.make_tensor(
                f"{name}_W",
                TensorProto.FLOAT,
                weights[name]["kernel"].T.shape,
                np.ascontiguousarray(weights[name]["kernel"].T, dtype=np.float32).tobytes(),
                raw=True,
            )
            for name in layer_names
        ]
        + [
            helper.make_tensor(
                f"{name}_b",
                TensorProto.FLOAT,
                weights[name]["bias"].shape,
                np.ascontiguousarray(weights[name]["bias"], dtype=np.float32).tobytes(),
                raw=True,
            )
            for name in layer_names
        ],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)

    buf = io.BytesIO()
    onnx.save(model, buf)
    return buf.getvalue()


def export_policy_to_onnx(
    agent: Any,
    example_obs: Dict[str, Any],
    input_dim: Optional[int] = None,
) -> bytes:
    """Export the deterministic mode of a Flax policy network to ONNX.

    This function traces the JIT-compiled policy and extracts the weights
    to build an ONNX model for the MLP part of the policy network.

    For pixel-based policies, the image encoder is NOT included in the ONNX
    export. Run the encoder separately on the device.

    Args:
        agent: The SACAgent (or similar) instance.
        example_obs: Example observation dict for tracing.
        input_dim: Override input dimension. If None, inferred from params.

    Returns:
        Serialized ONNX model bytes.

    Raises:
        ValueError: If the policy has an image encoder and input_dim is not provided.
        RuntimeError: If ONNX export fails.
    """
    _require_onnx()
    # Get the policy params
    params = agent.state.params.get("actor", agent.state.params)

    # Check if there's an encoder
    has_encoder = "encoder" in params or any("encoder" in k for k in _flatten_keys(params))

    if has_encoder and input_dim is None:
        # Try to infer from trace
        @jax.jit
        def get_features(obs):
            return agent.state.apply_fn(
                {"params": agent.state.params}, obs, name="actor", train=False,
            ).mode()

        try:
            features = get_features(example_obs)
            input_dim = features.shape[-1]
        except Exception:
            raise ValueError(
                "Policy has image encoder. Provide input_dim manually "
                "(the feature dimension after the encoder)."
            )

    if input_dim is None:
        # Find the first Dense layer input dimension from params
        weights = extract_mlp_weights(params, prefix="network")
        if not weights:
            weights = extract_mlp_weights(params, prefix="")
        layer_name = sorted(weights.keys())[0]
        input_dim = weights[layer_name]["kernel"].shape[0]

    # Extract weights
    weights = extract_mlp_weights(params, prefix="network")
    if not weights:
        weights = extract_mlp_weights(params, prefix="")

    if not weights:
        raise RuntimeError(f"No Dense layer weights found in params: {list(params.keys())}")

    # Infer hidden dims from weights
    sorted_names = sorted(weights.keys())
    hidden_dims = [weights[n]["kernel"].shape[-1] for n in sorted_names[:-1]]
    output_dim = weights[sorted_names[-1]]["kernel"].shape[-1]

    return build_mlp_onnx_graph(input_dim, output_dim, weights, hidden_dims)


def policy_to_numpy(params: Dict) -> Dict:
    """Convert JAX policy params to numpy dict for manual inference.

    Args:
        params: Flax params dict (nested FrozenDict).

    Returns:
        Nested dict with numpy arrays.
    """
    return jax.tree_util.tree_map(np.asarray, params)


def _flatten_keys(d, prefix=""):
    """Yield all leaf keys in a nested dict."""
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            yield from _flatten_keys(v, key)
        else:
            yield key
