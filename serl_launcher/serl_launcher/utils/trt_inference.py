"""
TensorRT/ONNX inference module for running policy on Jetson Thor.

Two inference modes:
1. ONNX Runtime CPU — lightweight, no GPU needed
2. TensorRT — optimized for Jetson GPU

Usage:
    # ONNX Runtime inference
    engine = OnnxInferenceEngine(model_bytes)
    actions = engine.infer(obs_features)

    # TensorRT inference
    engine = TensorRTEngine(model_bytes)
    engine.build()
    actions = engine.infer(obs_features)
"""
import time
import numpy as np

# ONNX Runtime
try:
    import onnxruntime as ort
except ImportError:
    ort = None

# TensorRT
try:
    import tensorrt as trt
except ImportError:
    trt = None


class OnnxInferenceEngine:
    """Run ONNX model with ONNX Runtime (CPU or TensorRT EP)."""

    def __init__(
        self,
        model_bytes: bytes,
        providers: list = None,
    ):
        """Initialize engine.

        Args:
            model_bytes: Serialized ONNX model.
            providers: ONNX Runtime providers. Default tries TensorRT, CUDA, CPU.
        """
        if ort is None:
            raise ImportError("onnxruntime not installed")

        self.providers = providers or self._default_providers()
        self.session = ort.InferenceSession(
            model_bytes,
            providers=self.providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    @staticmethod
    def _default_providers() -> list:
        """Get best available providers."""
        available = ort.get_available_providers()
        preferred = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        return [p for p in preferred if p in available] or ["CPUExecutionProvider"]

    def infer(self, obs: np.ndarray) -> np.ndarray:
        """Run inference.

        Args:
            obs: Input observation features, shape (N, input_dim).

        Returns:
            Action output, shape (N, action_dim).
        """
        return self.session.run(
            [self.output_name],
            {self.input_name: obs.astype(np.float32)},
        )[0]


class TensorRTEngine:
    """Build and run TensorRT engine from ONNX model.

    For Jetson devices, this provides optimal inference performance.
    """

    def __init__(
        self,
        model_bytes: bytes,
        workspace_size: int = 1 << 30,  # 1 GB
        fp16: bool = True,
    ):
        """Initialize TensorRT engine builder.

        Args:
            model_bytes: Serialized ONNX model.
            workspace_size: Max workspace size in bytes.
            fp16: Enable FP16 inference (supported on Jetson).
        """
        if trt is None:
            raise ImportError("tensorrt not installed")

        self.model_bytes = model_bytes
        self.workspace_size = workspace_size
        self.fp16 = fp16
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = None
        self.context = None
        self._input_idx = 0
        self._output_idx = 1
        self._input_shape = None

    def build(self) -> None:
        """Build TensorRT engine from ONNX model."""
        builder = trt.Builder(self.logger)
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.workspace_size)

        if self.fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # Parse ONNX
        explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(explicit_batch)
        parser = trt.OnnxParser(network, self.logger)

        if not parser.parse(self.model_bytes):
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parse failed: {errors}")

        # Build engine
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("Failed to build TensorRT engine")

        self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(serialized)
        self.context = self.engine.create_execution_context()

        # Cache input/output info
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_idx = i
                self._input_shape = list(self.engine.get_tensor_shape(name))
            else:
                self._output_idx = i

    def infer(self, obs: np.ndarray) -> np.ndarray:
        """Run inference.

        Args:
            obs: Input features, shape (N, input_dim). Batch dim must match engine.

        Returns:
            Action output.
        """
        if self.context is None:
            raise RuntimeError("Engine not built. Call build() first.")

        # Set input shape
        input_name = self.engine.get_tensor_name(self._input_idx)
        output_name = self.engine.get_tensor_name(self._output_idx)

        self.context.set_input_shape(input_name, obs.shape)

        # Allocate buffers
        output_shape = tuple(self.context.get_tensor_shape(output_name))
        output = np.zeros(output_shape, dtype=np.float32)

        # Set tensor addresses (device pointers for GPU, but for Jetson we use bindings)
        self.context.set_tensor_address(input_name, obs.ctypes.data)
        self.context.set_tensor_address(output_name, output.ctypes.data)

        # Execute
        self.context.execute_async_v3(stream_handle=0)  # synchronous on CPU/GPU

        return output


def test_onnx_export():
    """Quick test: verify onnx module is importable."""
    print("ONNX export module loaded successfully")
    print(f"ONNX Runtime: {'available' if ort else 'NOT available'}")
    print(f"TensorRT: {'available' if trt else 'NOT available'}")


if __name__ == "__main__":
    test_onnx_export()
