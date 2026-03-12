"""
TEE Inference Interface for MNIST Secure Inference

This module provides Python bindings for the LoRO secure inference:
1. Key Generation: Generate B, A matrices in TEE
2. Key Export: Get B, A for weight obfuscation (only during provisioning)
3. Key Lock: Lock keys to prevent further export
4. Inference: Compute correction using stored keys

Security Model:
- Provisioning: Export B, A once to compute W' = W + BA
- Runtime: B, A stay in TEE, REE only knows W'
"""

import ctypes
import numpy as np
import os
from typing import Tuple, List, Optional

# Layer configuration for MNIST 3-layer model
LAYER_CONFIGS = [
    (784, 256, 8),   # Layer 1: 784 -> 256, rank=8
    (256, 128, 8),   # Layer 2: 256 -> 128, rank=8
    (128, 10, 4),    # Layer 3: 128 -> 10, rank=4
]

NUM_LAYERS = 3


class TEEInference:
    """
    Interface to TEE for LoRO secure inference.
    """

    def __init__(self, lib_path: str = None):
        """
        Initialize TEE interface.

        Args:
            lib_path: Path to the shared library (default: ../host/libmnist_inference.so)
        """
        if lib_path is None:
            # Default path relative to this file
            lib_path = os.path.join(os.path.dirname(__file__), '..', 'host', 'libmnist_inference.so')

        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"TEE library not found: {lib_path}")

        self.lib = ctypes.cdll.LoadLibrary(lib_path)
        self._setup_functions()

        # Verify TEE connection works
        ret = self.lib.tee_init()
        if ret != 0:
            raise RuntimeError(f"Failed to connect to TEE (error {ret}). Is the TA installed in /lib/optee_armtz/?")
        print("TEE connection verified")

        self._initialized = True

    def _setup_functions(self):
        """Setup ctypes function signatures."""
        # tee_init
        self.lib.tee_init.restype = ctypes.c_int

        # tee_cleanup
        self.lib.tee_cleanup.restype = None

        # tee_generate_keys
        self.lib.tee_generate_keys.restype = ctypes.c_int

        # tee_get_key_status
        self.lib.tee_get_key_status.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.tee_get_key_status.restype = ctypes.c_int

        # tee_export_keys
        self.lib.tee_export_keys.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t)
        ]
        self.lib.tee_export_keys.restype = ctypes.c_int

        # tee_lock_keys
        self.lib.tee_lock_keys.restype = ctypes.c_int

        # tee_loro_inference
        self.lib.tee_loro_inference.argtypes = [
            ctypes.c_int,  # layer_idx
            ctypes.c_int,  # batch_size
            ctypes.POINTER(ctypes.c_float),  # input
            ctypes.POINTER(ctypes.c_float),  # output
        ]
        self.lib.tee_loro_inference.restype = ctypes.c_int

        # py_get_key_status
        self.lib.py_get_key_status.restype = ctypes.c_int

        # py_export_keys
        self.lib.py_export_keys.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int
        ]
        self.lib.py_export_keys.restype = ctypes.c_int

        # py_inference
        self.lib.py_inference.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.py_inference.restype = ctypes.c_int

        # py_full_init (v2 - includes pinned memory)
        self.lib.py_full_init.restype = ctypes.c_int

        # py_full_cleanup
        self.lib.py_full_cleanup.restype = None

        # py_is_pinned
        self.lib.py_is_pinned.restype = ctypes.c_int

    def full_init(self) -> bool:
        """Initialize TEE and pinned memory (v2 optimization)."""
        ret = self.lib.py_full_init()
        return ret == 0

    def full_cleanup(self):
        """Cleanup TEE and pinned memory."""
        self.lib.py_full_cleanup()

    def is_pinned(self) -> bool:
        """Check if using pinned memory."""
        return self.lib.py_is_pinned() == 1

    def get_key_status(self) -> Tuple[bool, bool]:
        """
        Get key status from TEE.

        Returns:
            Tuple of (loaded, locked)
        """
        status = self.lib.py_get_key_status()
        if status < 0:
            raise RuntimeError("Failed to get key status")
        loaded = bool(status & 1)
        locked = bool(status & 2)
        return loaded, locked

    def generate_keys(self) -> bool:
        """
        Generate new LoRO keys in TEE secure storage.

        Returns:
            True if successful
        """
        ret = self.lib.tee_generate_keys()
        return ret == 0

    def export_keys(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Export keys from TEE (only works if not locked).

        Returns:
            List of (B, A) tuples for each layer

        Raises:
            RuntimeError if keys are locked or export fails
        """
        # Check if we can export
        loaded, locked = self.get_key_status()
        if not loaded:
            raise RuntimeError("Keys not loaded")
        if locked:
            raise RuntimeError("Keys are locked - cannot export after provisioning")

        # Calculate buffer size
        # TEE stores: B(output_dim, rank), A(rank, input_dim)
        total_size = 0
        for input_dim, output_dim, rank in LAYER_CONFIGS:
            total_size += (output_dim * rank + rank * input_dim) * 4  # float32

        # Allocate buffer
        buffer = np.zeros(total_size // 4, dtype=np.float32)

        ret = self.lib.py_export_keys(
            buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            total_size
        )

        if ret != 0:
            raise RuntimeError(f"Failed to export keys: {ret}")

        # Parse buffer into B, A matrices for each layer
        # TEE stores: B(output_dim, rank), A(rank, input_dim)
        keys = []
        offset = 0
        for input_dim, output_dim, rank in LAYER_CONFIGS:
            B_size = output_dim * rank
            A_size = rank * input_dim

            B = buffer[offset:offset + B_size].reshape(output_dim, rank).copy()
            offset += B_size

            A = buffer[offset:offset + A_size].reshape(rank, input_dim).copy()
            offset += A_size

            keys.append((B, A))

        return keys

    def lock_keys(self) -> bool:
        """
        Lock keys after provisioning is complete.
        After this, export_keys() will fail.

        Returns:
            True if successful
        """
        ret = self.lib.tee_lock_keys()
        return ret == 0

    def inference(self, layer_idx: int, input_matrix: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Perform LoRO inference for one layer using stored keys.

        Args:
            layer_idx: Layer index (0, 1, or 2)
            input_matrix: Input array (batch_size, input_dim)

        Returns:
            Tuple of (output_correction, elapsed_ms)
        """
        input_dim, output_dim, rank = LAYER_CONFIGS[layer_idx]

        if input_matrix.ndim == 1:
            input_matrix = input_matrix.reshape(1, -1)

        batch_size = input_matrix.shape[0]

        if input_matrix.shape[1] != input_dim:
            raise ValueError(f"Input dimension mismatch: expected {input_dim}, got {input_matrix.shape[1]}")

        # Prepare output buffer
        output = np.zeros(batch_size * output_dim, dtype=np.float32)

        elapsed = self.lib.py_inference(
            layer_idx,
            batch_size,
            input_matrix.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )

        if elapsed < 0:
            raise RuntimeError(f"Inference failed: {elapsed}")

        return output.reshape(batch_size, output_dim), elapsed


class SecureInferenceSession:
    """
    High-level session for secure inference with LoRO.

    Usage:
        # First time setup (provisioning)
        session = SecureInferenceSession()
        session.setup_keys()  # Generates keys and exports them
        B_A_matrices = session.get_deobfuscation_keys()  # For weight obfuscation
        session.lock_keys()  # Lock after provisioning

        # Runtime inference
        session = SecureInferenceSession()
        correction = session.compute_correction(layer_idx, input)
    """

    def __init__(self, lib_path: str = None):
        self.tee = TEEInference(lib_path)
        self._cached_keys: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None

    def setup_keys(self, force_regenerate: bool = False) -> bool:
        """
        Setup LoRO keys. Generates new keys if not present.

        Args:
            force_regenerate: If True, regenerate even if keys exist

        Returns:
            True if new keys were generated
        """
        loaded, locked = self.tee.get_key_status()

        if loaded and locked:
            print("Keys already provisioned and locked")
            return False

        if not loaded or force_regenerate:
            print("Generating new keys in TEE...")
            if not self.tee.generate_keys():
                raise RuntimeError("Failed to generate keys")
            return True

        return False

    def get_deobfuscation_keys(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get B and A matrices for weight obfuscation.
        Only works before keys are locked.

        Returns:
            List of (B, A) tuples for each layer
        """
        if self._cached_keys is not None:
            return self._cached_keys

        keys = self.tee.export_keys()
        self._cached_keys = keys
        return keys

    def lock_keys(self) -> bool:
        """
        Lock keys after provisioning. This prevents further exports.
        """
        return self.tee.lock_keys()

    def compute_correction(self, layer_idx: int, x: np.ndarray) -> np.ndarray:
        """
        Compute correction term for deobfuscation.
        This uses stored keys in TEE - B and A are never exposed.

        Args:
            layer_idx: Layer index (0, 1, or 2)
            x: Input tensor (batch_size, input_dim)

        Returns:
            Correction to subtract: y' - correction = y
        """
        correction, _ = self.tee.inference(layer_idx, x)
        return correction

    def full_forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute all layer corrections in sequence.

        Args:
            x: Input tensor (batch_size, 784)

        Returns:
            List of corrections for each layer
        """
        corrections = []
        current = x

        for i in range(NUM_LAYERS):
            correction = self.compute_correction(i, current)
            corrections.append(correction)
            # Note: In real inference, intermediate activations change
            # This is just for demonstrating the TEE interface

        return corrections


# Test
if __name__ == '__main__':
    print("Testing TEE Inference Interface...")

    try:
        session = SecureInferenceSession()

        loaded, locked = session.tee.get_key_status()
        print(f"Key status: loaded={loaded}, locked={locked}")

        if not loaded:
            print("Generating keys...")
            session.setup_keys()

            print("Exporting keys for obfuscation...")
            keys = session.get_deobfuscation_keys()
            for i, (B, A) in enumerate(keys):
                print(f"  Layer {i}: B{B.shape}, A{A.shape}")

            print("Locking keys...")
            session.lock_keys()

        # Test inference
        print("\nTesting inference...")
        x = np.random.randn(1, 784).astype(np.float32)
        for i, (input_dim, output_dim, _) in enumerate(LAYER_CONFIGS):
            correction, elapsed = session.tee.inference(i, x)
            print(f"  Layer {i}: correction shape {correction.shape}, time {elapsed}ms")
            x = np.random.randn(1, output_dim).astype(np.float32)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please build the TEE library: cd MNIST_DEMO && ./compile.sh")
    except Exception as e:
        print(f"Error: {e}")
