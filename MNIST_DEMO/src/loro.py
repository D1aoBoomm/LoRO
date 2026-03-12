"""
LoRO: Low-Rank Obfuscation for Secure Inference

For PyTorch Linear layers, weights are stored as W (output_dim, input_dim).
Forward pass: y = x @ W.T + b

LoRO obfuscation: W' = W + B @ A
- W: (output_dim, input_dim)
- B: (output_dim, r)
- A: (r, input_dim)
- W': (output_dim, input_dim) - obfuscated weight

During inference:
- REE computes: y' = x @ W'.T + b (using obfuscated weights)
- TEE computes: correction = x @ A.T @ B.T
- Final result: y = y' - correction
"""

import torch


def generate_low_rank_mask(output_dim, input_dim, r, seed=None):
    """
    Generate low-rank matrices B and A for obfuscation.
    BA has shape (output_dim, input_dim) with rank r.

    Note: PyTorch Linear weights are stored as (output_dim, input_dim).

    Args:
        output_dim: output dimension
        input_dim: input dimension
        r: rank (should be small, e.g., 8-32)
        seed: random seed for reproducibility

    Returns:
        B: (output_dim, r) matrix
        A: (r, input_dim) matrix
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Generate B and A with small random values
    B = torch.randn(output_dim, r) * 0.1
    A = torch.randn(r, input_dim) * 0.1

    return B, A


def obfuscate_weight(W, B, A):
    """
    Obfuscate weight matrix W with low-rank mask BA.

    W' = W + B @ A

    Args:
        W: original weight (output_dim, input_dim)
        B: low-rank matrix (output_dim, r)
        A: low-rank matrix (r, input_dim)

    Returns:
        W_prime: obfuscated weight (output_dim, input_dim)
    """
    BA = B @ A
    W_prime = W + BA
    return W_prime


def compute_tee_correction(x, A, B):
    """
    Compute the correction term in TEE.

    For y = x @ W.T, the correction for BA is:
    correction = x @ (B @ A).T = x @ A.T @ B.T

    Args:
        x: input tensor (batch_size, input_dim)
        A: low-rank matrix (r, input_dim)
        B: low-rank matrix (output_dim, r)

    Returns:
        correction: (batch_size, output_dim)
    """
    # Step 1: IR = x @ A.T (batch_size, r)
    IR = x @ A.T
    # Step 2: correction = IR @ B.T (batch_size, output_dim)
    correction = IR @ B.T
    return correction


def deobfuscate_output(y_prime, correction):
    """
    Deobfuscate the output.

    y = y' - correction

    Args:
        y_prime: obfuscated output (batch_size, output_dim)
        correction: correction term from TEE (batch_size, output_dim)

    Returns:
        y: deobfuscated output (batch_size, output_dim)
    """
    return y_prime - correction


class LoROLinearLayer:
    """
    LoRO-enabled Linear Layer for secure inference.
    """

    def __init__(self, W, b, r, seed=None):
        """
        Initialize LoRO layer.

        Args:
            W: weight matrix (output_dim, input_dim) - PyTorch format
            b: bias vector (output_dim,)
            r: rank for low-rank mask
            seed: random seed
        """
        self.W = W.clone()
        self.b = b.clone() if b is not None else None
        self.r = r

        output_dim, input_dim = W.shape
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.B, self.A = generate_low_rank_mask(output_dim, input_dim, r, seed)
        self.W_prime = obfuscate_weight(W, self.B, self.A)

    def ree_inference(self, x):
        """
        Perform inference in REE (untrusted environment) with obfuscated weights.

        Args:
            x: input tensor (batch_size, input_dim)

        Returns:
            y_prime: obfuscated output (batch_size, output_dim)
        """
        # y' = x @ W'.T + b
        y_prime = x @ self.W_prime.T
        if self.b is not None:
            y_prime = y_prime + self.b
        return y_prime

    def tee_inference(self, x):
        """
        Compute correction term (simulated TEE execution).

        In real deployment, this would be executed inside TEE.

        Args:
            x: input tensor (batch_size, input_dim)

        Returns:
            correction: (batch_size, output_dim)
        """
        return compute_tee_correction(x, self.A, self.B)

    def full_inference(self, x):
        """
        Full inference with deobfuscation.

        Args:
            x: input tensor (batch_size, input_dim)

        Returns:
            y: deobfuscated output (batch_size, output_dim)
        """
        y_prime = self.ree_inference(x)
        correction = self.tee_inference(x)
        y = deobfuscate_output(y_prime, correction)
        return y

    def get_ree_weights(self):
        """Get weights for REE (obfuscated)"""
        return self.W_prime, self.b

    def get_tee_matrices(self):
        """Get matrices for TEE (A and B)"""
        return self.A, self.B


class LoROMNIST:
    """
    Full LoRO-protected MNIST model with 3 linear layers.
    """

    def __init__(self, weights_list, r_list, seed=None):
        """
        Initialize LoRO MNIST model.

        Args:
            weights_list: list of (weight, bias) tuples for each layer
            r_list: list of ranks for each layer
            seed: random seed
        """
        assert len(weights_list) == 3, "Must have 3 layers"
        assert len(r_list) == 3, "Must have 3 ranks"

        self.layers = []
        for i, ((W, b), r) in enumerate(zip(weights_list, r_list)):
            layer_seed = seed + i if seed is not None else None
            loro_layer = LoROLinearLayer(W, b, r, layer_seed)
            self.layers.append(loro_layer)

    def ree_forward(self, x, activations=True):
        """
        Forward pass in REE with obfuscated weights.

        Args:
            x: input tensor (batch_size, 784)
            activations: whether to apply ReLU

        Returns:
            outputs: list of outputs from each layer
        """
        outputs = []
        for i, layer in enumerate(self.layers):
            y = layer.ree_inference(x)
            # Apply ReLU for layers 1 and 2
            if activations and i < 2:
                y = torch.relu(y)
            outputs.append(y)
            x = y
        return outputs

    def tee_forward(self, x):
        """
        Compute corrections for all layers in TEE.

        Args:
            x: original input tensor (batch_size, 784)

        Returns:
            corrections: list of correction terms for each layer
        """
        corrections = []
        current_input = x
        for layer in self.layers:
            correction = layer.tee_inference(current_input)
            corrections.append(correction)
        return corrections

    def full_forward(self, x):
        """
        Full forward pass with deobfuscation.

        Args:
            x: input tensor (batch_size, 784)

        Returns:
            output: final logits (batch_size, 10)
        """
        current_input = x
        for i, layer in enumerate(self.layers):
            y_prime = layer.ree_inference(current_input)
            correction = layer.tee_inference(current_input)
            y = deobfuscate_output(y_prime, correction)
            # Apply ReLU for layers 1 and 2
            if i < 2:
                y = torch.relu(y)
            current_input = y
        return current_input

    def get_all_ree_weights(self):
        """Get all obfuscated weights for REE"""
        return [layer.get_ree_weights() for layer in self.layers]

    def get_all_tee_matrices(self):
        """Get all A, B matrices for TEE"""
        return [layer.get_tee_matrices() for layer in self.layers]
