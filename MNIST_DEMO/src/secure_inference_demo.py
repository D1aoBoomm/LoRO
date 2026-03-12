"""
Secure Inference Demo for MNIST with LoRO

This demonstrates the complete LoRO secure inference process:
1. Provisioning Phase:
   - Generate B, A matrices in TEE
   - Export B, A (once) to compute obfuscated weights W' = W + BA
   - Lock keys to prevent future export

2. Inference Phase:
   - REE computes y' = x @ W' (using obfuscated weights)
   - TEE computes correction = x @ B @ A (using stored keys)
   - Final result: y = y' - correction

The key insight: B and A matrices are NEVER exposed to REE during inference.
Only during provisioning can they be exported (by the trusted model owner).
"""

import torch
import time
import os
import argparse
import threading
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from model import MNISTNet
from loro import LoROMNIST

# Layer configuration
LAYER_CONFIGS = [
    (784, 256, 8),
    (256, 128, 8),
    (128, 10, 4),
]


class SecureMNISTDemo:
    """
    Complete LoRO secure inference demo with proper key management.
    """

    def __init__(self, weights_path=None, use_tee=False, use_gpu=True):
        """
        Initialize the demo.

        Args:
            weights_path: Path to model weights (default: ../weights/)
            use_tee: Whether to use real TEE (vs simulation)
            use_gpu: Whether to use GPU for REE inference
        """
        self.use_tee = use_tee

        # Default weights path relative to this file
        if weights_path is None:
            weights_path = os.path.join(os.path.dirname(__file__), '..', 'weights')
        self.weights_path = weights_path

        # Setup device
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        if self.use_gpu:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU")

        # Load original model
        self.model = MNISTNet().to(self.device)
        self._load_weights()

        # Initialize TEE if available
        self.tee_session = None
        self.loro_model = None
        self.obfuscated_weights = None

        if use_tee:
            self._init_tee()
        else:
            self._init_simulation()

    def _load_weights(self):
        """Load trained weights."""
        model_path = os.path.join(self.weights_path, 'mnist_model.pt')
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded weights from {model_path} to {self.device}")
        else:
            print(f"Warning: Weights not found at {model_path}")
            print("Run 'python train.py' first to train the model")

    def _init_tee(self):
        """Initialize real TEE interface."""
        try:
            from tee_inference import SecureInferenceSession
            self.tee_session = SecureInferenceSession()
            print("TEE interface initialized successfully")
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize TEE: {e}\n"
                f"Make sure the TA is installed:\n"
                f"  1. Build: ./compile.sh\n"
                f"  2. Install: sudo ./scripts/install_ta.sh\n"
                f"Or run without --tee for simulation mode."
            )

    def _init_simulation(self):
        """Initialize simulation mode (keys stored in Python)."""
        # Generate random B, A matrices for simulation
        # Note: PyTorch Linear weights are (output_dim, input_dim)
        torch.manual_seed(42)
        self.sim_keys = []
        for input_dim, output_dim, rank in LAYER_CONFIGS:
            # B: (output_dim, r), A: (r, input_dim)
            B = torch.randn(output_dim, rank, device=self.device) * 0.1
            A = torch.randn(rank, input_dim, device=self.device) * 0.1
            self.sim_keys.append((B, A))

        # Create obfuscated weights (on GPU if available)
        weights = self.model.get_layer_weights()
        self.obfuscated_weights = []
        for i, (W, b) in enumerate(weights):
            B, A = self.sim_keys[i]
            W_prime = W + B @ A
            self.obfuscated_weights.append((W_prime, b))

        print(f"Simulation mode initialized with random keys on {self.device}")

    def provisioning_phase(self):
        """
        Perform provisioning: generate keys, export for obfuscation, lock.
        This should be done ONCE during deployment.
        """
        print("\n" + "="*60)
        print("PROVISIONING PHASE")
        print("="*60)

        if self.use_tee and self.tee_session:
            loaded, locked = self.tee_session.tee.get_key_status()
            print(f"Current status: loaded={loaded}, locked={locked}")

            if locked:
                print("Keys already locked - provisioning complete")
                return

            if not loaded:
                print("\n1. Generating keys in TEE...")
                self.tee_session.setup_keys()

            print("\n2. Exporting keys for weight obfuscation...")
            keys = self.tee_session.get_deobfuscation_keys()

            print("\n3. Obfuscating weights: W' = W + BA")
            weights = self.model.get_layer_weights()
            self.obfuscated_weights = []
            for i, (W, b) in enumerate(weights):
                B, A = keys[i]
                # Convert numpy to torch, shapes: B(output_dim, r), A(r, input_dim)
                B_t = torch.from_numpy(B).to(self.device)
                A_t = torch.from_numpy(A).to(self.device)
                W_prime = W + B_t @ A_t
                self.obfuscated_weights.append((W_prime.clone(), b.clone()))
                print(f"   Layer {i+1}: W{list(W.shape)} -> W' obfuscated on {self.device}")

            print("\n4. Locking keys in TEE...")
            self.tee_session.lock_keys()

            print("\n✓ Provisioning complete!")
            print("  - Obfuscated weights W' are ready for REE deployment")
            print("  - Keys B, A are locked in TEE (cannot be exported)")

        else:
            # Simulation mode - keys are already generated
            print("Simulation mode: Keys already generated")
            weights = self.model.get_layer_weights()
            self.obfuscated_weights = []
            for i, (W, b) in enumerate(weights):
                B, A = self.sim_keys[i]
                W_prime = W + B @ A
                self.obfuscated_weights.append((W_prime.clone(), b.clone()))
                print(f"   Layer {i+1}: W{list(W.shape)} -> W' obfuscated")

    def ree_inference(self, x, layer_idx):
        """
        Perform inference in REE with obfuscated weights.
        This simulates the untrusted environment.

        Args:
            x: Input tensor (batch_size, input_dim)
            layer_idx: Layer index

        Returns:
            y_prime: Obfuscated output (batch_size, output_dim)
        """
        W_prime, b = self.obfuscated_weights[layer_idx]
        # y' = x @ W'.T + b (PyTorch Linear format)
        y_prime = x @ W_prime.T
        if b is not None:
            y_prime = y_prime + b
        return y_prime

    def tee_correction(self, x, layer_idx):
        """
        Compute correction in TEE using stored keys.
        B and A are NEVER exposed to REE!

        Args:
            x: Input tensor (batch_size, input_dim) - can be on GPU or CPU
            layer_idx: Layer index

        Returns:
            correction: Tensor to subtract from y' (batch_size, output_dim)
                        Returns on same device as input
        """
        if self.use_tee and self.tee_session:
            import numpy as np
            # TEE requires CPU memory - move if needed
            input_device = x.device
            x_cpu = x.cpu() if x.device.type == 'cuda' else x
            x_np = x_cpu.numpy()
            correction_np, _ = self.tee_session.tee.inference(layer_idx, x_np)
            correction = torch.from_numpy(correction_np)
            # Move back to original device
            return correction.to(input_device)
        else:
            # Simulation: correction = x @ A.T @ B.T (on GPU if available)
            B, A = self.sim_keys[layer_idx]
            correction = x @ A.T @ B.T
            return correction

    def secure_forward(self, x):
        """
        Complete secure forward pass (sequential - for compatibility).

        Args:
            x: Input tensor (batch_size, 784)

        Returns:
            output: Final logits (batch_size, 10)
            timing: Timing breakdown
        """
        timing = {}
        current = x

        for i in range(3):
            # REE: Compute with obfuscated weights
            start = time.time()
            y_prime = self.ree_inference(current, i)
            ree_time = time.time() - start

            # TEE: Compute correction
            start = time.time()
            correction = self.tee_correction(current, i)
            tee_time = time.time() - start

            # Deobfuscate
            y = y_prime - correction

            # ReLU for layers 1 and 2
            if i < 2:
                y = torch.relu(y)

            current = y
            timing[f'layer_{i+1}_ree'] = ree_time
            timing[f'layer_{i+1}_tee'] = tee_time

        return current, timing

    def secure_forward_parallel(self, x):
        """
        Parallel secure forward pass - REE and TEE run simultaneously.

        This is the OPTIMIZED version that exploits parallelism:
        - REE computes y' = x @ W'.T in one thread
        - TEE computes correction = x @ A.T @ B.T in another thread
        - Effective time = max(REE_time, TEE_time) instead of REE_time + TEE_time

        Args:
            x: Input tensor (batch_size, 784)

        Returns:
            output: Final logits (batch_size, 10)
            timing: Timing breakdown
        """
        timing = {}
        current = x

        for i in range(3):
            # Shared results
            results = {'y_prime': None, 'correction': None}
            errors = {'ree': None, 'tee': None}
            ree_time = [0]
            tee_time = [0]

            def ree_worker():
                """REE computation thread"""
                start = time.time()
                try:
                    results['y_prime'] = self.ree_inference(current, i)
                except Exception as e:
                    errors['ree'] = e
                ree_time[0] = time.time() - start

            def tee_worker():
                """TEE computation thread"""
                start = time.time()
                try:
                    results['correction'] = self.tee_correction(current, i)
                except Exception as e:
                    errors['tee'] = e
                tee_time[0] = time.time() - start

            # Start both threads simultaneously
            t_ree = threading.Thread(target=ree_worker)
            t_tee = threading.Thread(target=tee_worker)

            start_total = time.time()
            t_ree.start()
            t_tee.start()

            # Wait for both to complete
            t_ree.join()
            t_tee.join()
            total_time = time.time() - start_total

            # Check for errors
            if errors['ree']:
                raise errors['ree']
            if errors['tee']:
                raise errors['tee']

            # Deobfuscate
            y = results['y_prime'] - results['correction']

            # ReLU for layers 1 and 2
            if i < 2:
                y = torch.relu(y)

            current = y
            timing[f'layer_{i+1}_ree'] = ree_time[0]
            timing[f'layer_{i+1}_tee'] = tee_time[0]
            timing[f'layer_{i+1}_parallel'] = total_time

        return current, timing

    def baseline_forward(self, x):
        """Standard inference without obfuscation."""
        # Ensure input is on correct device
        if x.device != self.device:
            x = x.to(self.device)
        return self.model(x)

    def verify_correctness(self, x):
        """Verify secure inference matches baseline."""
        with torch.no_grad():
            baseline = self.baseline_forward(x)
            secure, _ = self.secure_forward(x)
            diff = torch.max(torch.abs(baseline - secure)).item()
        return diff

    def save_verification_results(self, output_dir=None, num_samples=10, use_parallel=False):
        """
        Save verification results with real MNIST images.

        Args:
            output_dir: Directory to save results (default: ../results/)
            num_samples: Number of test samples to visualize
            use_parallel: Use parallel execution mode
        """
        from torchvision import datasets, transforms

        # Default output directory relative to this file
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Data directory relative to this file
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

        # Load MNIST test dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

        # Select truly random samples (using current time as seed)
        np.random.seed(int(time.time() * 1000) % 2**31)
        indices = np.random.choice(len(test_dataset), num_samples, replace=False)

        # Prepare figure
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4.5))

        results = []
        correct_baseline = 0
        correct_secure = 0

        for i, idx in enumerate(indices):
            img, label = test_dataset[idx]
            x = img.view(1, 784).to(self.device)  # Move to GPU if available

            # Baseline inference
            with torch.no_grad():
                baseline_output = self.baseline_forward(x)
                baseline_probs = torch.softmax(baseline_output, dim=1)
                baseline_pred = baseline_output.argmax(dim=1).item()
                baseline_conf = baseline_probs[0, baseline_pred].item()

            # Secure inference
            with torch.no_grad():
                if use_parallel:
                    secure_output, _ = self.secure_forward_parallel(x)
                else:
                    secure_output, _ = self.secure_forward(x)
                secure_probs = torch.softmax(secure_output, dim=1)
                secure_pred = secure_output.argmax(dim=1).item()
                secure_conf = secure_probs[0, secure_pred].item()

            # Check correctness
            if baseline_pred == label:
                correct_baseline += 1
            if secure_pred == label:
                correct_secure += 1

            # Calculate difference
            diff = torch.max(torch.abs(baseline_output - secure_output)).item()

            results.append({
                'index': idx,
                'label': label,
                'baseline_pred': baseline_pred,
                'baseline_conf': baseline_conf,
                'secure_pred': secure_pred,
                'secure_conf': secure_conf,
                'max_diff': diff,
                'baseline_correct': baseline_pred == label,
                'secure_correct': secure_pred == label
            })

            # Plot original image
            img_np = img.squeeze().numpy()
            axes[0, i].imshow(img_np, cmap='gray')
            axes[0, i].set_title(f'True: {label}', fontsize=10)
            axes[0, i].axis('off')

            # Plot prediction comparison with confidence
            color_b = 'green' if baseline_pred == label else 'red'
            color_s = 'green' if secure_pred == label else 'red'

            axes[1, i].text(0.5, 0.75, f'Base: {baseline_pred} ({baseline_conf:.2f})',
                           ha='center', fontsize=8, color=color_b, fontweight='bold')
            axes[1, i].text(0.5, 0.25, f'Sec: {secure_pred} ({secure_conf:.2f})',
                           ha='center', fontsize=8, color=color_s, fontweight='bold')
            axes[1, i].set_xlim(0, 1)
            axes[1, i].set_ylim(0, 1)
            axes[1, i].axis('off')

        mode_str = "Parallel" if use_parallel else "Sequential"
        plt.suptitle(f'MNIST Secure Inference Verification (v2 {mode_str}+GPU)\n'
                    f'Device: {self.device} | '
                    f'Baseline: {correct_baseline}/{num_samples} | '
                    f'Secure: {correct_secure}/{num_samples}',
                    fontsize=11)
        plt.tight_layout()

        # Save figure
        fig_path = os.path.join(output_dir, 'mnist_verification.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved verification image: {fig_path}")

        # Save text results
        txt_path = os.path.join(output_dir, 'verification_results.txt')
        with open(txt_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("MNIST Secure Inference Verification Results\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Mode: {'TEE' if self.use_tee else 'Simulation'}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Number of samples: {num_samples}\n\n")

            f.write("Per-sample results:\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Idx':>5} {'Label':>5} {'Base':>5} {'Conf':>6} {'Secure':>6} {'Conf':>6} {'Diff':>10}\n")
            f.write("-" * 70 + "\n")
            for r in results:
                f.write(f"{r['index']:5d} {r['label']:5d} "
                       f"{r['baseline_pred']:5d} {r['baseline_conf']:6.3f} "
                       f"{r['secure_pred']:6d} {r['secure_conf']:6.3f} "
                       f"{r['max_diff']:10.6e}\n")
            f.write("-" * 70 + "\n\n")

            f.write(f"Baseline accuracy: {correct_baseline}/{num_samples} "
                   f"({100*correct_baseline/num_samples:.1f}%)\n")
            f.write(f"Secure accuracy:   {correct_secure}/{num_samples} "
                   f"({100*correct_secure/num_samples:.1f}%)\n\n")

            # Calculate overall difference
            max_diff = max(r['max_diff'] for r in results)
            avg_diff = sum(r['max_diff'] for r in results) / len(results)
            f.write(f"Max difference:  {max_diff:.6e}\n")
            f.write(f"Avg difference:  {avg_diff:.6e}\n")

            if max_diff < 1e-4:
                f.write("\n✓ Secure inference produces correct results!\n")
            else:
                f.write(f"\n⚠ Difference detected (threshold: 1e-4)\n")

        print(f"  Saved verification text: {txt_path}")

        return results

    def run_demo(self, batch_size=10, use_parallel=False):
        """Run the complete demo."""
        mode_str = "Parallel" if use_parallel else "Sequential"
        print("\n" + "="*60)
        print(f"SECURE INFERENCE DEMO v2 ({mode_str} + GPU Optimized)")
        print("="*60)

        print(f"\nConfiguration:")
        print(f"  Mode: {'TEE' if self.use_tee else 'Simulation'}")
        print(f"  Device: {self.device}")
        print(f"  Execution: {mode_str}")
        print(f"  Model: 3-layer MLP (784->256->128->10)")
        print(f"  Batch size: {batch_size}")

        # Run provisioning if needed
        if self.obfuscated_weights is None:
            self.provisioning_phase()

        # Generate test input on the appropriate device
        torch.manual_seed(123)
        x = torch.randn(batch_size, 784, device=self.device)

        # Verify correctness (using sequential for verification)
        print("\nVerifying correctness...")
        with torch.no_grad():
            baseline = self.baseline_forward(x)
            if use_parallel:
                secure, _ = self.secure_forward_parallel(x)
            else:
                secure, _ = self.secure_forward(x)
            diff = torch.max(torch.abs(baseline - secure)).item()

        print(f"  Max diff: {diff:.2e}")
        if diff < 1e-3:
            print("  ✓ Results are correct!")
        else:
            print("  ⚠ Significant difference detected")

        # Benchmark - Warmup (with GPU sync)
        print("\nBenchmarking...")
        for _ in range(10):
            _ = self.baseline_forward(x)
            if use_parallel:
                _, _ = self.secure_forward_parallel(x)
            else:
                _, _ = self.secure_forward(x)
        if self.use_gpu:
            torch.cuda.synchronize()

        # Time baseline (with GPU sync for accurate timing)
        if self.use_gpu:
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            _ = self.baseline_forward(x)
        if self.use_gpu:
            torch.cuda.synchronize()
        baseline_time = (time.time() - start) / 100

        # Time secure inference
        if self.use_gpu:
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            if use_parallel:
                _, timing = self.secure_forward_parallel(x)
            else:
                _, timing = self.secure_forward(x)
        if self.use_gpu:
            torch.cuda.synchronize()
        secure_time = (time.time() - start) / 100

        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)

        print(f"\nOverall Timing (batch_size={batch_size}):")
        print(f"  Baseline:  {baseline_time*1000:.3f} ms")
        print(f"  Secure:    {secure_time*1000:.3f} ms")

        print(f"\nOverhead vs Baseline: {(secure_time/baseline_time - 1)*100:.1f}%")

        print(f"\nPer-layer breakdown ({mode_str}):")
        for i in range(3):
            ree = timing[f'layer_{i+1}_ree'] * 1000
            tee = timing[f'layer_{i+1}_tee'] * 1000
            if use_parallel:
                par = timing[f'layer_{i+1}_parallel'] * 1000
                print(f"  Layer {i+1}: REE={ree:.3f}ms, TEE={tee:.3f}ms, Total={par:.3f}ms")
            else:
                print(f"  Layer {i+1}: REE={ree:.3f}ms, TEE={tee:.3f}ms")

        total_ree = sum(timing[f'layer_{i+1}_ree'] for i in range(3))
        total_tee = sum(timing[f'layer_{i+1}_tee'] for i in range(3))
        print(f"\n  Total REE: {total_ree*1000:.3f} ms")
        print(f"  Total TEE: {total_tee*1000:.3f} ms")

        # Save verification results with real MNIST images
        print("\n" + "="*60)
        print("Saving verification results...")
        print("="*60)
        self.save_verification_results(num_samples=16, use_parallel=use_parallel)


def main():
    parser = argparse.ArgumentParser(description='LoRO Secure Inference Demo')
    parser.add_argument('--weights', type=str, default='../weights/')
    parser.add_argument('--tee', action='store_true', help='Use real TEE')
    parser.add_argument('--gpu', action='store_true', default=True,
                        help='Use GPU for REE inference (default: True)')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU')
    parser.add_argument('--parallel', action='store_true',
                        help='Use parallel REE/TEE execution (default: sequential)')
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--provision', action='store_true',
                        help='Run provisioning phase only')
    parser.add_argument('--output', type=str, default='results/',
                        help='Output directory for verification results')
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of MNIST samples to verify')

    args = parser.parse_args()

    use_gpu = args.gpu and not args.no_gpu

    demo = SecureMNISTDemo(weights_path=args.weights, use_tee=args.tee, use_gpu=use_gpu)

    if args.provision:
        demo.provisioning_phase()
    else:
        demo.run_demo(batch_size=args.batch_size, use_parallel=args.parallel)


if __name__ == '__main__':
    main()
