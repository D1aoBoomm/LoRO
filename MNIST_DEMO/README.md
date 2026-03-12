# LoRO-MNIST-DEMO

A demonstration of **LoRO (Low-Rank Obfuscation)** for secure neural network inference using **OP-TEE Trusted Execution Environment (TEE)**.

## Key Features

- **Hardware-Backed Security**: Uses OP-TEE TEE for secure storage
- **GPU Acceleration**: REE inference runs on GPU for optimal performance
- **Zero Key Exposure**: B and A matrices NEVER leave TEE during inference
- **MNIST Demo**: Complete working example with 3-layer MLP

## Requirements

### Hardware
- ARM processor with TrustZone support
- NVIDIA Jetson Orin (tested) or similar TEE-enabled device

### Software
- OP-TEE OS (v3.x+)
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.x+ (for GPU acceleration and CUDA compile)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/D1aoBoomm/LoRO-MNIST-TEE.git
cd LoRO-MNIST-TEE
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
cd src
python train.py
```
This will train an MNIST classifier and save weights to `weights/mnist_model.pt`.

### 4. Build the TEE Application
Perhaps you need to modify the OPTEE lib path or something else in ```compile.sh``` and ```install_ta.sh``` according to your own setting.

```bash
./compile.sh
```

### 5. Install the TA (Trusted Application)
```bash
sudo ./scripts/install_ta.sh
```

## Usage

### Simulation Mode (No TEE Required)
For testing without TEE hardware:
```bash
cd src
python secure_inference_demo.py --batch-size 32
```

### TEE Mode (Real Secure Inference)
With TEE hardware:
```bash
cd src
sudo {your_python_path} secure_inference_demo.py --tee --batch-size 32
```

### Command Line Options
```
python secure_inference_demo.py [OPTIONS]

Options:
  --weights PATH    Path to model weights (default: ../weights/)
  --tee             Use real TEE for secure inference
  --no-gpu          Disable GPU acceleration
  --parallel        Use parallel REE/TEE execution (slower)
  --batch-size N    Batch size for inference (default: 10)
  --provision       Run provisioning phase only
  --samples N       Number of MNIST samples to verify (default: 10)
```

## Project Structure

```
MNIST_DEMO/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── compile.sh                  # Build script for TEE application
├── Makefile                    # Root Makefile
│
├── src/                        # Python source code
│   ├── model.py               # MNIST MLP model definition
│   ├── train.py               # Model training script
│   ├── loro.py                # LoRO implementation utilities
│   ├── tee_inference.py       # TEE Python bindings
│   └── secure_inference_demo.py  # Main demo application
│
├── ta/                         # Trusted Application (TEE)
│   ├── mnist_ta.c             # TA implementation
│   └── include/mnist_demo.h   # TA header with UUID
│
├── host/                       # Host Application (REE)
│   └── main.c                 # Host application with pinned memory
│
├── scripts/                    # Utility scripts
│   ├── install_ta.sh          # Install TA to system
│   └── run_demo.sh            # Run demo script
│
├── weights/                    # Trained model weights
├── data/                       # MNIST dataset
└── results/                    # Verification results
```

## Extending to Other Models

To apply LoRO to your own model:

1. **Modify Layer Configuration** in `src/secure_inference_demo.py`:
```python
LAYER_CONFIGS = [
    (input_dim, output_dim, rank),  # Your layer dimensions
    ...
]
```

2. **Update TA Configuration** in `ta/mnist_ta.c`:
```c
#define NUM_LAYERS your_num_layers
```

3. **Regenerate UUID** for your TA (use `uuidgen`)

## License

MIT License.
