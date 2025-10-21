# Code for "LoRO: Real-Time on-Device Secure Inference for LLMs via TEE-Based Low Rank Obfuscation"

The code for implementing the LoRO: Real-Time on-Device Secure Inference for LLMs via TEE-Based Low Rank Obfuscation.

## Requirements
Since our experiments is conducted on three different platforms: **Workstation, Nvidia Orin NX Board with TrustZone, and Laptop with SGX**, we provide the detailed requirements for each platform.

### Python Requirement
Our code in the ```accuracy``` and ```attack``` folder can be run on any platform with GPUs with the following requirements:
- Python >= 3.8
- PyTorch >= 2.0.0
- Cuda Environment
- Ctype
- Numpy
- Newest transformers, datasets, peft from HuggingFace
- Tqdm and other common packages

Actually, no specific version are required, and it's ok to run directly if you have used pytorch and transformers before.

### Arm TrustZone OP-TEE Environment
OP-TEE is a TEE-OS implementation based on ARM TrustZone technology. The code in the ```latency/trustzone``` folder is designed to run on devices installed with OP-TEE.

Please follow the [official instruction](https://optee.readthedocs.io/en/latest/) to install OP-TEE on your device, and test it with [examples](https://github.com/linaro-swg/optee_examples).

It is suggested to install OP-TEE>=1.0 to achieve acceleration with ARM NEON. For NVIDIA Jetson devices, this is already installed in newer L4T systems.

Notably, if you want to reproduce our results in large LLMs, such as LLaMA 3-8B, you need to compile OP-TEE by yourself and set the secure memory to 128MB at least.

### Intel SGX Gramine Environment
Gramine-sgx is a Lib OS designed for Intel SGX technology. The code in the ```latency/sgx``` folder is designed to run on devices installed with Intel SGX. A device equipped with Intel SGX is required as the hardware to run the code. It is recommended to test on Linux since we have not tested Gramine in Windows.

The following steps are necessary to build a Gramine environment.

1. Linux-SGX Driver. SGX-Driver is required to be installed, which is the fundemental environment. Please refer to [Linux-SGX Respository](https://github.com/intel/linux-sgx) to build from source-code. For some versions of CPUs and systems, SGX may already be integrated in the system driver.

2. Gramine-SGX. Please follow the [Gramine Respository](https://github.com/gramineproject/gramine) to install the Gramine.

3. Test. You can test your Gramine according to this simple [Pytorch Demo](https://github.com/gramineproject/examples/tree/master/pytorch).

**We strongly recommend to pass the example test before running our code.**

## Code Organization
Our code is organized as follows:

```
.
├── accuracy # code for accuracy evaluation
│   ├── [model_name]
│   │   ├── [dataset]
│   │   │   ├── ipynb # ipynb with our records
│   │   │   │   ├── original.ipynb # evaluate the original accuracy
│   │   │   │   ├── loro.ipynb # evaluate the accuracy of obfuscated model and de-obfuscate the results
│   │   │   │   ├── loro_ree.ipynb # evaluate the accuracy of obfuscated model but do not de-obfuscate the results
│   │   │   ├── py # runnable py code, the result is same as ipynb
├── latency # code for latency evaluation
│   ├── sgx
│   │   ├── our_c_lib # must compile the c lib firstly.
│   │   ├── [model_name]
│   │   │   ├── results # folder to save the results
│   │   │   ├── scripts # our code
│   │   │   ├── Makefile 
│   │   │   ├── python.manifest.template # manifest file for Gramine, you need to modify it according to your environment
│   │   │   ├── run.sh # script to run the experiment
│   ├── trustzone
│   │   ├── our_c_lib # must compile the c lib firstly.
│   │   ├── [model_name]
│   │   │   ├── result.txt # save the results
│   │   │   ├── run.sh # script to run the experiment
│   │   │   ├── [*].py # our code 
├── attack # demo for attacking TLG
├── LoRO # folder save our LoRO code
└── README.md
```

## Usage

All python code and shell should be run in the path where it exists!

### Accuracy Evaluation
Run the ipynb file in the ```accuracy/[model_name]/[dataset]/ipynb``` folder to evaluate the accuracy of the original model and the obfuscated model. 

Our code will download the model and datasets from huggingface automatically. Our results are saved in the ipynb file for your review.

You can run the python code in the ```accuracy/[model_name]/[dataset]/py``` folder to achieve the same results. One example is:
```
cd ./accuracy/qwen/gsm8k/py
python loro.py
python original.py
python loro_ree.py
```

### Trustzone Latency Evaluation
1. Clone the repository and install the requirements.
   
2. Compile the c lib in the ```latency/trustzone/our_c_lib``` folder.
   
Firstly, modify the Makefile in ```latency/trustzone/our_c_lib/trust_application``` to match your environment, especially **TA_DEV_KIT_DIR** and **TEEC_EXPORT** path. The other thing should not be changed if your OP-TEE is compiled by default.
Secondly, compile the c lib by:
```
cd ./latency/trustzone/our_c_lib/data_transfer
./compile.sh
cd ./latency/trustzone/our_c_lib/trust_application
./compile.sh
```
Notably, sudo is required to run the script in ```trust_application```, since it compiles trusted function with OP-TEE library. And CUDA Compiler (nvcc) is required to run the script in ```data_transfer```.

3. Run the experiment. Take roberta as an example:
```
cd ./latency/trustzone/roberta
./run.sh
```
Notably, sudo is required to run the ```run.sh```, since it calls secure world in TrustZone. 

4. The results are saved in the ```result.txt``` file in the ```latency/trustzone/[model_name]``` folder, also in terminal.
   
### SGX Latency Evaluation
1. Clone the repository and install the requirements.
2. Compile the c lib in the ```latency/[platform]/our_c_lib``` folder, by:
```
cd ./latency/sgx/our_c_lib/data_transfer
./compile.sh
```
CUDA Compiler (nvcc) is required to run the script in ```data_transfer```.

3. Run the experiment. 

Firstly, modify the setting file ```python.manifest.template```, **all the path should be on your machine**, rather than mine. 

Secondly, run the experiment, take roberta as an example by:
```
cd ./latency/sgx/roberta
./run.sh
```

4. The results are saved in the ```result.txt``` file in the ```latency/sgx/[model_name]``` folder, also in terminal.

## Attack
We present the de-obfuscation stage of our Model Stealing attack with Prior against TLG, in ```demo.ipynb``` file under ```attack```. The knockoff code can be found in [here](https://github.com/tribhuvanesh/knockoffnets).

### Acknowledgements
The project is released under MIT License.