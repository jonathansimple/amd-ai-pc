<!-- Getting started with RESNET18_BF16 flow test with Python and C++ deployment -->
<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><h1> Ryzen™ AI ResNet BF16 Tutorial </h1>
    </td>
 </tr>
</table>

## Introduction

This tutorial demonstrates the inference workflow using a pre trained ResNet18 model. The focus is to convert the model to bf16 precision using VAIML compiler and offloading it to an NPU using Python.

## Overview

This Tutorial will help with the steps to deploy RESNET18 model demonstrating:

- Download the RESNET18 model from Hugging Face and export to ONNX format.
- Quantize the model to BF16 using AMD Quark Quantizer.
- Compile and run the model on NPU using ONNX runtime with Vitis AI Execution provider using Python.

## Requirements

To build the example, the following software must be installed:

- Ryzen AI 1.5.0
- Cmake
- Visual Studio 2022

Please follow the steps given in Installation Instruction to install Rayzen AI 1.5.0 , NPU driver and Visual Studio 2022


### Step 1: Install Packages

Ensure that the Ryzen AI Software is correctly installed. For more details, see the installation instructions. The default RYZEN_AI_INSTALLATION_PATH is C:\Program Files\RyzenAI\1.5.0. Use the conda environment created during the installation for the rest of the steps. This example requires a couple of additional packages. Run the following command to install them:

```bash
conda create --name resnet_bf16 --clone ryzen-ai-1.5.0
conda activate resnet_bf16
set RYZEN_AI_INSTALLATION_PATH = <path/to/RyzenAI/installation>
python -m pip install -r requirements.txt
```

###  Step 2: Download Model and Dataset

The prepare_model_data.py script downloads the CIFAR-10 dataset in pickle format for python in data folder. This dataset will be used in the subsequent steps for BF16 compilation and inference. The script also exports the provided PyTorch model into ONNX format in models folder.

```bash
python prepare_model_data.py
```

### Step 3: Model Compilation

```bash
python compile.py --model models\resnet_trained_for_cifar10.onnx
```

Above script will use the config_file and cache_dir to compile model convert into BF16 precision for subsequent steps using VAIML(Vitis AI Model Compiler).

```python
            cache_dir = Path(__file__).parent.resolve()
            cache_dir = os.path.join(cache_dir,'my_cache_dir')
            cache_key   = pathlib.Path(onnx_model).stem
            provider_options_dict = {
                "config_file": config_file,
                "cache_dir":   cache_dir,
                "cache_key":   cache_key,
                "enable_cache_file_io_in_mem":0,
            }
```

### Expected output

The expected output after the model compilation

```bash
[Vitis AI EP] No. of Operators : VAIML   124
[Vitis AI EP] No. of Subgraphs : VAIML     1
Done
```

## Model Deployment


### Model Deployment on CPU

```bash
   python predict.py

   Expected output:
    [Vitis AI EP] No. of Operators :   CPU     5  VAIML   119
    [Vitis AI EP] No. of Subgraphs : VAIML     1
    Image 0: Actual Label cat, Predicted Label cat
    Image 1: Actual Label ship, Predicted Label ship
    Image 2: Actual Label ship, Predicted Label ship
    Image 3: Actual Label airplane, Predicted Label airplane
    Image 4: Actual Label frog, Predicted Label frog
    Image 5: Actual Label frog, Predicted Label frog
    Image 6: Actual Label automobile, Predicted Label truck
    Image 7: Actual Label frog, Predicted Label frog
    Image 8: Actual Label cat, Predicted Label cat
    Image 9: Actual Label automobile, Predicted Label automobile
```

### Model Deployment on NPU

```bash
    python predict.py --ep npu
    Expected output:

    execution started on NPU
    [Vitis AI EP] No. of Operators : VAIML   124
    [Vitis AI EP] No. of Subgraphs : VAIML     1
    Image 0: Actual Label cat, Predicted Label cat
    Image 1: Actual Label ship, Predicted Label ship
    Image 2: Actual Label ship, Predicted Label ship
    Image 3: Actual Label airplane, Predicted Label airplane
    Image 4: Actual Label frog, Predicted Label frog
    Image 5: Actual Label frog, Predicted Label frog
    Image 6: Actual Label automobile, Predicted Label truck
    Image 7: Actual Label frog, Predicted Label frog
    Image 8: Actual Label cat, Predicted Label cat
    Image 9: Actual Label automobile, Predicted Label automobile
```