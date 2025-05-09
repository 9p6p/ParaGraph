# ParaGraph: Accelerating Graph Indexing through GPU-CPU Parallel Processing for Efficient Cross-modal ANNS

This repository includes the codes for the SIGMOD's Workshop DaMoN 2025 paper ParaGraph.

![](https://api.visitorbadge.io/api/VisitorHit?user=9p6p&repo=ParaGraph&countColor=%237B1E7A)

[![GitHub Clones](https://img.shields.io/badge/dynamic/json?color=success&label=Clone&query=count&url=https://gist.githubusercontent.com/9p6p/e0630d25817c7ab360f7e85328c9559c/raw/clone.json&logo=github)]((https://github.com/MShawon/github-clone-count-badge))

The `master` branch contains the codebase for the ParaGraph paper.

## Getting Started & Reproducing Experiments

This section guides you through setting up the project and reproducing the experiments presented in our paper.

### File Format

All `~bin` data files adhere to the following structure (consistent with the Big-ANN competition format):

1.  **Number of vectors:** `uint32` (4 bytes)
2.  **Dimension of vectors:** `uint32` (4 bytes)
3.  **Vector data:** Sequentially listed vector components.

### Data Acquisition and Preparation

* You can obtain the required datasets from the [RoarGraph repository](https://github.com/matchyc/RoarGraph).
* We utilize Python scripts for the necessary data transformations.
* For the index construction process, we exclusively use the base vector set (`base`) and the corresponding ground truth data (`gt`).

The **base vector data** (`base_data`) is structured as an `num` x `dim` matrix, where:
* `num`: Signifies the total number of vectors.
* `dim`: Denotes the dimensionality of each vector.
> **Note:** These two parameters, `num` and `dim`, must be pre-defined within the source code.

To ensure efficient GPU memory management, the handling of the **ground truth (`gt`) data** is modified. Specifically, the number of ground truth neighbors recorded for each query vector (often denoted as `gt_num` or `top_k`) is limited or adjusted to 128.

## 0. Prerequisites

### Software & System Requirements:
* CMake `v3.24` or newer
* g++ `v9.4` or newer
* CPU with AVX-512 support

### Python Environment:
* Python `v3.8` or newer
* Required Python packages:
    * `numpy`

### GPU Requirements:
* NVIDIA GPU
* CUDA Toolkit
* cuDNN

### System Dependencies:
Install the following system libraries:
```bash
sudo apt update
sudo apt install -y libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev libmkl-full-dev
```
You can refer to additional resources to configure the GPU environment.

### Clone the Repository:
```bash
git clone https://github.com/9p6p/ParaGraph.git
cd ParaGraph
```

## 1. Compile and build
Follow these steps to compile the project:
```bash
mkdir -p build
cd build
cmake .. && make -j
```

## 2. Bulild Index
To build the ParaGraph index, run the provided script:
```bash
bash run_paragraph.sh
```

## License
This project is licensed under the MIT License.

## Contact
For questions or inquiries, feel free to reach out to me at
[yangyx2023@mail.sustech.edu.cn](mailto:yangyx2023@mail.sustech.edu.cn)
