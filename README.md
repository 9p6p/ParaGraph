# ParaGraph: Accelerating Graph Indexing through GPU-CPU Parallel Processing for Efficient Cross-modal ANNS

This repository includes the codes for the SIGMOD's Workshop DaMoN 2025 paper ParaGraph.

![](https://api.visitorbadge.io/api/VisitorHit?user=9p6p&repo=ParaGraph&countColor=%237B1E7A)

[![GitHub Clones](https://img.shields.io/badge/dynamic/json?color=success&label=Clone&query=count&url=https://gist.githubusercontent.com/9p6p/cf0fb22e0e7c80f5fed949e53d29eaca/raw/clone.json&logo=github)]((https://github.com/MShawon/github-clone-count-badge))

The master branch is the codebase of the ParaGraph paper.

## Getting Started & Reproduce Experiments in the Paper
File format: all `~bin` files begin with the number of vectors (uint32, 4 bytes), dimension (uint32, 4 bytes), and followed by the vector data. (Same format as big-ann competition.)

You can obtain the required datasets from the RoarGraph repository. We utilize Python scripts to perform the necessary data transformations. For the index construction process, we exclusively use the base vector set (base) and the corresponding ground truth data (gt).

The base vector data (base_data) is structured as an num x dim matrix, where num signifies the total number of vectors, and dim denotes the dimensionality of each vector. These two parameters, num and dim, must be pre-defined within the source code.

Furthermore, to ensure efficient GPU memory management, the handling of the ground truth (gt) data is modified. Specifically, the number of ground truth neighbors recorded for each query vector (often denoted as gt_num or top_k) is limited or adjusted to 128.

## 0. Prerequisite
```
cmake >= 3.24
g++ >= 9.4
CPU supports AVX-512

Python >= 3.8
Python package:
numpy
urllib
tarfile

NVIDIA GPU
CUDA Toolkit
cuDNN
```

```
sudo apt install libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev libmkl-full-dev
```

You can refer to additional resources to configure the GPU environment.

```bash
git clone https://github.com/9p6p/ParaGraph.git
```

## 1. Compile and build
```bash
mkdir -p build
cd build
cmake .. && make -j
```

## 2. Bulild Index
```bash
bash run_paragraph.sh
```

## License
MIT License

## Contact
For questions or inquiries, feel free to reach out to me at
[yangyx2023@mail.sustech.edu.cn](mailto:yangyx2023@mail.sustech.edu.cn)