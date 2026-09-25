# Sparse INT4 quantization with GPUBoscia

A standalone CUDA experiment for joint weight quantization and sparsity.
It consumes the **installed GPUBoscia package** through CMake, without a source
checkout, submodule, or vendored copy of the solver. The default problem has
576 weights per row (excluding indicators), suitable for CIFAR-10 ResNet20.

## Make this a separate repository

Copy **this whole directory** into your new repository. For example, from the
GPUBoscia checkout, on your machine:

```bash
cp -a standalone/quantization_sparse ../quantization_sparse
cd ../quantization_sparse
git init -b main
git add .
git commit -m "Initialize standalone quantization experiment"
```

Then connect it to your chosen remote in the usual way. No remote URL is assumed
here. This directory contains no nested Git repository; its local placement under
`standalone/` is only for handoff. Moving it does not change its build commands.
Generated build files, downloaded data, Python caches, and results are ignored.

You will have two repositories:

```text
GPUBoscia/                 General solver library
quantization_sparse/      Objective, sparsity oracle, experiments, and their tests
```

## 1. Install GPUBoscia once

On the CUDA machine, configure the **GPUBoscia repository**, not this experiment:

```bash
cmake -S /path/to/GPUBoscia -B /path/to/GPUBoscia/build-library \
  -DGPUBOSCIA_BUILD_EXAMPLES=OFF \
  -DGPUBOSCIA_BUILD_TESTS=OFF \
  -DGPUBOSCIA_CUDA_ARCH=80
cmake --install /path/to/GPUBoscia/build-library --prefix "$HOME/.local/gpuboscia"
```

Replace `80` with the GPU architecture. GPUBoscia is header-only, so no library
compilation step is needed between configuration and installation. Use a
CUDA-enabled installation; the Core-only package cannot run this solver.

After installation, the experiment needs only the installed prefix. The
GPUBoscia source checkout does not have to remain on the experiment machine.
When changing the library, reinstall it and rebuild the experiment.

## 2. Build the experiment

From **this experiment repository**:

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$HOME/.local/gpuboscia" \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DQUANTIZATION_SPARSE_WEIGHTS=576
cmake --build build --target quantization_sparse -j 4
```

The CMake dependency is simply:

```cmake
find_package(GPUBoscia 0.1 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(quantization_sparse PRIVATE gpuboscia::gpuboscia CUDA::cublas)
```

The experiment requires CMake 3.20+, a C++17/CUDA toolchain, cuBLAS, and the
installed GPUBoscia package. It does not require ZLIB. Tests also require Python 3;
use `-DQUANTIZATION_SPARSE_BUILD_TESTS=OFF` for an executable-only build.
`QUANTIZATION_SPARSE_WEIGHTS` is an experiment option, not a GPUBoscia option.
The default is 576; set 4096 in a separate build directory to reproduce the
previous synthetic problem dimension.

## 3. Run

First check the executable on synthetic inputs, without PyTorch or downloads:

```bash
./build/quantization_sparse --k 288 --samples 2048 --time-limit 60
```

This checks the experiment's execution path, not model accuracy. For the real
ResNet20 experiment, install the optional Python dependencies and run:

```bash
python3 -m pip install -r requirements.txt
python3 experiments/resnet20_quantization.py \
  --solver build/quantization_sparse --rows 1 --time-limit 60
```

The first real-model run downloads model code, pretrained weights, and CIFAR-10.
It optimizes one 576-weight row, compares against magnitude pruning plus
rounding, and evaluates the modified model. Results appear in
`experiments/results/resnet20/report.json`. A one-row run is a smoke experiment,
not whole-network quantization.

Use `--rows 64` and a fresh `--output` directory to process the whole selected
convolution. Data preparation/evaluation and CUDA solving can run on different
machines; the CUDA runner requires only the Python standard library. See the
[detailed experiment guide](experiments/README.md) for that workflow, offline
operation, resuming runs, baselines, and file formats.

## Tests

Build all test executables and run the default test set on a CUDA machine:

```bash
cmake --build build -j 4
ctest --test-dir build --output-on-failure
```

This includes the Python data boundary tests, rounding/pruning regressions, and
four-weight end-to-end problems checked by exhaustive enumeration. CUDA tests
require an available GPU. The longer root-only diagnostic is explicitly opt-in:

```bash
./build/test_quantization --quantization-root-exact 1000
```

The diagnostic uses the configured row dimension and compares GPU components
with CPU calculations. Root-only completion does not certify integer optimality.

Without CUDA, configure only the Python boundary tests:

```bash
cmake -S . -B build-cpu -DQUANTIZATION_SPARSE_BUILD_SOLVER=OFF
ctest --test-dir build-cpu --output-on-failure
```

This mode requires neither GPUBoscia nor PyTorch, and does not build a substitute
CPU solver. The Python tests use a fake solver process.

## Layout

```text
CMakeLists.txt                       Installed-package dependency and targets
include/quantization_sparse/problem.cuh  Objective, LMO, input data, GPU buffers
src/main.cu                          Experiment command-line executable
experiments/                         ResNet20 scripts and Python boundary tests
tests/test_quantization.cu           Quantization regressions and root diagnostic
requirements.txt                     Optional model preparation/evaluation packages
```

The problem header and CLI are separate, so tests reuse the actual problem
implementation without including an executable source file.

## Validation status

The Python boundary tests and GPUBoscia's CPU/package tests were run locally.
The experiment was also copied outside the library checkout and its CPU-only
configuration and tests were checked there. The current environment has no CUDA
compiler, GPU, or PyTorch, so the independent CUDA build, GPU diagnostics, and
real-model measurements remain to be verified on your CUDA machine.
