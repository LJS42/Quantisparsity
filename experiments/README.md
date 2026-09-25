# CIFAR-10 ResNet20: INT4 quantization with sparsity

The default experiment optimizes output channel 0 of the final convolution,
`layer3.2.conv2`: **576 weight variables**, 576 indicator variables, signed INT4
values in `[-8, 7]`, and at most 288 nonzeros. Use `--rows 64` to optimize all
output channels of that convolution. This does not quantize the entire network.

PyTorch loads the model, extracts activations, and evaluates accuracy.
Optimization runs in the C++/CUDA GPUBoscia solver. Preparation, solving, and
evaluation can run on different machines; the CUDA machine does not need PyTorch.

## Build the CUDA executable

First install GPUBoscia as described in the [project README](../README.md).
From this experiment repository's root, replace `80` with your GPU's CUDA architecture:

```bash
cmake -S . -B build-resnet20 -DCMAKE_PREFIX_PATH="$HOME/.local/gpuboscia" -DCMAKE_CUDA_ARCHITECTURES=80 -DQUANTIZATION_SPARSE_WEIGHTS=576
cmake --build build-resnet20 --target quantization_sparse -j 4
```

Requirements are the project's CUDA/cuBLAS, CMake, C++ compiler, Threads, and
installed GPUBoscia package. ZLIB is not required by this experiment. The separate build directory preserves other builds with
different compiled row dimensions.

## Run across machines

Preparation and evaluation require compatible installations of `torch`,
`torchvision`, and `numpy`. CPU PyTorch is sufficient: activation extraction
and evaluation run on the CPU. The first online run downloads
[model code and pretrained weights](https://github.com/chenyaofo/pytorch-cifar-models)
and CIFAR-10. Preprocessing follows the checkpoint author's
[training configuration](https://github.com/chenyaofo/image-classification-codebase/blob/master/conf/cifar10.conf).

1. Export data on the machine with PyTorch:

   ```bash
   python3 experiments/resnet20_quantization.py --stage prepare --rows 1 --output experiments/results/resnet20
   ```

2. Copy the entire exported directory to the CUDA machine. From the repository
   root, run the solver. This step requires only the Python standard library and
   the compiled CUDA executable; it imports neither PyTorch nor NumPy.

   ```bash
   python3 experiments/solve_exported_rows.py --output experiments/results/resnet20 --solver build-resnet20/quantization_sparse --time-limit 60
   ```

   Add `--resume` after an interruption. Completed rows are reused only when
   input and solver checksums, sparsity budget, seed, sample count, and time limit
   match. Existing results are not overwritten by default. If a short time limit
   produces no feasible incumbent, the runner keeps the log and reports failure;
   it does not substitute the baseline. Use a new experiment directory when
   increasing the time limit.

3. Copy the results directory back to the machine with PyTorch and evaluate the
   full CIFAR-10 test set:

   ```bash
   python3 experiments/resnet20_quantization.py --stage evaluate --output experiments/results/resnet20
   ```

If all dependencies are on one machine, run all stages with:

```bash
python3 experiments/resnet20_quantization.py --solver build-resnet20/quantization_sparse --rows 1 --time-limit 60
```

To process the entire selected convolution, change `--rows 1` to `--rows 64`
and choose a new `--output` directory. The 60-second limit applies per row to
the solver; downloads, GPU initialization, and evaluation take additional time.

A single exported row can also be solved without Python:

```bash
build-resnet20/quantization_sparse --input experiments/results/resnet20/row_00.txt --samples 2048 --k 288 --time-limit 60 --output /tmp/row_00.solution
```

Direct invocation produces only a row solution. Use the Python runner for
resume support and the metadata required by the evaluation stage.

## Experiment definition

- **Calibration:** randomly select 128 CIFAR-10 training images, then uniformly
  sample 2048 convolution input patches without replacement. Only selected
  patches are retained instead of accumulating all unfolded activations.
- **Evaluation:** use all 10,000 test images, independently of calibration and
  scale selection.
- **Fixed per-row scale:** `max(max(w)/7, -min(w)/8, 1e-12)`.
- **Baseline:** retain the k largest original weight magnitudes, then round and
  clip to the same INT4 grid.
- **Objective:** `||(w - scale*z) X||²`. Export `w/scale` and `scale*X` so that
  the existing solver computes the physical reconstruction loss. Reported
  losses are sums of squared errors, not means.
- **Initialization:** the baseline is evaluated separately, not used as a solver
  warm start. A time-limited solver can return a worse result; this is reported
  without substitution.
- **Model changes:** no BatchNorm folding, fine-tuning, or BatchNorm recalibration.
  Models remain in evaluation mode. Only the selected convolution's first N
  rows change; all other parameters are preserved.

A one-row accuracy change is not evidence of whole-network compression quality.
The next experiment should cover the full selected convolution before extending
to other layers.

The export preserves the original checkpoint, calibration arrays, selected
image/patch indices, row inputs, and scales. Checkpoint and calibration files
have SHA-256 checksums. Evaluation reloads the saved checkpoint, rather than
replacing it with newly downloaded pretrained weights.

## Outputs

| File | Contents |
|---|---|
| `manifest.json` | Experiment settings, data checksums, and baseline |
| `row_XX.txt` | CUDA solver input |
| `row_XX.log` | Solver log |
| `row_XX.solution`, `row_XX.run.json` | Validated solution and run identity |
| `solver_report.json` | Losses, bounds, runtime, node counts, nonzeros, and timeout flags |
| `report.json` | Evaluated results, including original/baseline/solver test accuracy |
| `original_state.pt`, `solved_state.pt` | Original and partially quantized model state dictionaries |

The solve stage independently recomputes `||(z-w)X||²` with Python's standard
library and checks the reported loss, INT4 range, integrality, nonzero budget,
and lower bound. Evaluation checks the physical loss again using the original
floating-point weights and activations. A negative-infinite lower bound is
stored as `null` in JSON.

## Offline use

Use `--model-dir /path/to/pytorch-cifar-models` for local model code. Pin the
repository version and use the same version for preparation and evaluation.
Use `--checkpoint /path/to/state.pt` to load a local model state dictionary,
not a training snapshot containing an optimizer and other fields.

Offline preparation requires `--offline`, `--model-dir`, `--checkpoint`, and
an existing CIFAR-10 training dataset under `--data`. Offline evaluation requires
`--offline`, `--model-dir`, and the local test dataset; model weights come from
the experiment directory. The solve stage never requires network access.

## File format and validation

Solver input is whitespace-delimited text: `Weights samples`, followed by
`Weights` normalized weights and `Weights*samples` row-major scaled activation
values. Dimensions must match the compiled weight count and `--samples`.
Output starts with `loss lower_bound seconds nodes time_limit_flag`, followed by
one line of integer weights.

Run CPU-only boundary tests with:

```bash
python3 -m unittest discover -s experiments -p 'test_*.py' -v
```

Tests cover non-unit scales, correlated-input reconstruction loss, file
round-trips, solution feasibility, incorrect objective values, corrupted inputs,
and resume behavior. They substitute a small fake process for the CUDA solver
and do not validate the CUDA algorithm.

The development environment has neither CUDA/GPU nor PyTorch. CUDA compilation,
checkpoint loading, and real-model accuracy/runtime measurements have not been
performed here; no end-to-end results are claimed.
