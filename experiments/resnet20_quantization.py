"""Prepare/evaluate real CIFAR-10 ResNet20 rows; optimization stays in CUDA."""
import argparse
import copy
import json
from pathlib import Path

from solve_exported_rows import digest, read_solution, solve, write_json


MODEL_REPO = 'chenyaofo/pytorch-cifar-models'
LAYER = 'layer3.2.conv2'


def load_model(args, pretrained):
    import torch
    if args.model_dir:
        return torch.hub.load(str(args.model_dir.resolve()), 'cifar10_resnet20',
                              source='local', pretrained=pretrained).eval()
    return torch.hub.load(MODEL_REPO, 'cifar10_resnet20', pretrained=pretrained,
                          trust_repo=True).eval()


def dataset(args, train):
    from torchvision import datasets, transforms
    # Matches the checkpoint author's CIFAR-10 training configuration.
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    return datasets.CIFAR10(str(args.data), train=train, download=not args.offline,
                            transform=transform)


def prepare(args):
    import numpy as np
    import torch
    import torch.nn.functional as F

    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('export directory is not empty; choose a new --output to preserve existing runs')
    model = load_model(args, pretrained=args.checkpoint is None)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True), strict=True)
    train = dataset(args, train=True)
    layer = model.get_submodule(LAYER)
    if tuple(layer.weight.shape) != (64, 64, 3, 3) or layer.groups != 1:
        raise ValueError('unexpected model architecture at ' + LAYER)
    # Independent generators keep calibration selection stable across model-loading changes.
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(train), generator=generator)[:args.calibration_images].tolist()
    selected = None
    captured = []
    offset = 0

    def capture(module, inputs):
        nonlocal selected, offset
        patches = F.unfold(inputs[0], module.kernel_size, dilation=module.dilation,
                           padding=module.padding, stride=module.stride)
        patches = patches.permute(1, 0, 2).reshape(576, -1)
        if selected is None:
            positions_per_image = patches.shape[1] // len(inputs[0])
            population = args.calibration_images * positions_per_image
            if population < args.samples:
                raise ValueError('not enough patches; increase --calibration-images')
            selected = torch.randperm(population, generator=generator)[:args.samples].sort().values
        local = selected[(selected >= offset) & (selected < offset + patches.shape[1])] - offset
        captured.append(patches[:, local].double())
        offset += patches.shape[1]

    handle = layer.register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            for images, _ in torch.utils.data.DataLoader(
                    torch.utils.data.Subset(train, indices), batch_size=32):
                model(images)
    finally:
        handle.remove()
    X = torch.cat(captured, dim=1).numpy()
    if X.shape != (576, args.samples) or not np.isfinite(X).all():
        raise ValueError('invalid calibration activations')
    weights = layer.weight.detach().reshape(64, 576).double().numpy()
    if not np.isfinite(weights).all():
        raise ValueError('nonfinite model weights')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output / 'original_state.pt')
    np.savez(args.output / 'calibration.npz', X=X, weights=weights)
    manifest = {
        'format_version': 1, 'model_repo': MODEL_REPO, 'layer': LAYER,
        'weights_per_row': 576, 'k': args.k, 'samples': args.samples, 'seed': args.seed,
        'calibration_indices': indices, 'patch_indices': selected.tolist(), 'rows': [],
        'scope': 'Only the selected rows of layer3.2.conv2 are changed; no fine-tuning.',
        'torch_version': torch.__version__,
        'checkpoint_sha256': digest(args.output / 'original_state.pt'),
        'calibration_sha256': digest(args.output / 'calibration.npz'),
    }
    for row in range(args.rows):
        w = weights[row]
        scale = max(float(w.max()) / 7, float(-w.min()) / 8, 1e-12)
        baseline = np.zeros(576)
        support = np.argsort(-np.abs(w), kind='stable')[:args.k]
        baseline[support] = np.clip(np.rint(w[support] / scale), -8, 7)
        path = args.output / f'row_{row:02d}.txt'
        with path.open('w') as stream:
            stream.write(f'576 {args.samples}\n')
            np.savetxt(stream, (w / scale)[None, :], fmt='%.17g')
            np.savetxt(stream, scale * X, fmt='%.17g')
        baseline_loss = float(np.square((w - scale * baseline) @ X).sum())
        manifest['rows'].append(dict(row=row, scale=scale, baseline_loss=baseline_loss,
                                     baseline_z=baseline.astype(int).tolist(), input_sha256=digest(path)))
    write_json(args.output / 'manifest.json', manifest)
    print(f'Exported {args.rows} rows of 576 weights to {args.output}', flush=True)


def evaluate(args):
    import numpy as np
    import torch

    manifest = json.loads((args.output / 'manifest.json').read_text())
    if manifest.get('format_version') != 1 or manifest['layer'] != LAYER:
        raise ValueError('unsupported experiment manifest')
    for name, key in (('original_state.pt', 'checkpoint_sha256'),
                      ('calibration.npz', 'calibration_sha256')):
        if digest(args.output / name) != manifest[key]:
            raise ValueError(f'{name}: checksum mismatch')
    model = load_model(args, pretrained=False)
    model.load_state_dict(torch.load(args.output / 'original_state.pt', map_location='cpu',
                                     weights_only=True), strict=True)
    baseline_model, solved_model = copy.deepcopy(model), copy.deepcopy(model)
    with np.load(args.output / 'calibration.npz', allow_pickle=False) as calibration:
        X, weights = calibration['X'], calibration['weights']
    if X.shape != (576, manifest['samples']) or weights.shape != (64, 576):
        raise ValueError('invalid calibration dimensions')
    if not np.array_equal(model.get_submodule(LAYER).weight.detach().reshape(64, 576).double().numpy(), weights):
        raise ValueError('calibration weights do not match saved checkpoint')
    for item in manifest['rows']:
        row, scale = item['row'], item['scale']
        path = args.output / f'row_{row:02d}.txt'
        if digest(path) != item['input_sha256']:
            raise ValueError(f'{path}: checksum mismatch')
        identity = json.loads((args.output / f'row_{row:02d}.run.json').read_text())
        if (identity['input_sha256'] != item['input_sha256'] or identity['k'] != manifest['k']
                or identity['samples'] != manifest['samples']):
            raise ValueError('solution belongs to a different experiment')
        result = read_solution(args.output / f'row_{row:02d}.solution', 576, manifest['k'])
        actual = float(np.square((weights[row] - scale * np.array(result['z'])) @ X).sum())
        if not np.isclose(result['solver_loss'], actual, rtol=1e-7, atol=1e-7):
            raise ValueError(f'row {row}: solver loss differs from direct reconstruction loss')
        baseline_loss = item['baseline_loss']
        result['relative_improvement'] = ((baseline_loss-actual)/baseline_loss if baseline_loss else None)
        result['direct_reconstruction_loss'] = actual
        item.update(result)
        for target, values in ((baseline_model, item['baseline_z']), (solved_model, result['z'])):
            with torch.no_grad():
                target.get_submodule(LAYER).weight[row].copy_(
                    torch.from_numpy(scale * np.array(values)).reshape(64, 3, 3))
    loader = torch.utils.data.DataLoader(dataset(args, train=False), batch_size=128)
    for name, target in (('original', model), ('baseline', baseline_model), ('solver', solved_model)):
        correct = total = 0
        with torch.no_grad():
            for images, labels in loader:
                correct += int((target(images).argmax(1) == labels).sum())
                total += len(labels)
        manifest[name + '_test_accuracy'] = correct / total
        print(f'{name}: {correct}/{total} = {correct/total:.4%}', flush=True)
    torch.save(solved_model.state_dict(), args.output / 'solved_state.pt')
    write_json(args.output / 'report.json', manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('all', 'prepare', 'solve', 'evaluate'), default='all')
    parser.add_argument('--solver', type=Path)
    parser.add_argument('--data', type=Path, default=Path('experiments/data'))
    parser.add_argument('--output', type=Path, default=Path('experiments/results/resnet20'))
    parser.add_argument('--model-dir', type=Path, help='Local clone of pytorch-cifar-models')
    parser.add_argument('--checkpoint', type=Path, help='Local model state_dict for prepare stage')
    parser.add_argument('--offline', action='store_true', help='Require local model code and existing data')
    parser.add_argument('--rows', type=int, default=1, help='Prepare first N output channels, 1..64')
    parser.add_argument('--samples', type=int, default=2048)
    parser.add_argument('--calibration-images', type=int, default=128)
    parser.add_argument('--k', type=int, default=288)
    parser.add_argument('--time-limit', type=float, default=60)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true', help='Resume matching completed rows in solve stage')
    parser.add_argument('--prepare-only', action='store_true', help='Alias for --stage prepare')
    args = parser.parse_args()
    if args.prepare_only:
        args.stage = 'prepare'
    if not (1 <= args.rows <= 64 and 0 <= args.k <= 576 and 0 <= args.seed <= 1000000
            and 0 < args.samples <= 1000000 and 1 <= args.calibration_images <= 50000
            and 0 < args.time_limit < float('inf')):
        parser.error('invalid dimensions, budget, seed or time limit')
    if args.stage in ('all', 'solve') and (args.solver is None or not args.solver.is_file()):
        parser.error('--solver must name an existing executable for all/solve stages')
    if args.offline and args.stage != 'solve' and not args.model_dir:
        parser.error('--offline requires --model-dir for prepare/evaluate')
    if args.offline and args.stage in ('all', 'prepare') and args.checkpoint is None:
        parser.error('--offline preparation requires a local --checkpoint state_dict')
    if args.resume and args.stage != 'solve':
        parser.error('--resume is only supported with --stage solve')
    try:
        if args.stage != 'solve':
            import torch
            torch.manual_seed(args.seed)
            torch.set_num_threads(4)
        if args.stage in ('all', 'prepare'):
            prepare(args)
        if args.stage in ('all', 'solve'):
            solve(args.output, args.solver, args.time_limit, args.resume)
        if args.stage in ('all', 'evaluate'):
            evaluate(args)
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'error: {error}\n')


if __name__ == '__main__':
    main()
