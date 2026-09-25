"""Run exported GPUBoscia rows on a CUDA host; Python standard library only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_solution(path, dimension, k):
    lines = path.read_text().splitlines()
    if len(lines) != 2 or len(lines[0].split()) != 5:
        raise ValueError(f'{path}: expected statistics and weight lines')
    loss, bound, seconds, nodes, timed_out = map(float, lines[0].split())
    if (not math.isfinite(loss) or loss < 0 or not math.isfinite(seconds) or seconds < 0
            or not math.isfinite(nodes) or nodes < 0 or nodes != int(nodes)
            or timed_out not in (0, 1) or math.isnan(bound) or bound == math.inf
            or bound > loss + 1e-6 * (1 + abs(loss))):
        raise ValueError(f'{path}: invalid solver statistics or lower bound')
    z = [float(value) for value in lines[1].split()]
    if (len(z) != dimension or any(not math.isfinite(v) or v != round(v)
            or v < -8 or v > 7 for v in z) or sum(v != 0 for v in z) > k):
        raise ValueError(f'{path}: infeasible INT4 solution')
    return dict(solver_loss=loss, dual_bound=bound if math.isfinite(bound) else None,
                seconds=seconds, nodes=int(nodes), time_limit_reached=bool(timed_out),
                nnz=sum(v != 0 for v in z), z=[int(v) for v in z])


def reconstruction_loss(path, z, samples):
    """Independent streamed check of ||(z-w)X||^2; no numerical packages needed."""
    with path.open() as stream:
        tokens = (token for line in stream for token in line.split())
        try:
            dimension, count = int(next(tokens)), int(next(tokens))
            if dimension != len(z) or count != samples:
                raise ValueError(f'{path}: wrong dimensions')
            w = [float(next(tokens)) for _ in range(dimension)]
            if not all(math.isfinite(value) for value in w):
                raise ValueError(f'{path}: nonfinite weights')
            residual = [0.0] * samples
            for i in range(dimension):
                delta = z[i] - w[i]
                for j in range(samples):
                    value = float(next(tokens))
                    if not math.isfinite(value):
                        raise ValueError(f'{path}: nonfinite activations')
                    residual[j] += delta * value
            if next(tokens, None) is not None:
                raise ValueError(f'{path}: trailing input')
        except StopIteration as error:
            raise ValueError(f'{path}: truncated input') from error
    loss = math.fsum(value * value for value in residual)
    if not math.isfinite(loss):
        raise ValueError(f'{path}: nonfinite reconstruction loss')
    return loss


def solve(output, solver, time_limit, resume=False):
    if not solver.is_file():
        raise ValueError(f'solver binary not found: {solver}')
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError('time limit must be finite and positive')
    report = json.loads((output / 'manifest.json').read_text())
    if report.get('format_version') != 1:
        raise ValueError('unsupported experiment format')
    dimension, k = report['weights_per_row'], report['k']
    if not (dimension > 0 and 0 <= k <= dimension and 0 < report['samples'] <= 1000000):
        raise ValueError('invalid experiment dimensions or budget')
    report['solver_sha256'] = digest(solver)
    report['time_limit_per_row'] = time_limit
    for item in report['rows']:
        row = item['row']
        stem = f'row_{row:02d}'
        input_path = output / (stem + '.txt')
        if digest(input_path) != item['input_sha256']:
            raise ValueError(f'{input_path}: input checksum mismatch')
        result_path = output / (stem + '.solution')
        metadata_path = output / (stem + '.run.json')
        identity = dict(input_sha256=item['input_sha256'], k=k, samples=report['samples'],
                        solver_sha256=report['solver_sha256'], time_limit=time_limit,
                        seed=report['seed'])
        if resume and metadata_path.exists():
            if json.loads(metadata_path.read_text()) != identity:
                raise ValueError(f'{result_path}: resume settings differ; use a new output directory')
        if not (resume and result_path.exists() and metadata_path.exists()):
            if not resume and (result_path.exists() or metadata_path.exists()):
                raise ValueError(f'{result_path}: existing run; use --resume or a new output directory')
            # A failed process can leave a temporary result, but never a resumable result.
            pending = result_path.with_suffix('.solution.pending')
            log_path = output / (stem + '.log')
            command = [str(solver.resolve()), '--input', str(input_path), '--output', str(pending),
                       '--samples', str(report['samples']), '--k', str(k),
                       '--time-limit', str(time_limit), '--seed', str(report['seed'])]
            print(f'Running row {row} ({dimension} weights); log: {log_path}', flush=True)
            with log_path.open('w') as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f'row {row} exited with {completed.returncode}; see {log_path}')
            pending_result = read_solution(pending, dimension, k)
            actual = reconstruction_loss(input_path, pending_result['z'], report['samples'])
            if not math.isclose(actual, pending_result['solver_loss'], rel_tol=1e-7, abs_tol=1e-7):
                raise ValueError(f'{pending}: reported loss differs from reconstruction loss')
            pending.replace(result_path)
            write_json(metadata_path, identity)
        result = read_solution(result_path, dimension, k)
        actual = reconstruction_loss(input_path, result['z'], report['samples'])
        if not math.isclose(actual, result['solver_loss'], rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError(f'{result_path}: reported loss differs from reconstruction loss')
        result['direct_reconstruction_loss'] = actual
        baseline_loss = item['baseline_loss']
        result['relative_improvement'] = ((baseline_loss-result['solver_loss']) / baseline_loss
                                          if baseline_loss else None)
        item.update(result)
        write_json(output / 'solver_report.json', report)
        print(f"Row {row}: baseline={baseline_loss:.8g}, solver={result['solver_loss']:.8g}, "
              f"seconds={result['seconds']:.3f}, timeout={result['time_limit_reached']}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Exported experiment directory')
    parser.add_argument('--solver', type=Path, required=True)
    parser.add_argument('--time-limit', type=float, default=60)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    try:
        solve(args.output, args.solver, args.time_limit, args.resume)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'error: {error}\n')


if __name__ == '__main__':
    main()
