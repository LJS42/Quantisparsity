"""CPU-only checks for the data/solver boundary; no PyTorch or GPU required."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

from solve_exported_rows import digest, read_solution, reconstruction_loss, solve, write_json


class ExportedRowsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / 'row_00.txt'
        # w=[.25,-.5], scale=.25, X=[[1,2],[3,4]].
        # z=[0,-2] => physical residual [.25,.5], SSE=.3125.
        self.input.write_text('2 2\n1 -2\n.25 .5\n.75 1\n')
        self.manifest = dict(format_version=1, weights_per_row=2, k=1, samples=2,
                             seed=42, rows=[dict(row=0, baseline_loss=.3125,
                                                input_sha256=digest(self.input))])
        write_json(self.root / 'manifest.json', self.manifest)
        self.solver = self.root / 'fake_solver'
        self.fake_solver('0.3125 0 0.01 1 1\n0 -2\n')

    def fake_solver(self, result):
        self.solver.write_text(f'#!{sys.executable}\n'
                               'from pathlib import Path\nimport sys\n'
                               "counter=Path(__file__).with_suffix('.count')\n"
                               "counter.write_text(str(int(counter.read_text())+1) if counter.exists() else '1')\n"
                               f"Path(sys.argv[sys.argv.index('--output')+1]).write_text({result!r})\n")
        self.solver.chmod(0o755)

    def test_scale_and_correlated_inputs(self):
        self.assertEqual(reconstruction_loss(self.input, [0, -2], 2), .3125)
        # Changing the second weight introduces cross terms; blocks aren't separable.
        self.assertEqual(reconstruction_loss(self.input, [0, -1], 2), .5)

    def test_round_trip_and_resume(self):
        report = solve(self.root, self.solver, 1)
        self.assertTrue(report['rows'][0]['time_limit_reached'])
        self.assertEqual(report['rows'][0]['direct_reconstruction_loss'], .3125)
        resumed = solve(self.root, self.solver, 1, resume=True)
        self.assertEqual(resumed, report)
        self.assertEqual(self.solver.with_suffix('.count').read_text(), '1')
        with self.assertRaisesRegex(ValueError, 'existing run'):
            solve(self.root, self.solver, 1)
        with self.assertRaisesRegex(ValueError, 'settings differ'):
            solve(self.root, self.solver, 2, resume=True)

    def test_tampered_input(self):
        self.input.write_text(self.input.read_text() + '0\n')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            solve(self.root, self.solver, 1)

    def test_wrong_objective_rejected(self):
        self.fake_solver('0.5 0 0.01 1 1\n0 -2\n')
        with self.assertRaisesRegex(ValueError, 'differs from reconstruction'):
            solve(self.root, self.solver, 1)
        self.assertFalse((self.root / 'row_00.solution').exists())

    def test_resume_incomplete_commit_reruns(self):
        # Simulate interruption between committing the solution and its metadata.
        (self.root / 'row_00.solution').write_text('0.3125 0 0.01 1 1\n0 -2\n')
        solve(self.root, self.solver, 1, resume=True)
        self.assertTrue((self.root / 'row_00.run.json').exists())
        self.assertEqual(self.solver.with_suffix('.count').read_text(), '1')

    def test_invalid_solutions(self):
        path = self.root / 'bad.solution'
        for text in ('nan 0 1 1 0\n0 -2\n', '1 2 1 1 0\n0 -2\n',
                     '1 0 1 1 0\n1 -2\n', '1 0 1 1 0\n0 -9\n',
                     '1 0 1 1 0\n0 -1.5\n', '1 0 1 1 0\n0\n'):
            with self.subTest(text=text):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read_solution(path, 2, 1)

    def test_unbounded_certificate_is_valid_json(self):
        self.fake_solver('0.3125 -inf 0.01 1 1\n0 -2\n')
        solve(self.root, self.solver, 1)
        report = json.loads((self.root / 'solver_report.json').read_text())
        self.assertIsNone(report['rows'][0]['dual_bound'])

    def test_truncated_input(self):
        self.input.write_text('2 2\n1 -2\n.25 .5\n.75\n')
        with self.assertRaisesRegex(ValueError, 'truncated'):
            reconstruction_loss(self.input, [0, -2], 2)


if __name__ == '__main__':
    unittest.main()
