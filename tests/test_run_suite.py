"""CLI safety retained after removing the experimental policy routers."""
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


class TestRunSuite(unittest.TestCase):
    def test_run_suite_rejects_stale_solution_artifacts(self):
        from m3d.cli import cmd_run_suite
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'case_01.sol.json').write_text('{}')
            # This must be rejected before any suite file is opened or routing starts.
            args = Namespace(router='negotiated', out_dir=directory,
                             suite='does-not-exist', policy=None)
            with self.assertRaisesRegex(ValueError, 'empty'):
                cmd_run_suite(args)
