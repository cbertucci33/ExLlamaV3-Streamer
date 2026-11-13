"""Stdlib-only QA: AST-extract real parser/resume preflight, never import EXL3.
Run: python -m unittest discover -s streamer_tests -v
List/legacy tuple contracts are emulated; native contract is tested unchanged.
"""
import argparse
import ast
import copy
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "exllamav3/conversion/convert_model.py"
TREE = ast.parse(SOURCE.read_text())
start = next(i for i, n in enumerate(TREE.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "parser" for t in n.targets))
end = next(i for i in range(start, len(TREE.body)) if isinstance(TREE.body[i], ast.Assign) and any(isinstance(t, ast.Name) and t.id == "num_ref_states" for t in TREE.body[i].targets))
ns = {"argparse": argparse}
exec(compile(ast.Module(body=TREE.body[start:end], type_ignores=[]), str(SOURCE), "exec"), ns)
PARSER = ns["parser"]
PREPARE = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "prepare")
stop = next(i for i, n in enumerate(PREPARE.body) if isinstance(n, ast.If) and ast.unparse(n.test) == "not args.work_dir")
PREFIX = copy.deepcopy(PREPARE.body[:stop])
NORMALIZE = ast.parse("""if isinstance(parsed, list):
    parsed = parsed[0] if parsed else None
""").body[0]

def make_preflight(fixed):
    body = copy.deepcopy(PREFIX)
    loop = next(n for n in body if isinstance(n, ast.For))
    loop.body = [n for n in loop.body if ast.dump(n, include_attributes=False) != ast.dump(NORMALIZE, include_attributes=False)]
    if fixed:
        loop = next(n for n in body if isinstance(n, ast.For))
        index = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "parsed")
        loop.body.insert(index + 1, copy.deepcopy(NORMALIZE))
    body += ast.parse("return args, explicit").body
    func = ast.FunctionDef(name="preflight", args=copy.deepcopy(PREPARE.args), body=body, decorator_list=[], returns=None)
    module = ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[]))
    env = {"sys": sys, "os": os, "parser": PARSER, "check_system": lambda: None}
    exec(compile(module, str(SOURCE), "exec"), env)
    return env["preflight"], env

NATIVE = PARSER._parse_optional

def contract(mode):
    def parse(token):
        result = NATIVE(token)
        if result is None or mode == "native":
            return result
        if isinstance(result, list):
            result = result[0]
        action, option = result[:2]
        explicit_arg = result[-1]
        item = (action, option, explicit_arg) if mode.endswith("3") else (action, option, "=" if explicit_arg is not None else None, explicit_arg)
        return [item] if mode.startswith("list") else item
    return parse

class OptionalQA(unittest.TestCase):
    def run_case(self, argv, mode="native", fixed=True, saved=None):
        # Public parser validation precedes prepare in production too.
        args = PARSER.parse_args(argv)
        fn, env = make_preflight(fixed)
        if saved is None:
            saved = vars(PARSER.parse_args(["-w", "old", "-i", "input", "-o", "output", "-b", "4", "--calibration_state_storage", "disk"])).copy()
        env["disk_runtime"] = types.SimpleNamespace(load_checkpoint=lambda root: {"args": saved.copy()})
        with patch.object(sys, "argv", ["convert.py"] + argv), patch.object(PARSER, "_parse_optional", side_effect=contract(mode)), patch.object(os.path, "isfile", return_value=True):
            return fn(args)

    def test_original_reproduces_reported_failure(self):
        for mode in ("list3", "list4"):
            with self.subTest(mode=mode), self.assertRaisesRegex(AttributeError, "tuple.*dest"):
                self.run_case(["-w", "work", "-b", "4"], mode, fixed=False)

    def test_normalized_cross_version_matrix(self):
        modes = ("native", "tuple3", "tuple4", "list3", "list4")
        cases = [
            (["-w", "work", "--bits", "4"], {"work_dir", "bits"}),
            (["--work_dir=work", "--bits=4"], {"work_dir", "bits"}),
            (["-wwork", "-b4"], {"work_dir", "bits"}),
            (["-w", "work", "--verbose"], {"work_dir", "verbose"}),
            (["-w", "work", "--bits", "-4"], {"work_dir", "bits"}),
            (["-w", "work", "--bits=4", "-b4", "-hq"], {"work_dir", "bits", "hq"}),
        ]
        for mode in modes:
            for argv, expected in cases:
                with self.subTest(mode=mode, argv=argv):
                    self.assertEqual(self.run_case(argv, mode)[1], expected)

    def test_disk_resume_guards_and_inheritance(self):
        for mode in ("native", "tuple3", "tuple4", "list3", "list4"):
            with self.subTest(mode=mode):
                args, explicit = self.run_case(["-w", "new", "-r"], mode)
                self.assertEqual(args.bits, 4)
                self.assertEqual(args.work_dir, "new")
                self.assertEqual(args.calibration_state_storage, "disk")
                self.assertEqual(explicit, {"work_dir", "resume"})
                self.run_case(["-w", "new", "-r", "--bits=4"], mode)
                for option, error in [
                    (["--bits=3"], "conflicts.*bits"),
                    (["--out_scales=never"], "conflicts.*out_scales"),
                    (["--override_anyway"], "cannot override disk"),
                    (["--calibration_state_storage=memory"], "Cannot switch storage"),
                ]:
                    with self.assertRaisesRegex(ValueError, error):
                        self.run_case(["-w", "new", "-r"] + option, mode)
                args, _ = self.run_case(["-w", "new", "-r", "--max_module=2", "--checkpoint_interval=9", "--last_checkpoint_index=1", "--parallel_mode"], mode)
                self.assertEqual((args.max_module, args.checkpoint_interval, args.last_checkpoint_index, args.parallel_mode), (2, 9, 1, True))

    def test_double_dash_stops_scan(self):
        fn, _ = make_preflight(True)
        with patch.object(sys, "argv", ["convert.py", "-w", "work", "--", "--bits=3"]):
            self.assertEqual(fn(PARSER.parse_args(["-w", "work"]))[1], {"work_dir"})

    def test_unknown_and_negative_not_explicit(self):
        fn, _ = make_preflight(True)
        for result in (None, (None, "--unknown", None), [(None, "--unknown", None)], []):
            with self.subTest(result=result), patch.object(sys, "argv", ["convert.py", "--unknown"]), patch.object(PARSER, "_parse_optional", return_value=result):
                args = PARSER.parse_args([])
                self.assertEqual(fn(args)[1], set())

if __name__ == "__main__":
    print("Interpreter:", sys.version)
    print("Native _parse_optional shape:", repr(NATIVE("--bits")))
    unittest.main(verbosity=2)
