"""CPU storage regressions; GPU runtime initialization is intentionally excluded."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = '_streamer_storage_tests'
for suffix, directory in [('', 'exllamav3'), ('.conversion', 'exllamav3/conversion'),
                          ('.loader', 'exllamav3/loader')]:
    module = types.ModuleType(BASE + suffix)
    module.__path__ = [str(ROOT / directory)]
    sys.modules[BASE + suffix] = module

def load(name):
    return importlib.import_module(BASE + '.' + name)

state = load('conversion.state_store')
hessian = load('conversion.hessian_store')
stream = load('conversion.streaming')
runtime = load('conversion.disk_runtime')
loader = load('loader.safetensors_alt')

class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pack = state.Pack(self.root / 'packs')

    def roundtrip(self, value):
        return self.pack.decode(json.loads(json.dumps(self.pack.encode(value))))

    def test_tensor_dtypes_and_layouts(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64,
                      torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool):
            for tensor in (torch.arange(12).reshape(3, 4).to(dtype),
                           torch.arange(12).reshape(3, 4).to(dtype).T,
                           torch.ones(1, 4, dtype=dtype).expand(3, 4),
                           torch.empty(0, dtype=dtype), torch.tensor(1, dtype=dtype)):
                with self.subTest(dtype=dtype, shape=tensor.shape):
                    restored = self.roundtrip(tensor)
                    self.assertTrue(torch.equal(tensor, restored))
                    self.assertEqual(tensor.stride(), restored.stride())

    def test_aliases_and_container_types(self):
        tensor = torch.arange(5)
        shared = [tensor]
        result = self.roundtrip({'a': shared, 'b': shared, 'c': (tensor, torch.device('cpu'))})
        self.assertIs(result['a'], result['b'])
        self.assertIs(result['a'][0], result['c'][0])
        self.assertIsInstance(result['c'], tuple)
        self.assertEqual(result['c'][1], torch.device('cpu'))

    def test_numpy_primitives(self):
        for kind in (np.bool_, np.int8, np.int16, np.int32, np.int64, np.longlong,
                     np.uint8, np.uint16, np.uint32, np.uint64, np.ulonglong,
                     np.float16, np.float32, np.float64, np.str_):
            value = kind('sample') if kind is np.str_ else kind(1)
            self.assertEqual(self.roundtrip(value), value.item())
        for value in (np.array([1]), np.complex64(1j), object()):
            with self.assertRaises(TypeError): self.pack.encode(value)

    def test_row_slices_and_incomplete_generation(self):
        tensor = torch.arange(30).reshape(10, 3)
        record = self.pack.put(tensor)
        self.assertTrue(torch.equal(self.pack.get_rows(record, 3, 4), tensor[3:7]))
        writer = state.Writer(self.root / 'rows', 2)
        writer.write_row(0, tensor, {'x': tensor})
        with self.assertRaises(RuntimeError): writer.close()
        writer.skip_row(1, 'excluded')
        generation = writer.close()
        hidden, preserve = generation.read_row(0)
        self.assertIs(hidden, preserve['x'])
        self.assertIsNone(generation.read_row(1))

    def test_hessian_lease_eviction_and_retirement(self):
        store = hessian.Hessians(self.root / 'hessians')
        store.add({'q': {'H': torch.eye(4), 'count': np.int64(3),
                         'num_total': 0, 'inf_nan': torch.zeros(2, dtype=torch.long)}})
        store.plan([[types.SimpleNamespace(qmap='q')], [types.SimpleNamespace(qmap='q')]])
        with self.assertRaises(RuntimeError): store['q']
        old = set((self.root / 'hessians').glob('*.pack'))
        with store.lease(['q']): store['q']['H'].mul_(2)
        self.assertFalse(any(p.exists() for p in old))
        with store.lease(['q']): self.assertTrue(torch.equal(store['q']['H'], torch.eye(4) * 2))
        self.assertNotIn('q', store)
        self.assertFalse(list((self.root / 'hessians').glob('*.pack')))

    def test_failed_lease_does_not_complete_group(self):
        store = hessian.Hessians(self.root / 'hessians')
        store.add({'q': {'H': torch.eye(2), 'num_total': 0, 'inf_nan': 0}})
        store.plan([[types.SimpleNamespace(qmap='q')]])
        with self.assertRaisesRegex(ValueError, 'injected'):
            with store.lease(['q']): raise ValueError('injected')
        self.assertEqual(store.remaining['q'], 1)
        with store.lease(['q']): self.assertTrue(torch.equal(store['q']['H'], torch.eye(2)))
        self.assertNotIn('q', store)

    def test_chunked_compilation_and_truncated_input(self):
        tensors = {'a': torch.arange(48).reshape(12, 4), 'b': torch.tensor(2.0),
                   'c': torch.empty(0), 'd': torch.ones(4, dtype=torch.bfloat16)}
        original = self.root / 'input.safetensors'
        loader.save_file(tensors, str(original))
        with loader.safe_open(str(original)) as opened:
            self.assertEqual(set(opened.keys()), set(tensors))
        # Exercise mixed file-backed and transformed in-memory entries.
        raw = original.read_bytes()
        size = int.from_bytes(raw[:8], 'little')
        header = json.loads(raw[8:8+size])
        selected = {}
        for key in ('a', 'c', 'd'):
            start, end = header[key]['data_offsets']
            selected[key] = stream.Entry(header[key]['dtype'], tuple(header[key]['shape']),
                                         end-start, str(original), 8+size+start)
        selected['b'] = stream.tensor_entry(tensors['b'])
        output = str(self.root / 'output.safetensors')
        with patch.object(stream, 'CHUNK', 16):
            stream.write_entries(selected, output)
        with loader.safe_open(output) as opened:
            for key, tensor in tensors.items(): self.assertTrue(torch.equal(opened.get_tensor(key), tensor))
        selected = {'bad': stream.Entry('I64', (1000,), 8000, str(original), len(raw))}
        with self.assertRaises(EOFError): stream.write_entries(selected, str(self.root / 'bad.safetensors'))
        self.assertFalse((self.root / 'bad.safetensors').exists())

    def test_checkpoint_restore_and_pack_reclamation(self):
        root = str(self.root / 'work')
        tokens = [torch.arange(4).view(1, -1), torch.arange(4, 8).view(1, -1)]
        rows, saved_tokens = runtime.initialize(root, tokens)
        orphan = state.Pack(Path(root) / 'orphan')
        orphan.put(torch.ones(4))
        runtime.checkpoint(root, {'devices': ''}, {'next_module_idx': 1}, rows, {}, {'step': 1})
        saved = runtime.load_checkpoint(root)
        restored, token_rows = runtime.restore_rows(root, saved)
        runtime.restore_rng(saved)
        self.assertEqual(saved['job']['next_module_idx'], 1)
        self.assertEqual(saved['scheduler']['step'], 1)
        self.assertEqual(token_rows.num_tokens, 8)
        self.assertTrue(torch.equal(restored.generation.read_row(1)[0], tokens[1]))
        self.assertTrue(torch.equal(token_rows[0], tokens[0]))
        self.assertFalse((Path(root) / 'orphan').exists())
        self.assertFalse(list(Path(root).rglob('*.partial')))

if __name__ == '__main__': unittest.main(verbosity=2)
