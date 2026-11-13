"""Synchronous finite-tree calibration storage (no pickle or mmap)."""
import json
import os
import threading
import uuid

CHUNK = 8 << 20
PACK_BYTES = 1 << 30


def atomic_json(path, value):
    temporary = str(path) + '.partial'
    with open(temporary, 'w', encoding='utf8') as f:
        json.dump(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


class Pack:
    """One writer lane. Reads have independent, immediately closed handles."""
    def __init__(self, directory):
        self.directory = os.path.abspath(os.fspath(directory))
        os.makedirs(self.directory, exist_ok=True)
        self.name = None
        self.size = 0

    def put(self, tensor):
        import torch
        from ..loader.safetensors_alt import _tensor_bytes_view, _stream_write
        t = tensor.detach().to('cpu').contiguous()
        size = t.numel() * t.element_size()
        if self.name is None or self.size + size > PACK_BYTES:
            self.name = uuid.uuid4().hex + '.pack'
            self.size = 0
        offset = self.size
        with open(os.path.join(self.directory, self.name), 'ab') as f:
            _stream_write(f, _tensor_bytes_view(t), CHUNK)
            f.flush()
        self.size += size
        return dict(directory=self.directory, file=self.name, offset=offset, size=size,
                    stride=list(tensor.stride()),
                    dtype=str(t.dtype).split('.')[-1], shape=list(t.shape),
                    device=str(tensor.device))

    def get(self, descriptor, restore_device=False):
        import torch
        from ..loader.safetensors_alt import _tensor_bytes_view, _stream_readinto
        d = descriptor
        t = torch.empty(d['shape'], dtype=getattr(torch, d['dtype']), device='cpu')
        with open(os.path.join(d.get('directory', self.directory), d['file']), 'rb') as f:
            f.seek(d['offset'])
            _stream_readinto(f, _tensor_bytes_view(t), d['size'], CHUNK)
        stride = tuple(d.get('stride', t.stride()))
        if stride != t.stride():
            shape = tuple(t.shape)
            compact_shape = tuple(1 if st == 0 and sz else sz for sz, st in zip(shape, stride))
            span = 0 if t.numel() == 0 else 1 + sum((sz-1)*st for sz,st in zip(compact_shape,stride))
            if any(st < 0 for st in stride) or span > t.numel():
                # No required persistent producer at the pin emits gapped/negative
                # layouts. Do not copy unrelated allocator storage to emulate one.
                raise ValueError('Calibration layout requires non-row backing storage')
            base = torch.empty_strided(compact_shape, stride, dtype=t.dtype)
            source = t[tuple(slice(0,1) if st == 0 else slice(None) for st in stride)]
            base.copy_(source)
            t = base.as_strided(shape, stride)
        return t.to(d['device']) if restore_device else t

    def get_rows(self, descriptor, start, count):
        import torch
        from ..loader.safetensors_alt import _tensor_bytes_view, _stream_readinto
        d = descriptor
        shape = list(d['shape'])
        row_elements = 1
        for n in shape[1:]: row_elements *= n
        shape[0] = count
        t = torch.empty(shape, dtype=getattr(torch, d['dtype']))
        with open(os.path.join(d.get('directory', self.directory), d['file']), 'rb') as f:
            f.seek(d['offset'] + start * row_elements * t.element_size())
            _stream_readinto(f, _tensor_bytes_view(t), t.numel()*t.element_size(), CHUNK)
        return t

    def encode(self, value, reuse=None, tensor_pack=None):
        import torch
        import numpy as np
        # Only exact scalar types whose item() is lossless in the existing
        # primitive schema. Do not coerce arrays, user objects, complex values
        # or extended-precision floats through int()/float().
        numpy_primitives = (np.bool_, np.int8, np.int16, np.int32, np.int64, np.longlong,
                            np.uint8, np.uint16, np.uint32, np.uint64, np.ulonglong,
                            np.float16, np.float32, np.float64, np.str_)
        memo = {}
        def visit(v):
            if v is None or type(v) in (str, int, bool, float):
                return ['value', v]
            if type(v) in numpy_primitives:
                return ['value', v.item()]
            if isinstance(v, torch.device):
                return ['device', str(v)]
            ident = id(v)
            if ident in memo:
                return ['ref', memo[ident]]
            index = len(memo)
            memo[ident] = index
            if torch.is_tensor(v):
                return ['tensor', index, (reuse or {}).get(id(v)) or (tensor_pack or self).put(v)]
            if isinstance(v, dict):
                return ['dict', index, [[visit(k), visit(x)] for k, x in v.items()]]
            if isinstance(v, (list, tuple)):
                return [type(v).__name__, index, [visit(x) for x in v]]
            raise TypeError('Unsupported calibration value: ' + type(v).__name__)
        return visit(value)

    def decode(self, record, restore_device=False, descriptors=None):
        decoder = _Decoder(self, restore_device, descriptors)
        try:
            return decoder.visit(record)
        finally:
            # Returned containers own their live aliases. The temporary identity
            # memo must never extend a row/Hessian payload's lifetime.
            decoder.memo.clear()


class _Decoder:
    def __init__(self, pack, restore_device, descriptors):
        self.pack = pack
        self.restore_device = restore_device
        self.descriptors = descriptors
        self.memo = {}

    def visit(self, record):
        import torch
        tag = record[0]
        if tag == 'value': return record[1]
        if tag == 'device': return torch.device(record[1])
        if tag == 'ref': return self.memo[record[1]]
        if tag == 'tensor':
            result = self.pack.get(record[2], self.restore_device)
            if self.descriptors is not None:
                self.descriptors[id(result)] = record[2]
        elif tag == 'dict':
            result = {}
            self.memo[record[1]] = result
            for key, value in record[2]:
                result[self.visit(key)] = self.visit(value)
        elif tag in ('list', 'tuple'):
            result = [self.visit(value) for value in record[2]]
            if tag == 'tuple': result = tuple(result)
        else:
            raise ValueError(tag)
        self.memo[record[1]] = result
        return result


class Generation:
    """Immutable metadata; every read reconstructs fresh call-owned values."""
    def __init__(self, directory, rows):
        self.directory = os.path.abspath(os.fspath(directory))
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def read_row(self, row_id):
        return Pack(self.directory).decode(self.rows[row_id])


class Writer:
    def __init__(self, directory, count):
        self.directory = os.path.abspath(os.fspath(directory))
        os.makedirs(self.directory, exist_ok=True)
        self.rows = [None] * count
        self.local = threading.local()

    def write_row(self, row_id, hidden, preserve, reuse=None):
        if not hasattr(self.local, 'pack'):
            self.local.pack = Pack(os.path.join(self.directory, 'preserve'))
            self.local.hidden = Pack(os.path.join(self.directory, 'hidden'))
        # Encode hidden first using its own stream; preserve values go to the other
        # stream. One memo remains authoritative for the complete row bundle.
        pack = self.local.pack
        original_put = pack.put
        first = [True]
        def put(t):
            if first[0] and t is hidden:
                first[0] = False
                return self.local.hidden.put(t)
            return original_put(t)
        pack.put = put
        try:
            self.rows[row_id] = pack.encode((hidden, preserve), reuse=reuse)
        finally:
            pack.put = original_put

    def skip_row(self, row_id, reason):
        self.rows[row_id] = ['value', None]

    def close(self):
        if any(r is None for r in self.rows):
            raise RuntimeError('Incomplete calibration generation')
        atomic_json(os.path.join(self.directory, 'rows.json'), self.rows)
        return Generation(self.directory, self.rows)
