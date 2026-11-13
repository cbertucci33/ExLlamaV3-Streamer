"""Metadata-only safetensors collection and bounded identity-copy output."""
import json
import os
import struct
from dataclasses import dataclass

CHUNK = 8 << 20


@dataclass(frozen=True)
class Entry:
    dtype: str
    shape: tuple
    n_bytes: int
    filename: str = None
    offset: int = 0
    tensor: object = None


def file_entry(stc, key):
    filename = stc.tensor_file_map[key]
    header = stc.file_headers[filename]
    h = header[key]
    start, end = h['data_offsets']
    return Entry(h['dtype'], tuple(h['shape']), end - start,
                 filename, header['_header_offset'] + start)


def tensor_entry(t):
    from ..loader.safetensors_alt import _TORCH_DTYPE_TO_ST
    return Entry(_TORCH_DTYPE_TO_ST[t.dtype], tuple(t.shape),
                 t.numel() * t.element_size(), tensor=t)


class EntryCollection:
    """Finite selection interface for existing compile collectors."""
    def __init__(self, stc):
        self.stc = stc

    def get_tensor(self, key, *args, **kwargs):
        return file_entry(self.stc, key)

    def get_tensors(self, prefix, *args, **kwargs):
        stc = self.stc
        keys = [prefix] if prefix in stc.tensor_file_map else []
        keys += stc.get_tensor_file_map_trie().keys(prefix + '.')
        return {k: file_entry(stc, k) for k in keys}


def entries(module, stc):
    # All pinned overrides only select/delegate; transformed weights already
    # reside in qtensors. Embedding may instead return its compiled dictionary.
    selected = module.get_compile_tensors(EntryCollection(stc))
    return {k: v if isinstance(v, Entry) else tensor_entry(v)
            for k, v in selected.items()}


def _tensor_chunks(t):
    from ..loader.safetensors_alt import _tensor_bytes_view
    if t.numel() * t.element_size() <= CHUNK:
        yield _tensor_bytes_view(t.detach().cpu().contiguous())
    else:
        row_bytes = t[0].numel() * t.element_size()
        if row_bytes <= CHUNK:
            step = max(1, CHUNK // max(row_bytes, 1))
            for start in range(0, t.shape[0], step):
                yield _tensor_bytes_view(t[start:start+step].detach().cpu().contiguous())
        else:
            for part in t:
                yield from _tensor_chunks(part)


def write_entries(selected, filename):
    header, offset = {}, 0
    for key, e in selected.items():
        header[key] = dict(dtype=e.dtype, shape=list(e.shape),
                           data_offsets=[offset, offset + e.n_bytes])
        offset += e.n_bytes
    raw = json.dumps(header, separators=(',', ':')).encode('utf8')
    raw += b' ' * (-len(raw) % 8)
    temporary = filename + '.partial'
    buffer = bytearray(CHUNK)
    with open(temporary, 'wb') as out:
        out.write(struct.pack('<Q', len(raw)))
        out.write(raw)
        for e in selected.values():
            if e.filename is not None:
                with open(e.filename, 'rb') as src:
                    src.seek(e.offset)
                    remaining = e.n_bytes
                    while remaining:
                        got = src.readinto(memoryview(buffer)[:min(CHUNK, remaining)])
                        if not got:
                            raise EOFError(e.filename)
                        out.write(memoryview(buffer)[:got])
                        remaining -= got
            else:
                for chunk in _tensor_chunks(e.tensor):
                    out.write(chunk)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, filename)
    return offset
