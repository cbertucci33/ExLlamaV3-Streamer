"""Converter-owned immutable row boundaries and complete checkpoint publication."""
import json
import os
import random
import threading
import uuid
from .state_store import Pack, Writer, Generation, atomic_json


class Rows:
    def __init__(self, generation):
        self.generation = generation
        self.local = threading.local()
        self.writer = None
        self.known_policy = {}

    def configure(self, modules):
        for module in modules:
            for child in module:
                for path in child.calibration_context_policy(): self.known_policy[path] = False

    def __len__(self):
        return len(self.generation)

    def read_call(self, row_id, params, module):
        descriptors = {}
        hidden, preserve = Pack(self.generation.directory).decode(self.generation.rows[row_id], descriptors=descriptors)
        policy = dict(self.known_policy)
        for child in module:
            policy.update(child.calibration_context_policy())
        def containers(v):
            if isinstance(v, dict): return {k: containers(x) for k, x in v.items()}
            if isinstance(v, tuple): return tuple(containers(x) for x in v)
            if isinstance(v, list): return [containers(x) for x in v]
            return v
        params['_disk_context'] = (containers(preserve), descriptors, policy)
        params.update(preserve)
        params['quant_preserve'] = dict(preserve)
        return hidden

    def begin(self, root):
        self.writer = Writer(os.path.join(root, uuid.uuid4().hex), len(self))

    def write(self, row_id, hidden, params):
        before, descriptors, policy = params.pop('_disk_context')
        after = params['quant_preserve']
        reuse = {}
        def same_values(old, new):
            import torch
            if torch.is_tensor(old) and torch.is_tensor(new):
                if old.shape == new.shape and old.dtype == new.dtype:
                    reuse[id(new)] = descriptors[id(old)]
            elif isinstance(old, dict) and isinstance(new, dict):
                for k in old.keys() & new.keys(): same_values(old[k], new[k])
            elif isinstance(old, (list, tuple)) and isinstance(new, (list, tuple)):
                for a, b in zip(old, new): same_values(a, b)
        def match(old, new, path):
            if not path:
                same_values(old, new)
            elif isinstance(old, dict) and isinstance(new, dict):
                keys = old.keys() & new.keys() if path[0] == '*' else [path[0]]
                for k in keys:
                    if k in old and k in new: match(old[k], new[k], path[1:])
        for path, publishes in policy.items():
            if not publishes: match(before, after, path)
        self.writer.write_row(row_id, hidden, after, reuse=reuse)

    def finish(self, bad_rows, terminal_rows=()):
        for i in set(bad_rows) | set(terminal_rows):
            self.writer.skip_row(i, 'excluded' if i in bad_rows else 'terminal')
        self.generation = self.writer.close()
        self.writer = None


class Tokens:
    def __init__(self, directory, records):
        self.directory, self.records = directory, records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return Pack(self.directory).decode(self.records[index])

    @property
    def num_tokens(self):
        return sum(r[2]['shape'][-1] for r in self.records)


def initialize(root, tokens):
    directory = os.path.join(root, 'tokens')
    pack = Pack(directory)
    records = []
    writer = Writer(os.path.join(root, uuid.uuid4().hex), len(tokens))
    for i, t in enumerate(tokens):
        records.append(pack.encode(t))
        writer.write_row(i, t, {})
    atomic_json(os.path.join(root, 'tokens.json'), records)
    return Rows(writer.close()), Tokens(directory, records)


def checkpoint(root, args, job, rows, inventory, scheduler):
    import torch
    boundary = uuid.uuid4().hex
    directory = os.path.join(root, boundary)
    pack = Pack(directory)
    rng = dict(cpu=torch.get_rng_state(), python=random.getstate(),
               cuda={str(d): torch.cuda.get_rng_state(int(d)) for d in args['devices'].split(',') if d})
    record = pack.encode(rng)
    args = {k: v for k, v in args.items() if k != 'clear_files'}
    payload = dict(args=args, job=job, inventory=inventory, rng=record,
                   scheduler=scheduler,
                   generation=None if rows is None else dict(directory=rows.generation.directory, rows=rows.generation.rows))
    sync_packs(payload)
    token_path = os.path.join(root, 'tokens.json')
    if os.path.isfile(token_path):
        with open(token_path) as f: sync_packs(json.load(f))
    atomic_json(os.path.join(directory, 'boundary.json'), payload)
    atomic_json(os.path.join(root, 'checkpoint.json'), dict(boundary=boundary))
    reclaim(root, payload, directory)


def load_checkpoint(root):
    with open(os.path.join(root, 'checkpoint.json')) as f:
        boundary = json.load(f)['boundary']
    directory = os.path.join(root, boundary)
    with open(os.path.join(directory, 'boundary.json')) as f:
        value = json.load(f)
    value['rng_directory'] = directory
    return value


def restore_rows(root, saved):
    g = saved['generation']
    rows = None if g is None else Rows(Generation(g['directory'], g['rows']))
    if rows is None: return None, None
    with open(os.path.join(root, 'tokens.json')) as f:
        tokens = Tokens(os.path.join(root, 'tokens'), json.load(f))
    return rows, tokens


def restore_rng(saved):
    import torch
    rng = Pack(saved['rng_directory']).decode(saved['rng'])
    torch.set_rng_state(rng['cpu'])
    random.setstate(rng['python'])
    for d, state in rng['cuda'].items():
        torch.cuda.set_rng_state(state, int(d))


class Reference:
    def __init__(self, hidden, directory):
        self.record = Pack(directory).encode(hidden)
        self.directory = directory

    def read(self):
        return Pack(self.directory).decode(self.record)


def reference(rows, hidden):
    if not isinstance(rows, Rows): return hidden.cpu()
    root = os.path.dirname(rows.generation.directory)
    return Reference(hidden, os.path.join(root, 'references', uuid.uuid4().hex))


def reclaim(root, payload, rng_directory):
    # Only closed, task-owned packs. Publication already succeeded; no rollback.
    from pathlib import Path
    live = set()
    def scan(value):
        if isinstance(value, dict):
            if 'file' in value and 'offset' in value:
                live.add(os.path.abspath(os.path.join(value.get('directory', rng_directory), value['file'])))
            for v in value.values(): scan(v)
        elif isinstance(value, (list, tuple)):
            for v in value: scan(v)
    scan(payload)
    tokens = Path(root) / 'tokens.json'
    if tokens.exists():
        with tokens.open() as f: scan(json.load(f))
    for p in Path(root).rglob('*.pack'):
        if str(p.resolve()) not in live: p.unlink()
    generation = payload.get('generation')
    current_rows = None if generation is None else os.path.abspath(generation['directory'])
    for p in Path(root).rglob('rows.json'):
        if str(p.parent.resolve()) != current_rows: p.unlink()
    for p in Path(root).rglob('boundary.json'):
        if str(p.parent.resolve()) != os.path.abspath(rng_directory): p.unlink()
    for p in sorted(Path(root).rglob('*'), key=lambda p: len(p.parts), reverse=True):
        if p.is_dir() and not any(p.iterdir()): p.rmdir()


def file_identity(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            block = f.read(8 << 20)
            if not block: break
            h.update(block)
    return dict(size=os.path.getsize(path), sha256=h.hexdigest())


def source_identity(directory):
    from pathlib import Path
    result = {}
    for p in sorted(Path(directory).iterdir()):
        if not p.is_file(): continue
        if p.suffix in ('.json', '.py'):
            result[p.name] = file_identity(p)
        elif p.suffix == '.safetensors':
            st = p.stat()
            result[p.name] = dict(size=st.st_size, mtime_ns=st.st_mtime_ns)
    return result


def sync_packs(value):
    paths = set()
    def visit(v):
        if isinstance(v, dict):
            if 'file' in v and 'directory' in v and 'offset' in v:
                paths.add(os.path.join(v['directory'], v['file']))
            for x in v.values(): visit(x)
        elif isinstance(v, (list, tuple)):
            for x in v: visit(x)
    visit(value)
    for path in paths:
        with open(path, 'ab') as f:
            f.flush()
            os.fsync(f.fileno())


def external_identity(args):
    from pathlib import Path
    result = {}
    for option in ('hessians', 'ngram_file'):
        value = args.get(option)
        if not value: continue
        path = Path(value)
        paths = sorted(path.glob('*.safetensors')) if path.is_dir() else [path]
        result[option] = {}
        for p in paths:
            st = p.stat()
            result[option][str(p.resolve())] = dict(size=st.st_size, mtime_ns=st.st_mtime_ns)
    return result


def sync_file(path):
    with open(path, 'ab') as f:
        f.flush()
        os.fsync(f.fileno())
