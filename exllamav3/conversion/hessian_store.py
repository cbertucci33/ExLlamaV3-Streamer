"""Per-qmap shared mutable ownership, bounded to active quantization groups."""
from collections.abc import Mapping
from contextlib import contextmanager
import os
import threading
from .state_store import Pack


class Hessians(Mapping):
    def __init__(self, directory):
        self.directory = directory
        self.records = {}
        self.meta = {}
        self.owners = {}
        self.remaining = {}
        self.local = threading.local()

    def __len__(self): return len(self.records)
    def __iter__(self): return iter(self.records)
    def __contains__(self, key): return key in self.records

    def __getitem__(self, key):
        owned = getattr(self.local, 'owned', {})
        if key not in owned:
            raise RuntimeError('Hessian read without active group lease: ' + str(key))
        return owned[key]

    def items(self): return self.meta.items()

    def add(self, values):
        for key, value in values.items():
            # A raw spill belongs to one qmap. A replaced/final record can then
            # retire its closed files without invalidating another qmap.
            self.records[key] = Pack(self.directory).encode(value)
            self.meta[key] = {k: value[k].detach().cpu().clone() if hasattr(value[k], 'detach') else value[k]
                              for k in ('num_total', 'inf_nan')}
            self.owners[key] = dict(condition=threading.Condition(), active=0, state='parked', value=None,
                                    files=_record_files(self.records[key]))

    def plan(self, groups):
        self.remaining = {}
        for group in groups:
            for key in {l.qmap for l in group if l.qmap in self}:
                self.remaining[key] = self.remaining.get(key, 0) + 1

    def acquire(self, key):
        o = self.owners[key]
        c = o['condition']
        with c:
            while o['state'] in ('loading', 'evicting'): c.wait()
            if o['state'] == 'failed': raise RuntimeError('Failed Hessian owner: ' + str(key))
            if o['state'] == 'resident':
                o['active'] += 1
                return o['value']
            o['state'] = 'loading'
        try:
            value = Pack(self.directory).decode(self.records[key], restore_device=True)
        except BaseException:
            with c:
                o['state'] = 'parked'; c.notify_all()
            raise
        with c:
            o.update(value=value, active=1, state='resident')
            c.notify_all()
            return value

    def release(self, key, completed):
        o = self.owners[key]; c = o['condition']
        with c:
            if completed and key in self.remaining:
                self.remaining[key] -= 1
            o['active'] -= 1
            if o['active']: return
            o['state'] = 'evicting'
            value = o.pop('value')
            old_files = o['files']
        try:
            finished = key in self.remaining and self.remaining[key] == 0
            record = None if finished else Pack(self.directory).encode(value)
            del value
            new_files = set() if record is None else _record_files(record)
            with c:
                if finished:
                    del self.records[key]
                    self.meta.pop(key, None)
                else:
                    self.records[key] = record
                o['files'] = new_files
            # No lease/reader remains. New pack writes have closed, and waiters
            # cannot acquire while this qmap is still marked evicting.
            for path in old_files - new_files:
                os.unlink(path)
            with c:
                o.update(value=None, state='parked')
                c.notify_all()
        except BaseException:
            with c:
                o.update(value=None, state='failed'); c.notify_all()
            raise

    @contextmanager
    def lease(self, keys, completed=True):
        keys = list(dict.fromkeys(k for k in keys if k in self))
        previous = getattr(self.local, 'owned', {})
        owned = dict(previous)
        acquired = []
        ok = False
        try:
            for key in keys:
                if key not in owned:
                    owned[key] = self.acquire(key); acquired.append(key)
            self.local.owned = owned
            yield
            ok = True
        finally:
            self.local.owned = previous
            owned.clear()
            for key in acquired:
                self.release(key, completed and ok)


def group_lease(store, group):
    from contextlib import nullcontext
    return store.lease([l.qmap for l in group]) if isinstance(store, Hessians) else nullcontext()


def _record_files(record):
    files = set()
    pending = [record]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if 'directory' in value and 'file' in value and 'offset' in value:
                files.add(os.path.join(value['directory'], value['file']))
            pending.extend(value.values())
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
    return files
