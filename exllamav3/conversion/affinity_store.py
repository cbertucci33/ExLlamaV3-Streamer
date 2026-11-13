"""Disk sample records, preserving worker/append order and global match chunks."""
import os
import torch
from .state_store import Pack


class Samples:
    def __init__(self, directory):
        self.directory = directory
        self.records = []
        self.count = 0
        self.width = None
        self.pack = Pack(directory)

    def append(self, x, sel, w):
        self.width = x.shape[1]
        self.count += x.shape[0]
        self.records.append((self.directory, self.pack.encode((x, sel, w))))

    def extend(self, other):
        self.records.extend(other.records)
        self.count += other.count
        self.width = self.width or other.width

    def matching(self, expert, chunk, device):
        # Carry the partial selected-row chunk over all sample/worker boundaries.
        xs = ws = hits = None
        used = 0
        for directory, record in self.records:
            xd, sd, wd = [r[2] for r in record[2]]
            pack = Pack(directory)
            for start in range(0, xd['shape'][0], chunk):
                count = min(chunk, xd['shape'][0]-start)
                x, s, w = [pack.get_rows(d, start, count) for d in (xd,sd,wd)]
                hit = s == expert
                ri = hit.any(dim=1).nonzero().flatten()
                xc = x.index_select(0, ri)
                wc = w.index_select(0, ri)
                hc = hit.index_select(0, ri)
                if xs is None:
                    xs = torch.empty((chunk, x.shape[1]), dtype=x.dtype)
                    ws = torch.empty((chunk, w.shape[1]), dtype=w.dtype)
                    hits = torch.empty((chunk, s.shape[1]), dtype=torch.bool)
                pos = 0
                while pos < ri.numel():
                    take = min(chunk-used, ri.numel()-pos)
                    xs[used:used+take].copy_(xc[pos:pos+take])
                    ws[used:used+take].copy_(wc[pos:pos+take])
                    hits[used:used+take].copy_(hc[pos:pos+take])
                    used += take; pos += take
                    if used == chunk:
                        yield xs.to(device).float(), ws.to(device).float(), hits.to(device)
                        used = 0
        if used:
            yield xs[:used].to(device).float(), ws[:used].to(device).float(), hits[:used].to(device)


def append_sample(aff, key, st, x, sel, w):
    if 'disk_directory' in aff:
        if 'samples' not in st:
            st['samples'] = Samples(os.path.join(aff['disk_directory'], str(len(aff['layers']))))
        st['samples'].append(x, sel, w)
    else:
        st['x'].append(x)
        st['sel'].append(sel)
        st['w'].append(w)
