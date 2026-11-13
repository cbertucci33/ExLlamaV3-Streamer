"""Read only an active linear's external output sensitivity vector."""
import os
import torch
from .disk_calibration import read_slice
from ..loader.safetensors_alt import safe_open


class Diagonals:
    def __init__(self, args, linears):
        self.directory = args['hessians']
        self.widths = {l.key: l.out_features for l in linears}

    def get(self, key, default=None):
        path = os.path.join(self.directory, key + '.safetensors')
        if not os.path.isfile(path): return default
        n = self.widths[key]
        with safe_open(path) as f:
            if 'hout_diag' in f.keys():
                diag = f.get_tensor('hout_diag').float()
            elif 'hout' in f.keys():
                info = f._infos['hout']
                diag = torch.empty(n, dtype=torch.float)
                for i in range(n):
                    pos = i*n+i if len(info.shape)==2 else i*n-i*(i-1)//2
                    diag[i] = read_slice(f, 'hout', pos, 1)[0]
            else:
                return default
        assert diag.shape == (n,), key + ': hout diagonal shape'
        return diag
