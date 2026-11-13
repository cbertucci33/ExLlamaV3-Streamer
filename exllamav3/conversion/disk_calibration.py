"""Non-mmap bounded file calibration with upstream validation semantics."""
import torch
from ..loader.safetensors_alt import safe_open, _ST_DTYPE_TO_TORCH, _tensor_bytes_view, _stream_readinto


def read_slice(f, key, start, count):
    info = f._infos[key]
    dtype = _ST_DTYPE_TO_TORCH[info.dtype_tag]
    t = torch.empty(count, dtype=dtype)
    f._f.seek(f._data_start + info.begin + start * t.element_size())
    _stream_readinto(f._f, _tensor_bytes_view(t), t.numel() * t.element_size())
    return t


class FileTokens:
    def __init__(self, args, tokenizer):
        self.path = args['cal_data']
        self.columns = args['cal_cols']
        with safe_open(self.path) as f:
            shape = f._infos['input_ids'].shape
            self.rows, self.width = shape
            if self.rows < args['cal_rows']:
                print(f" !! Calibration file contains {self.rows} rows, less than cal_rows ({args['cal_rows']})")
                args['cal_rows'] = self.rows
            self.rows = args['cal_rows']
            total = shape[0] * shape[1]
            dtype = _ST_DTYPE_TO_TORCH[f._infos['input_ids'].dtype_tag]
            size = torch.empty((), dtype=dtype).element_size()
            for start in range(0, total, (8 << 20) // size):
                t = read_slice(f, 'input_ids', start, min((8 << 20) // size, total-start))
                if t.max().item() >= tokenizer.actual_vocab_size:
                    raise ValueError(f"Calibration file {self.path} contains token ids outside the model's vocab; was it produced with a different tokenizer?")

    def __len__(self): return self.rows

    def __iter__(self):
        with safe_open(self.path) as f:
            for i in range(self.rows):
                n = self.columns if 'lengths' not in f.keys() else min(self.columns, int(read_slice(f, 'lengths', i, 1)[0]))
                if n < 1: raise ValueError(f"Calibration file {self.path}: row {i} is empty")
                yield read_slice(f, 'input_ids', i*self.width, min(n, self.width)).to(torch.long).view(1,-1)
