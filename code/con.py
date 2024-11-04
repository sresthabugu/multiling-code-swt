#!/usr/bin/env python3

from dataclasses import dataclass

@dataclass
class CFG:
    max_sentence_len: int = 50
    batch_size: int = 64
    seed: int = 3407 
    n_epochs: int = 100
    lr: float = 1e-3 
    wd: float = 1e-5 
    pad_length: int = 20 
    out_ch1: int = 3*7
    out_ch2: int = 3*5 
