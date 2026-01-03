#!/usr/bin/env python
import numpy as np
from pyfr.backends import get_backend
from pyfr.inifile import Inifile
from scipy.io import mmread

import argparser
import os
import sys
from pathlib import Path


def benchmark(backend, a_np, N, beta):
    M, K = a_np.shape
    dsize = np.dtype(backend.fpdtype).itemsize

    b_np = np.random.randn(K, N)
    c_np = np.random.randn(M, N)

    a_be = backend.const_matrix(a_np)
    b_be = backend.matrix(b_np.shape, b_np, tags={"align"})
    c_be = backend.matrix(c_np.shape, c_np, tags={"align"})

    kern = backend.kernel("mul", a_be, b_be, c_be, beta=beta)

    fp_dense = 2*M*N*K / kern.dt / 10**9
    fp_sparse = 2*N*np.count_nonzero(a_np) / kern.dt / 10**9
    bw = (M + (M if beta else 0) + K)*N*dsize / kern.dt / 1024**3

    return fp_dense, fp_sparse, bw


def main():
    parser = argparse.ArgumentParser(
        prog="GiMMiK Test",
        description="Test FLOPs and BW of GiMMiK kernels",
    )
    parser.add_argument("-b", "--backend", required=True, type=str)
    parser.add_argument("-p", "--precision", required=True, type=str)
    parser.add_argument("-e", "--elements", required=True, type=str)
    parser.add_argument("-n", type=int, default=64)
    parser.add_argument("dir", required=True, type=Path)

    args = parser.parse_args()

    ORDERS = [2, 3, 4, 5, 6]
    MATS = ["m0", "m3", "m6", "m132", "m460"]

    inistr = f"""
    [backend]
    precision = {args.precision}
    """
    ini = Inifile(inistr)
    backend = get_backend(args.backend, ini)

    eles = args.elements.strip().split(",")

    print("p,ele,mat,GFLOP/s (dense),GFLOP/s (sparse),bw (GiB/s)")
    for p in ORDERS:
        for e in eles:
            for m in MATS:
                mpath = args.dir / f"p{p}" / e / f"{m}-sp.mtx"
                a_np = mmread(mpath).todense()
                beta = 1 if m in ("m3", "m6") else 0

                fp_dense, fp_sparse, bw = benchmark(backend, a_np, args.n, beta)

                print(p, e, m, fp_dense, fp_sparse, bw)


if __name__ == "__main__":
    main()
