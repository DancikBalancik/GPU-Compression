#!/usr/bin/env python3
"""
Standalone GPU-Accelerated Zstd Decompression
"""
import os
import io
import sys
import argparse
import tarfile
import time
import numpy as np
import cupy as cp
import zstandard as zstd
from pathlib import Path
from tqdm import tqdm

# ------------------------------------------------------------
# CUDA kernels for postprocessing
cuda_preprocess_basic = """
extern "C" __global__
void preprocess_data_basic(const unsigned char* input,
                         unsigned char* output,
                         int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < size; i += stride) {
        output[i] = input[i];
    }
}
"""

cuda_preprocess_xor = """
extern "C" __global__
void preprocess_data_xor(const unsigned char* input,
                       unsigned char* output,
                       int size,
                       unsigned char key) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < size; i += stride) {
        output[i] = input[i] ^ key;
    }
}
"""

cuda_postprocess_delta = """
extern "C" __global__
void postprocess_data_delta(const unsigned char* input,
                            unsigned char* output,
                            int size) {
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    if (idx == 0) {
        output[0] = input[0];
    }
    for (int i = idx + 1; i < size; i += stride) {
        output[i] = input[i] + output[i-1];
    }
}
"""
# ------------------------------------------------------------

class GPUZstdDecompressor:
    def __init__(self, preprocess='basic', xor_key=42, gpu_id=0):
        cp.cuda.Device(gpu_id).use()
        self.pre = preprocess
        self.key = xor_key
        self.block = 256
        self.stream = cp.cuda.Stream.null
        self._compile_kernels()
        self.dctx = zstd.ZstdDecompressor()

    def _compile_kernels(self):
        self.k_basic    = cp.RawKernel(cuda_preprocess_basic, "preprocess_data_basic")
        self.k_xor      = cp.RawKernel(cuda_preprocess_xor,   "preprocess_data_xor")
        self.k_delta_po = cp.RawKernel(cuda_postprocess_delta, "postprocess_data_delta")

    def _gpu_post(self, buf: bytes) -> bytes:
        arr = np.frombuffer(buf, np.uint8)
        d_in  = cp.asarray(arr)
        d_out = cp.empty_like(d_in)
        n     = arr.size
        threads = self.block
        needed  = (n + threads - 1) // threads
        maxb    = cp.cuda.runtime.getDeviceProperties(0)['maxGridSize'][0]
        gx = min(needed, maxb); gy = (needed + maxb - 1) // maxb
        grid = (gx, gy)
        if self.pre == 'xor':
            self.k_xor(grid, (threads,), (d_in, d_out, np.int32(n), np.uint8(self.key)), stream=self.stream)
        elif self.pre == 'delta':
            self.k_delta_po(grid, (threads,), (d_in, d_out, np.int32(n)), stream=self.stream)
        else:
            self.k_basic(grid, (threads,), (d_in, d_out, np.int32(n)), stream=self.stream)
        self.stream.synchronize()
        return cp.asnumpy(d_out).tobytes()

    class _DecompressReader(io.RawIOBase):
        def __init__(self, path, dctx, gpu, block_size, pbar=None):
            self.fin = open(path, 'rb')
            self.reader = dctx.stream_reader(self.fin)
            self.gpu = gpu
            self.bs = block_size
            self.pbar = pbar
            self.total_read = 0
            
        def readable(self):
            return True

        def read(self, n=-1):
            data = self.reader.read(self.bs if n < 0 else n)
            if not data:
                return b''
            result = self.gpu._gpu_post(data)
            if self.pbar is not None:
                self.total_read += len(data)
                self.pbar.update(len(data))
            return result

        def readinto(self, b):
            data = self.read(len(b))
            n = len(data)
            b[:n] = data
            return n

        def close(self):
            self.reader.close()
            self.fin.close()
            super().close()

    def decompress(self, inp, out, block_size):
        # Get compressed file size for progress tracking
        compressed_size = os.path.getsize(inp)
        pbar = tqdm(total=compressed_size, unit='B', unit_scale=True, desc=f"Decompressing: {inp}")
        
        rdr = self._DecompressReader(inp, self.dctx, self, block_size, pbar)
        try:
            with tarfile.open(fileobj=io.BufferedReader(rdr), mode='r|*') as tar:
                tar.extractall(path=out)
                print(f"✅ Directory extracted → {out}")
                rdr.close()
                pbar.close()
                return
        except tarfile.ReadError:
            pass
            
        os.makedirs(out, exist_ok=True)
        data = rdr.read(-1)
        base = os.path.splitext(os.path.basename(inp))[0]
        dst  = os.path.join(out, base)
        with open(dst, 'wb') as f:
            f.write(data)
        print(f"✅ File decompressed → {dst}")
        rdr.close()
        pbar.close()

# ----------------------
# CLI
# ----------------------
def main():
    parser = argparse.ArgumentParser(
        description="GPU-Accelerated Zstd Decompression with progress bar"
    )
    parser.add_argument('input', help='Input file to decompress')
    parser.add_argument('output', help='Output directory')
    parser.add_argument('-b', '--block-size', type=int, default=4*1024*1024, 
                       help='Block size for reading (default: 4MB)')
    parser.add_argument('-p', '--preprocess', choices=['basic', 'xor', 'delta'], default='basic',
                       help='Preprocessing algorithm (must match compression)')
    parser.add_argument('-k', '--xor-key', type=int, default=42,
                       help='XOR key if using xor preprocessing (default: 42)')
    parser.add_argument('-g', '--gpu-id', type=int, default=0,
                       help='GPU device ID to use (default: 0)')

    args = parser.parse_args()

    # Create decompressor with specified options
    z = GPUZstdDecompressor(
        preprocess=args.preprocess,
        xor_key=args.xor_key,
        gpu_id=args.gpu_id
    )
    
    t0 = time.time()
    z.decompress(args.input, args.output, args.block_size)
    print(f"⏱️  Done in {time.time()-t0:.2f}s")

if __name__ == '__main__':
    sys.exit(main())