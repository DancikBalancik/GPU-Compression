#!/usr/bin/env python3
"""
Streaming GPU-Accelerated Zstd Compression/Decompression
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
# CUDA kernels for preprocessing/postprocessing
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

cuda_preprocess_delta = """
extern "C" __global__
void preprocess_data_delta(const unsigned char* input,
                         unsigned char* output,
                         int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    if (idx == 0) {
        output[0] = input[0];
    }
    for (int i = idx + 1; i < size; i += stride) {
        output[i] = input[i] - input[i-1];
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

def get_dir_size(path):
    """Calculate total size of all files in a directory"""
    total_size = 0
    for root, _, files in os.walk(path):
        total_size += sum(os.path.getsize(os.path.join(root, name)) for name in files)
    return total_size

class GPUZstdStream:
    def __init__(self, level=3, preprocess='basic', xor_key=42, gpu_id=0):
        cp.cuda.Device(gpu_id).use()
        self.pre = preprocess
        self.key = xor_key
        self.block = 256
        self.stream = cp.cuda.Stream.null
        self._compile_kernels()
        self.cctx = zstd.ZstdCompressor(level=level)
        self.dctx = zstd.ZstdDecompressor()

    def _compile_kernels(self):
        self.k_basic    = cp.RawKernel(cuda_preprocess_basic, "preprocess_data_basic")
        self.k_xor      = cp.RawKernel(cuda_preprocess_xor,   "preprocess_data_xor")
        self.k_delta    = cp.RawKernel(cuda_preprocess_delta, "preprocess_data_delta")
        self.k_delta_po = cp.RawKernel(cuda_postprocess_delta, "postprocess_data_delta")

    def _gpu_pre(self, buf: bytes) -> bytes:
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
            self.k_delta(grid, (threads,), (d_in, d_out, np.int32(n)), stream=self.stream)
        else:
            self.k_basic(grid, (threads,), (d_in, d_out, np.int32(n)), stream=self.stream)
        self.stream.synchronize()
        return cp.asnumpy(d_out).tobytes()

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

    def tar_stream(self, dir_path, chunk_size=4*1024*1024):
        """Generator that yields chunks from tar archive of directory"""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode='w|') as tar:
            for root, _, files in os.walk(dir_path):
                for fn in files:
                    path = os.path.join(root, fn)
                    arc  = os.path.relpath(path, dir_path)
                    ti   = tarfile.TarInfo(arc)
                    ti.size = os.path.getsize(path)
                    ti.mtime = int(os.path.getmtime(path))
                    buf.seek(0); buf.truncate(0)
                    tar.addfile(ti, open(path, 'rb'))
                    yield buf.getvalue()
            buf.seek(0); buf.truncate(0)
        yield b''

    def compress(self, inp, out, block_size):
        out_f = open(out, 'wb')
        comp = self.cctx.compressobj()
        
        # Determine total size for progress bar
        if os.path.isdir(inp):
            total_size = get_dir_size(inp)
            src = self.tar_stream(inp, block_size)
            desc = f"Compressing directory: {inp}"
        else:
            total_size = os.path.getsize(inp)
            src = open(inp, 'rb')
            desc = f"Compressing file: {inp}"
        
        # Setup progress bar
        pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=desc)
        processed = 0
        
        # Process chunks with progress updates
        if os.path.isdir(inp):
            for chunk in src:
                if not chunk: continue
                chunk_size = len(chunk)
                pre = self._gpu_pre(chunk)
                c = comp.compress(pre)
                if c: out_f.write(c)
                processed += chunk_size
                pbar.update(chunk_size)
        else:
            while True:
                chunk = src.read(block_size)
                if not chunk: break
                chunk_size = len(chunk)
                pre = self._gpu_pre(chunk)
                c = comp.compress(pre)
                if c: out_f.write(c)
                processed += chunk_size
                pbar.update(chunk_size)
            src.close()
            
        tail = comp.flush()
        if tail: out_f.write(tail)
        out_f.close()
        pbar.close()
        print(f"✅ Compressed → {out}")

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
        description="Streaming GPU Zstd (on-the-fly tar for dirs) with progress bars"
    )
    sub = parser.add_subparsers(dest='cmd')

    pc = sub.add_parser('compress', help='Compress file or dir')
    pc.add_argument('input')
    pc.add_argument('output')
    pc.add_argument('-b','--block-size',type=int,default=4*1024*1024)

    pd = sub.add_parser('decompress', help='Decompress file or dir')
    pd.add_argument('input')
    pd.add_argument('output')
    pd.add_argument('-b','--block-size',type=int,default=4*1024*1024)

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return 1

    z = GPUZstdStream(level=3, preprocess='basic', xor_key=42, gpu_id=0)
    t0 = time.time()
    if args.cmd == 'compress':
        z.compress(args.input, args.output, args.block_size)
    else:
        z.decompress(args.input, args.output, args.block_size)
    print(f"⏱️  Done in {time.time()-t0:.2f}s")

if __name__=='__main__':
    sys.exit(main())