#!/usr/bin/env python3
"""
GPU-Accelerated Zstd Decompression with GUI
"""
import os
import io
import sys
import argparse
import tarfile
import time
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import numpy as np
try:
    import cupy as cp
    import zstandard as zstd
    from tqdm import tqdm
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False

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
    def __init__(self, preprocess='basic', xor_key=42, gpu_id=0, callback=None):
        cp.cuda.Device(gpu_id).use()
        self.pre = preprocess
        self.key = xor_key
        self.block = 256
        self.stream = cp.cuda.Stream.null
        self.callback = callback
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
                if self.gpu.callback:
                    self.gpu.callback(len(data))
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
                return True, "Directory extracted"
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
        return True, f"File decompressed → {dst}"


class GPUUnzstdGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GPU Zstd Decompressor")
        self.root.geometry("600x600")
        self.root.resizable(True, True)
        
        # Set icon if available
        try:
            self.root.iconbitmap("app_icon.ico")
        except:
            pass
            
        self.setup_ui()
        
    def scan_zst_files(self):
        """Scan for .zst files in the current directory"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        zst_files = []
        
        # Find all .zst files
        for file in os.listdir(current_dir):
            if file.lower().endswith('.zst'):
                zst_files.append(os.path.join(current_dir, file))
                
        return zst_files
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20 20 20 20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # App title and description
        ttk.Label(
            main_frame, 
            text="GPU-Accelerated Zstd Decompressor", 
            font=("Helvetica", 16, "bold")
        ).pack(pady=(0, 10))
        
        ttk.Label(
            main_frame, 
            text="Extract .zst files using GPU acceleration",
            font=("Helvetica", 10)
        ).pack(pady=(0, 20))
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="10 10 10 10")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Input file selection
        input_frame = ttk.Frame(file_frame)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(input_frame, text="Input .zst file:").pack(side=tk.LEFT, padx=(0, 10))
        self.input_path = tk.StringVar()
        
        # Get list of .zst files in current directory
        zst_files = self.scan_zst_files()
        
        # Create dropdown if files were found
        if zst_files:
            self.input_dropdown = ttk.Combobox(input_frame, textvariable=self.input_path, values=zst_files)
            self.input_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True)
            if zst_files:  # Set first file as default
                self.input_path.set(zst_files[0])
        else:
            # Fallback to text entry if no files found
            ttk.Entry(input_frame, textvariable=self.input_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
            
        ttk.Button(input_frame, text="Browse...", command=self.browse_input).pack(side=tk.LEFT, padx=(10, 0))
        
        # Refresh button
        ttk.Button(input_frame, text="↻", width=3, command=self.refresh_file_list).pack(side=tk.LEFT, padx=(5, 0))
        
        # Auto-generate output dir checkbox
        self.auto_output = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            file_frame, 
            text="Auto-create output folder with same name as .zst file", 
            variable=self.auto_output,
            command=self.toggle_output_field
        ).pack(anchor=tk.W, pady=(5, 5))
        
        # Output directory selection (hidden by default with auto-output enabled)
        self.output_frame = ttk.Frame(file_frame)
        
        ttk.Label(self.output_frame, text="Output folder:").pack(side=tk.LEFT, padx=(0, 10))
        self.output_path = tk.StringVar()
        ttk.Entry(self.output_frame, textvariable=self.output_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(self.output_frame, text="Browse...", command=self.browse_output).pack(side=tk.LEFT, padx=(10, 0))
        
        # Advanced options frame
        adv_frame = ttk.LabelFrame(main_frame, text="Advanced Options", padding="10 10 10 10")
        adv_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Preprocessing algorithm
        pre_frame = ttk.Frame(adv_frame)
        pre_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(pre_frame, text="Preprocessing:").pack(side=tk.LEFT, padx=(0, 10))
        self.preprocess = tk.StringVar(value="basic")
        ttk.Combobox(pre_frame, textvariable=self.preprocess, values=["basic", "xor", "delta"], width=10).pack(side=tk.LEFT)
        
        # XOR key setting
        xor_frame = ttk.Frame(adv_frame)
        xor_frame.pack(fill=tk.X)
        
        ttk.Label(xor_frame, text="XOR Key:").pack(side=tk.LEFT, padx=(0, 10))
        self.xor_key = tk.IntVar(value=42)
        ttk.Spinbox(xor_frame, from_=0, to=255, textvariable=self.xor_key, width=5).pack(side=tk.LEFT)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(main_frame, text="Decompression Progress", padding="10 10 10 10")
        progress_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.progress = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress.pack(fill=tk.X, pady=(0, 10))
        
        self.status_text = tk.StringVar(value="Ready to decompress")
        ttk.Label(progress_frame, textvariable=self.status_text).pack(fill=tk.X)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        self.decompress_btn = ttk.Button(button_frame, text="Decompress", command=self.start_decompression)
        self.decompress_btn.pack(side=tk.RIGHT)
        
        ttk.Button(button_frame, text="Exit", command=self.root.destroy).pack(side=tk.RIGHT, padx=(0, 10))
        
        # Check for dependencies
        if not HAS_DEPENDENCIES:
            self.show_dependency_warning()
            
    def show_dependency_warning(self):
        messagebox.showwarning(
            "Missing Dependencies", 
            "Some required packages are missing. Please install:\n" +
            "- cupy (for GPU support)\n" +
            "- zstandard (for zstd decompression)\n" +
            "- tqdm (for progress tracking)\n\n" +
            "You can install them using pip:\n" +
            "pip install cupy zstandard tqdm"
        )
        
    def refresh_file_list(self):
        """Refresh the list of .zst files in the dropdown"""
        zst_files = self.scan_zst_files()
        
        # Update dropdown values
        if hasattr(self, 'input_dropdown'):
            self.input_dropdown['values'] = zst_files
            if zst_files and not self.input_path.get() in zst_files:
                self.input_path.set(zst_files[0])
            
        # Show a message if no files found
        if not zst_files:
            messagebox.showinfo("No Files Found", "No .zst files found in the current directory.")
    
    def toggle_output_field(self):
        """Show or hide the output directory field based on auto-output checkbox"""
        if self.auto_output.get():
            self.output_frame.pack_forget()
        else:
            self.output_frame.pack(fill=tk.X)
    
    def browse_input(self):
        file_path = filedialog.askopenfilename(
            title="Select Compressed File",
            filetypes=[("Zstd Files", "*.zst"), ("All Files", "*.*")]
        )
        if file_path:
            self.input_path.set(file_path)
    
    def browse_output(self):
        dir_path = filedialog.askdirectory(title="Select Output Directory")
        if dir_path:
            self.output_path.set(dir_path)
    
    def get_auto_output_dir(self, input_file):
        """Generate output directory based on input filename"""
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        return os.path.join(os.path.dirname(input_file), base_name)
    
    def update_progress(self, bytes_read):
        self.processed_bytes += bytes_read
        progress_pct = min(100, int(100 * self.processed_bytes / self.total_size))
        self.root.after(0, lambda: self.progress.configure(value=progress_pct))
        
    def start_decompression(self):
        if not HAS_DEPENDENCIES:
            self.show_dependency_warning()
            return
            
        input_file = self.input_path.get().strip()
        
        # Validate input file
        if not input_file:
            messagebox.showerror("Error", "Please select an input file to decompress")
            return
            
        if not os.path.exists(input_file):
            messagebox.showerror("Error", "Input file does not exist")
            return
        
        # Determine output directory
        if self.auto_output.get():
            output_dir = self.get_auto_output_dir(input_file)
        else:
            output_dir = self.output_path.get().strip()
            if not output_dir:
                messagebox.showerror("Error", "Please select an output directory")
                return
            
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up progress tracking
        self.total_size = os.path.getsize(input_file)
        self.processed_bytes = 0
        self.progress.configure(value=0, maximum=100)
        
        # Disable UI during decompression
        self.decompress_btn.configure(state="disabled")
        self.status_text.set(f"Decompressing to: {os.path.basename(output_dir)}...")
        
        # Start decompression in a separate thread
        threading.Thread(
            target=self.run_decompression, 
            args=(input_file, output_dir, self.preprocess.get(), self.xor_key.get()), 
            daemon=True
        ).start()
    
    def run_decompression(self, input_file, output_dir, preprocess, xor_key):
        try:
            # Create decompressor
            z = GPUZstdDecompressor(
                preprocess=preprocess,
                xor_key=xor_key,
                gpu_id=0,
                callback=self.update_progress
            )
            
            # Run decompression
            success, message = z.decompress(input_file, output_dir, 4*1024*1024)
            
            # Update UI
            if success:
                self.root.after(0, lambda: self.decompression_complete(True, message))
            else:
                self.root.after(0, lambda: self.decompression_complete(False, message))
                
        except Exception as e:
            self.root.after(0, lambda: self.decompression_complete(False, str(e)))
    
    def decompression_complete(self, success, message):
        # Update UI
        self.decompress_btn.configure(state="normal")
        
        if success:
            self.status_text.set("Decompression complete")
            self.progress.configure(value=100)
            messagebox.showinfo("Success", message)
            
            # Ask if user wants to open the output folder
            if messagebox.askyesno("Open Folder", "Do you want to open the output folder?"):
                self.open_output_folder()
        else:
            self.status_text.set("Decompression failed")
            messagebox.showerror("Error", f"Decompression failed:\n{message}")
    
    def open_output_folder(self):
        output_dir = self.output_path.get().strip()
        if os.path.exists(output_dir):
            # Open file explorer with the output folder
            if sys.platform == 'win32':
                os.startfile(output_dir)
            elif sys.platform == 'darwin':  # macOS
                subprocess.run(['open', output_dir])
            else:  # Linux
                subprocess.run(['xdg-open', output_dir])


def cli_main():
    parser = argparse.ArgumentParser(
        description="GPU-Accelerated Zstd Decompression with progress bar"
    )
    parser.add_argument('input', nargs='?', help='Input file to decompress (optional)')
    parser.add_argument('-a', '--auto', action='store_true', 
                       help='Auto-process all .zst files in current directory')
    parser.add_argument('-b', '--block-size', type=int, default=4*1024*1024, 
                       help='Block size for reading (default: 4MB)')
    parser.add_argument('-p', '--preprocess', choices=['basic', 'xor', 'delta'], default='basic',
                       help='Preprocessing algorithm (must match compression)')
    parser.add_argument('-k', '--xor-key', type=int, default=42,
                       help='XOR key if using xor preprocessing (default: 42)')
    parser.add_argument('-g', '--gpu-id', type=int, default=0,
                       help='GPU device ID to use (default: 0)')
    parser.add_argument('--gui', action='store_true', help='Launch GUI mode')

    args = parser.parse_args()
    
    # Check if GUI mode was requested
    if args.gui or len(sys.argv) == 1:
        root = tk.Tk()
        app = GPUUnzstdGUI(root)
        root.mainloop()
        return 0
        
    # Check for required dependencies
    if not HAS_DEPENDENCIES:
        print("Error: Missing required dependencies. Please install cupy, zstandard, and tqdm.")
        return 1

    # Create decompressor with specified options
    z = GPUZstdDecompressor(
        preprocess=args.preprocess,
        xor_key=args.xor_key,
        gpu_id=args.gpu_id
    )
    
    # Auto-process all .zst files in current directory
    if args.auto:
        current_dir = os.path.abspath(os.path.dirname(__file__))
        zst_files = [f for f in os.listdir(current_dir) if f.lower().endswith('.zst')]
        
        if not zst_files:
            print("No .zst files found in the current directory.")
            return 0
            
        print(f"Found {len(zst_files)} .zst files to process.")
        
        for file in zst_files:
            input_path = os.path.join(current_dir, file)
            output_dir = os.path.join(current_dir, os.path.splitext(file)[0])
            
            print(f"\nProcessing: {file}")
            t0 = time.time()
            z.decompress(input_path, output_dir, args.block_size)
            print(f"⏱️  Done in {time.time()-t0:.2f}s")
            
        return 0
    
    # Process single file
    if not args.input:
        print("Error: Please specify an input file or use --auto to process all .zst files")
        return 1
        
    input_path = args.input
    output_dir = os.path.join(
        os.path.dirname(input_path), 
        os.path.splitext(os.path.basename(input_path))[0]
    )
    
    t0 = time.time()
    z.decompress(input_path, output_dir, args.block_size)
    print(f"⏱️  Done in {time.time()-t0:.2f}s")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments, launch GUI
        root = tk.Tk()
        app = GPUUnzstdGUI(root)
        root.mainloop()
    else:
        # Command line arguments provided, run CLI mode
        sys.exit(cli_main())