# GPU-Accelerated Zstd Decompressor & GUI

A versatile, high-performance tool for **streaming** Zstd decompression using GPU acceleration, available as both a command-line interface (CLI) and a graphical user interface (GUI).

---

## 🚀 Features

- **GPU-Accelerated Decompression** : Leverages NVIDIA GPUs via CuPy and custom CUDA kernels for fast postprocessing.
- **Streaming Tar Extraction** : On-the-fly `tar`-stream extraction without materializing the full archive in memory or on disk.
- **Dual Mode** :
- **CLI** : Scriptable via command line for batch workflows and automation.
- **GUI** : Intuitive Tkinter-based interface for users who prefer a desktop app.
- **Progress Reporting** :
- CLI shows elapsed time and per-file progress.
- GUI updates status per file as it extracts.
- **Flexible Preprocessing** : Supports `basic`, `xor`, and `delta` modes (must match the compressor).

---

## 📋 Requirements

- **Python** ≥ 3.7
- **Dependencies** :
- `cupy` (for CUDA kernels)
- `zstandard` (for Zstd streaming API)
- `numpy`
- `tqdm` (CLI progress bars)
- `tkinter` (GUI, included with standard Python installs)

Install via pip:

```bash
pip install cupy zstandard numpy tqdm
```

_(Ensure you have a compatible CUDA toolkit installed for CuPy.)_

---

## ⚙️ Installation

1. **Clone or download** this repository.
2. **Make the script executable** (Unix/macOS):
   ```bash
   chmod +x setup.py
   ```
3. **Verify** GPU drivers and CUDA toolkit are installed and CuPy works:
   ```bash
   python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
   ```

---

## 💻 Usage

### Command-Line (CLI)

```bash
# Decompress using default settings
python setup.py input.zst output_directory

# With options:
python setup.py input.zst output_directory \
  --block-size 4194304 \
  --preprocess delta \
  --xor-key 123 \
  --gpu-id 1
```

**Options** :

- `-b, --block-size`: I/O chunk size (default: 4 MiB)
- `-p, --preprocess`: `basic` | `xor` | `delta` (must match compressor)
- `-k, --xor-key`: Key byte for `xor` mode (default: 42)
- `-g, --gpu-id`: GPU device index (default: 0)

### Graphical Interface (GUI)

Simply run without arguments:

```bash
python setup.py
```

- Browse for the `.zst` file and choose an output folder.
- Click **Decompress** and monitor status updates.

---

## 🔧 Configuration

- The default decompression level and GPU settings can be overridden via CLI flags or modified directly in `setup.py` if embedding into a larger application.

---

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`.
3. Commit your changes: `git commit -m "Add new feature"`.
4. Push to your branch: `git push origin feature/my-feature`.
5. Submit a pull request.

Please follow PEP 8 style and include tests when adding functionality.

---

## 📄 License

This project is licensed under the **MIT License** . See the [LICENSE](LICENCE) file for details.
