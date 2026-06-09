# Weight Atlas

> *Human said, AI made.* — Davut Engin

A local AI model explorer that runs entirely on your machine. Open any `.safetensors`, `.gguf` or MLX model file and instantly browse its weights, tensors, and metadata — no cloud, no telemetry, no account.

![Weight Atlas](https://img.shields.io/badge/format-SafeTensors%20%7C%20GGUF%20%7C%20MLX-6366f1) ![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

![Weight Atlas home screen](docs/screenshots/main.png)

---

## What it does

- **Overview** — model name, format, parameter count, file size, quantization type, architecture
- **Atlas** — interactive treemap showing how parameters are distributed across layers. Click to drill in, navigate with the tree on the left
- **Tensors** — sortable/searchable table of every tensor. Click a row to view the actual values as a heatmap or bar chart, or browse them in an Excel-like grid
- **Metadata** — all embedded key-value metadata from the model file

Supported formats: `SafeTensors`, `GGUF` (including quantized: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, BF16…), `MLX`

---

## Screenshots

**Atlas — parameter distribution treemap**
![Atlas view](docs/screenshots/atlas.png)

**Tensors — full tensor list with search and filter**
![Tensor explorer](docs/screenshots/tensors.png)

**Weight viewer — heatmap (2D tensors)**
![Weight heatmap](docs/screenshots/weight_chart.png)

**Weight viewer — bar chart (1D tensors)**
![Weight bar chart](docs/screenshots/weight_chart_2.png)

**Weight viewer — table / Excel mode**
![Weight table](docs/screenshots/weight_table.png)

---

## Installation

### Requirements

- Python 3.11 or newer
- Node.js 18 or newer (only needed to build the frontend once)

---

### Windows

```bat
git clone https://github.com/davutengin/weight-atlas.git
cd weight-atlas
setup.bat
```

`setup.bat` installs Python dependencies and builds the frontend in one step.

Then to run:

```bat
python weightatlas.py
```

The app opens at `http://localhost:8000` in your browser.

---

### macOS / Linux

```bash
git clone https://github.com/davutengin/weight-atlas.git
cd weight-atlas

# Install Python dependencies
pip install -r backend/requirements.txt

# Build the frontend
cd frontend
npm install
npm run build
cd ..

# Run
python weightatlas.py
```

The app opens at `http://localhost:8000` in your browser.

---

### Using a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r backend/requirements.txt
```

---

## Usage

1. Run `python weightatlas.py`
2. Click **Open Model** and pick a `.gguf` or `.safetensors` file (or an MLX model folder)
3. Explore

To close a model and open another, click the **✕** button next to the model name in the top bar.

---

## License

MIT
