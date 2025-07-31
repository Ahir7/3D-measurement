# 3D Measurement System

A robust system for calculating 3D measurements from multiple images using DUSt3R (Deep Unsupervised Structure-from-motion 3D Reconstruction).

## 🌟 Features

- 3D reconstruction from multiple images
- Automatic dimension calculation
- EXIF metadata extraction
- GPU acceleration support with CPU fallback
- JSON-safe data handling
- Support for truncated JPEG images

## 🔧 Prerequisites

- Python 3.x
- CUDA-capable GPU (recommended) or CPU
- Required libraries (see `requirements.txt`)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/Ahir7/3D-measurement.git
cd 3D-measurement
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run quick test to verify installation:
```bash
python quick_test.py
```

## 📂 Project Structure

```
├── dust3r/               # DUSt3R implementation
├── mast3r/              # MASt3R implementation
├── test_images/         # Sample images for testing
├── results/             # Output directory
├── test_dimension_calculator.py  # Main implementation
├── quick_test.py        # Installation verification
└── requirements.txt     # Project dependencies
```

## 🎯 Usage

1. Basic usage with the dimension calculator:

```python
from test_dimension_calculator import DUSt3RDimensionCalculator

# Initialize the calculator
calculator = DUSt3RDimensionCalculator()

# Calculate dimensions from images
image_paths = [
    "test_images/image1.jpg",
    "test_images/image2.jpg",
    "test_images/image3.jpg"
]
calculator.calculate_dimensions(image_paths)
```

## 📤 Output

The system generates:
- Point cloud data (`.npy` format)
- Dimension results (`.json` format)
- All outputs are saved in the `results/` directory

## 🔍 Model Information

The project uses the DUSt3R model for 3D reconstruction:
- Default model: `DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`
- Located in: `dust3r/checkpoints/`
- Uses ViT-Large architecture with base decoder

## 💻 Hardware Requirements

- Recommended: CUDA-capable GPU with 4GB+ VRAM
- Minimum: Any CPU (significantly slower processing)
- RAM: 8GB minimum, 16GB recommended

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the terms of the LICENSE file included in the repository.

## 🚨 Troubleshooting

1. CUDA Issues:
   - Verify CUDA installation with `quick_test.py`
   - Check GPU compatibility
   - Ensure correct PyTorch version

2. Memory Issues:
   - Reduce image resolution
   - Process fewer images simultaneously
   - Use CPU mode if VRAM is insufficient

## 📞 Contact

- GitHub: [@Ahir7](https://github.com/Ahir7)
