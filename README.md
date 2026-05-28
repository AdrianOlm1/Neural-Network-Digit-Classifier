# Neural Network Digit Classifier

**A CNN-based system that reads a photo of a Sudoku puzzle, recognizes the digits, and solves it.**

Takes an image of a Sudoku grid, extracts each cell using computer vision, classifies the digits with a trained convolutional neural network, then solves the completed puzzle with a backtracking algorithm.

## Pipeline

```
Input Image  ──►  Grayscale + Adaptive Threshold  ──►  Grid Extraction
                                                            │
Solution  ◄──  Backtracking Solver  ◄──  CNN Digit Recognition (per cell)
```

1. **Preprocessing** — Converts the image to grayscale, applies adaptive thresholding to isolate the grid lines and digits
2. **Cell Extraction** — Splits the 9x9 grid into 81 individual cell images
3. **Digit Recognition** — A 3-layer CNN (Conv2d → MaxPool → FC) trained on MNIST classifies each cell as 0-9
4. **Solving** — Feeds the recognized grid into a recursive backtracking solver

## CNN Architecture

```
Input (1x28x28)
  → Conv2d(1, 32) → ReLU → MaxPool
  → Conv2d(32, 64) → ReLU → MaxPool
  → Conv2d(64, 128) → ReLU → MaxPool
  → Flatten → FC(256) → FC(10)
```

Trained on the MNIST dataset using PyTorch.

## Tech Stack

- **PyTorch** — CNN model definition and training
- **OpenCV** — Image preprocessing and grid extraction
- **NumPy** — Matrix operations
- **Matplotlib** — Visualization

## Usage

```bash
pip install torch torchvision opencv-python numpy matplotlib

# Run the full pipeline on an image
python nn.py

# Or explore interactively
jupyter notebook nn.ipynb
```

## Files

| File | Purpose |
|------|---------|
| `nn.py` | Full pipeline — image to solved Sudoku |
| `nn.ipynb` | Interactive notebook version |
| `BACKTRACKER.py` | Recursive Sudoku solver |
| `sudoku_cnn_full.pth` | Pre-trained model weights |
| `test.jpg` | Sample Sudoku image |

## License

MIT
