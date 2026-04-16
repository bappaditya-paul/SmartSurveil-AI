# Jupyter Notebooks

This directory contains Jupyter notebooks for experimentation, prototyping,
and analysis related to the surveillance system.

## Notebook Index

| Notebook | Purpose |
|----------|---------|
| `01_detection_demo.ipynb` | Test YOLOv8 detection on sample images |
| `02_tracking_demo.ipynb` | Demonstrate DeepSORT tracking |
| `03_behavior_analysis.ipynb` | Prototype behavior classification |
| `04_performance_analysis.ipynb` | Benchmark system performance |
| `05_heatmap_visualization.ipynb` | Generate and visualize activity heatmaps |

## Usage

Launch Jupyter:
```bash
jupyter notebook
# or
jupyter lab
```

Create a new notebook for experimentation:
```python
# Standard imports for all notebooks
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
import sys
sys.path.append('../src')

# Import system modules
from detection_module import DetectionModule
from tracking_module import TrackingModule
```

## Notebook Best Practices

1. **Restart and run all** before committing to ensure reproducibility
2. **Clear outputs** before git commit (keeps repository clean)
3. **Document assumptions** in markdown cells
4. **Save experimental results** to `data/experiments/`
