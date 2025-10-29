# Slitherbot

A computer vision project using YOLOv8 for real-time object detection in Slither.io gameplay. This project provides tools for dataset creation, model training, and live detection overlay during gameplay.

> **Note**: This project currently focuses on detection only. AI gameplay using neural networks is planned for future development.

## Features

- 🎮 **Game-Specific Detection**: Trained to detect Slither.io game elements (borders, points, snakes)
- 📸 **Dataset Tools**: Automated screenshot capture for building custom datasets
- 🏋️ **Model Training**: Easy-to-use training pipeline with YOLOv8
- 🎯 **Real-Time Overlay**: Live detection visualization during gameplay
- 🏷️ **Annotation Tools**: Label Studio integration for efficient data labeling

## What This Project Does

This is a **detection-only** system that:
- Captures and labels Slither.io gameplay screenshots
- Trains a YOLO model to detect game elements
- Displays real-time bounding boxes over the game window

**Future Plans**: Neural network for autonomous gameplay (not yet implemented)

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for training and real-time inference)
- Slither.io (browser-based game)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/gravyt1/slitherbot.git
cd slitherbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
slitherbot/
├── dataset/
│   ├── images/
│   │   ├── train/          # Training images
│   │   └── val/            # Validation images
│   ├── labels/
│   │   ├── train/          # Training labels (YOLO format)
│   │   └── val/            # Validation labels
│   ├── config.yaml         # Dataset configuration
│   └── test.png            # Test image
├── screenshots/            # Raw captured screenshots
├── tools/
│   └── labeling/
│       ├── create_dataset.py   # Screenshot capture tool
│       ├── model.py            # Label Studio ML backend
│       └── readme.txt
├── runs/                   # Training outputs (auto-generated)
│   └── detect/
│       └── train/
│           └── weights/
│               └── best.pt # Trained model
├── real_time.py           # Live detection overlay
├── train_model.py         # Model training script
├── LICENSE                # MIT License
└── README.md
```

## Quick Start

### 1. Capture Screenshots

Start Slither.io in your browser and run:

```bash
cd tools/labeling
python create_dataset.py
```

The script will:
- Wait 3 seconds before starting
- Capture screenshots at 0.1s intervals
- Save to `slitherbot/screenshots/`

**Tip**: Adjust the `MONITOR_REGION` in the script to match your game window position.

### 2. Annotate Your Data

Use [Label Studio](https://labelstud.io/) to label your screenshots:

1. Install Label Studio: `pip install label-studio`
2. Start Label Studio: `label-studio start`
3. Create a new project and import screenshots
4. Label the following classes:
   - `border`: Game boundaries
   - `point`: Food pellets
   - `snake`: Snake bodies (yours and others)

Organize labeled data in YOLO format under `dataset/images/` and `dataset/labels/`.

**Example `dataset/config.yaml`:**
```yaml
path: ./slitherbot/dataset
train: images/train
val: images/val

nc: 3
names: ['border', 'point', 'snake']
```

### 3. Train the Model

From the project root:

```bash
python train_model.py
```

**Default training parameters:**
- Model: YOLOv11n (nano - fastest)
- Image size: 640x640
- Batch size: 4
- Epochs: 75
- Data augmentation: Enabled

Training will save the best model to `runs/detect/train/weights/best.pt`.

### 4. Run Real-Time Detection

Start Slither.io and run the detection overlay:

```bash
python real_time.py
```

**Controls:**
- Press `Esc` to stop detection

A window will display the game with bounding boxes around detected objects.

## Configuration

### Screen Capture Region

Adjust based on your monitor setup and game window position.

**In `real_time.py` and `tools/labeling/create_dataset.py`:**
```python
MONITOR_REGION = {
    "top": 120,              # Offset from top of screen
    "left": 0,               # Offset from left
    "width": 1920,           # Capture width
    "height": int(1080 * 0.84)  # Capture height (excludes UI)
}
```

### Detection Settings

**In `real_time.py`:**
```python
MODEL_PATH = "runs/detect/train/weights/best.pt"  # Your trained model
CONF_THRESHOLD = 0.25  # Minimum confidence (0.0-1.0)
DELAY = 0.01          # Frame delay (lower = faster, higher CPU usage)
```

### Training Hyperparameters

**In `train_model.py`:**
```python
model.train(
    data="dataset/config.yaml",
    imgsz=640,        # Image size (640, 1280, etc.)
    batch=4,          # Batch size (reduce if OOM)
    epochs=75,        # Training duration
    workers=6,        # CPU workers for data loading
    augment=True      # Enable data augmentation
)
```

## Label Studio ML Backend (Optional)

Speed up annotation with auto-labeling:

1. Set environment variables in .env file
```
MODEL_PATH="runs/detect/train/weights/best.pt"
LABEL_STUDIO_URL="http://localhost:8080"
LABEL_STUDIO_API_TOKEN="your_token_here"
```

2. Start the ML backend:
```bash
cd tools/labeling
label-studio-ml start model.py
```

3. Connect in Label Studio project settings under "Machine Learning"

## Performance Optimization

### For Real-Time Detection:
- **Use GPU**: Ensure CUDA is properly installed
- **Lighter model**: Use `yolo11n.pt` (nano) instead of larger variants
- **Reduce resolution**: Lower `imgsz` parameter
- **Adjust delay**: Increase `DELAY` value if FPS is low

### For Training:
- **GPU memory**: Reduce `batch` size if out of memory
- **Faster training**: Use smaller image size (imgsz=416)
- **Better accuracy**: Increase epochs, use larger model (yolo11m.pt)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Low FPS in detection** | Increase `DELAY`, use lighter model, enable GPU |
| **CUDA out of memory** | Reduce `batch` size or `imgsz` in training |
| **Wrong screen region** | Adjust `MONITOR_REGION` top/left offsets |
| **Poor detection accuracy** | Collect more training data, increase epochs |
| **Model doesn't detect anything** | Lower `CONF_THRESHOLD`, check if model trained properly |

## Roadmap

- [x] Screenshot capture tool
- [x] YOLO model training pipeline
- [x] Real-time detection overlay
- [x] Label Studio integration
- [ ] Neural network for decision-making
- [ ] Autonomous gameplay agent
- [ ] Reinforcement learning implementation

## Contributing

Contributions are welcome! Feel free to:
- Report bugs or request features via [Issues](https://github.com/gravyt1/slitherbot/issues)
- Submit pull requests for improvements
- Share your trained models or datasets

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This project is for educational and research purposes only. Use of automation tools may violate Slither.io's terms of service. The authors are not responsible for any consequences of using this software.

## Acknowledgments

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) - State-of-the-art object detection
- [Label Studio](https://github.com/heartexlabs/label-studio) - Data annotation platform
- [MSS](https://github.com/BoboTiG/python-mss) - Fast screenshot library
- Slither.io game by Steve Howse

## Citation

If you use this project in your research:

```bibtex
@software{slitherbot_yolo,
  author = {ESNAULT Jules},
  title = {Slitherbot},
  year = {2025},
  url = {https://github.com/gravyt1/slitherbot}
}
```