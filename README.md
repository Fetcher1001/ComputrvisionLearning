## Computer Vision Learning

A repository for learning computer vision and experimentation around computer vision concepts.

Minimal, hands-on computer vision experiments classification, detection, and small demos.

![Project screenshot](doc/Screenshot%202026-03-11%20125018.png)

## Repo layout
- data/          datasets (train / val / test)
- doc/                 documentation and screenshots
- edge_detection/      edge detection demo (scripts and images)
- Experimentation/     notebooks and pictures used during experiments
- mnist/                MNIST examples, notebooks and training scripts
- LICENSE
- README.md

## TL;DR
- Quick examples for MNIST training/inference and small computer-vision experiments.

## Screenshots
![Demo screenshot 1](doc/Screenshot%202026-03-11%20124908.png)
*Demo view.*

![Demo screenshot 2](doc/Screenshot%202026-03-11%20124943.png)
*Additional visualization from experiments.*

## Prerequisites
- Python 3.8+
- pip
- (optional) CUDA-enabled GPU for training

## Quickstart (Windows)
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python mnist/main.py
```

## Quickstart (macOS / Linux)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python mnist/main.py
```

## Examples
- MNIST example: [mnist/main.py](mnist/main.py)
- Experimentation notebook: [Experimentation/testing.ipynb](Experimentation/testing.ipynb)

Notes: this README was updated to match the repository layout. A full visual gallery is available at [doc/GALLERY.md](doc/GALLERY.md) (edge-detection images are deliberately excluded from the gallery).

## Contributing
- Create issues for bugs or features.
- Fork, create a branch, add tests/docs, and open a PR.

## License
See [LICENSE](LICENSE).
