
# Neural Style Transfer: A Comparative Study of Backbone, Layer Selection and Loss Weighting

CM3070 Final Year Project, BSc Computer Science, University of London.

Implementation of the optimisation-based neural style transfer method of
Gatys, Ecker and Bethge (2016), wrapped in an experimental harness that
varies one factor at a time and logs every run.

## Research questions

1. How the choice of backbone (VGG16 vs VGG19) and the style layer set affect the output.
2. How sensitive the result is to the style/content loss weighting.
3. How SGD and Adam compare as the optimiser driving the process.

Twenty runs in total: six for RQ1, six for RQ2, two for RQ3 and a six-run
gallery across three content images and two style images.

## Contents

| Path | What it holds |
|---|---|
| `CM3070_Experiments_NST_Colab.ipynb` | Full implementation with all cell outputs |
| `results/` | Output images per run, and `results_log.csv` |
| `figures/` | Comparison figures used in the report |
| `images/` | Content and style inputs |
| `requirements.txt` | Package versions captured via `pip freeze` |

Intermediate checkpoints were saved every 200 iterations during each run but
are excluded here for size.

## Running it

The notebook was written for Google Colab Pro with a Tesla T4 GPU and mounts
Google Drive for storage, so it will not run unchanged elsewhere.

1. Open the notebook in Colab and select a GPU runtime.
2. Set `PROJECT_ROOT` to your own Drive folder.
3. Place content and style images under `images/content/` and `images/style/`.

The notebook creates the remaining directories and initialises
`results_log.csv` on first run. Every run is keyed by a `run_id` encoding its
parameters, and the sweep runner skips any run already logged, so an
interrupted session resumes rather than repeating finished work.

Each run records its loss components, SSIM against the content image and
wall-clock time to `results_log.csv`.

## Environment

TensorFlow 2.20.0, Python 3.13, Keras 3.13.2, NumPy 2.1.3, Pillow 11.3.0,
scikit-image 0.25.2.
