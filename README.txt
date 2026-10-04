# Multimodal Search: Image Retrieval Based on Text Descriptions

A text-to-image retrieval system that lets you type a free-form description and get back the images in a collection that match it best. Built on the pre-trained CLIP ViT-B/32 dual encoder, with a second caption-based re-ranking stage to sharpen the results, and wrapped in an interactive Gradio demo.

This was developed as my Bachelor's thesis (diploma project) at the Technical University of Cluj-Napoca, Faculty of Electronics, Telecommunications and Information Technology, 2026.

## Overview

Given a text query, the system:

1. **Encodes** the query and every candidate image into a shared embedding space using CLIP ViT-B/32.
2. **Retrieves** the top-K images by cosine similarity between the query embedding and each image embedding.
3. **Re-ranks** that shortlist by comparing the query against each candidate's ground-truth captions in the text embedding space, correcting cases where the first stage matches the general scene but misses the actual subject.
4. **Serves** the result through a Gradio web interface, where a user types a query, picks K, and sees the retrieved images with their scores.

No additional training or fine-tuning is used; the system runs entirely on inference with the pre-trained CLIP checkpoint.

## How it works

```
Text query ──► CLIP text encoder ──┐
                                    ├──► cosine similarity ──► top-K images (stage 1)
Image collection ─► CLIP image encoder ─┘
                                              │
                                              ▼
                              Caption-based re-ranking (stage 2)
                                              │
                                              ▼
                                     Final ranked results
```

- **Stage 1 (retrieval):** all images are L2-normalized CLIP embeddings; the query embedding is compared against the full collection by cosine similarity, and the top-K are kept.
- **Stage 2 (re-ranking):** each of the top-K candidates' captions is embedded and compared against the query in the text embedding space, re-ordering the shortlist. This fixes the main failure mode observed in testing: a result that matches the scene (e.g., a park) but not the actual subject of the query (e.g., a dog in the park).

## Dataset

[Flickr30k]: 31,783 images, 158,915 captions (5 per image), stored as a single image folder plus a pipe-separated `results.csv` caption file.

## Tech stack

| Component | Library | Role |
|---|---|---|
| Tensor ops / inference | PyTorch | Encoding, L2 normalization, similarity computation, GPU acceleration |
| CLIP model | HuggingFace Transformers (`CLIPModel`, `CLIPProcessor`) | Pre-trained ViT-B/32 checkpoint and matching preprocessing |
| Data handling | Pandas | Reading/cleaning the dataset, caption lookup |
| Image I/O | Pillow | Loading and RGB conversion |
| Visualization | Matplotlib | Result grids, comparison figures |
| Interface | Gradio | Interactive web demo |

Developed in Python, using Jupyter notebooks inside VS Code.

## Project structure

```
.
├── data_preparation.ipynb      # Loads, cleans and verifies the Flickr30k dataset
├── retrieval_system.ipynb      # Builds the CLIP pipeline, re-ranking, evaluation and Gradio demo
├── results.csv                 # Pipe-separated image–caption pairs
└── flickr30k_images/           # Dataset images (not included, download separately)
```

> Rename/adjust the above to match your actual repo layout.

## Getting started

### Requirements

```bash
pip install torch transformers pandas pillow matplotlib gradio
```

A CUDA-capable GPU is recommended but not required; the system falls back to CPU automatically.

### Setup

1. Download the [Flickr30k dataset](http://shannon.cs.illinois.edu/DenotationGraph/) (images + `results.csv`).
2. Update the dataset paths (`IMAGE_DIR`, `PATH_CSV`, `OUTPUT_DIR`) at the top of the notebook to point to your local copy.
3. Run `data_preparation.ipynb` once to verify dataset integrity (checks for missing/duplicate images).
4. Run `retrieval_system.ipynb` to build the retrieval pipeline and launch the Gradio demo.

### Usage

Launch the Gradio interface from the notebook, then open the local URL it prints. Type a text query (e.g. *"a man riding a bicycle on a mountain trail"*), pick how many results to retrieve, and view the ranked images with their similarity scores.

## Results

Recall@K measured on three nested collection sizes, using every caption of the collection's images as a query:

| Collection size | Caption queries | R@1 | R@5 | R@10 |
|---|---|---|---|---|
| 1,000 | 5,000 | 59.16% | 84.50% | 90.66% |
| 5,000 | 25,000 | 40.33% | 66.66% | 76.20% |
| 20,000 | 100,000 | 25.36% | 47.17% | 56.85% |

These are internal measurements (not the standard Flickr30k test split), useful mainly as a baseline for comparing future changes to the system. As expected, recall drops as the collection grows, while staying far above random chance (0.1% and 0.005% R@1 respectively for the 1,000 and 20,000-image collections).

Qualitatively, the two-stage design reliably fixes the main failure mode of the first stage: results that match the general scene but not the query's actual subject.

## Limitations and future work

- **Scalability:** retrieval is brute-force cosine similarity, which stops being practical past tens of thousands of images. An approximate nearest-neighbour index (e.g. [FAISS](https://github.com/facebookresearch/faiss)) would keep search fast at much larger scale.
- **No caching:** image embeddings are recomputed on every run/query session rather than being stored, which dominates response time (~32s for a 5,000-image query). Precomputing and caching embeddings is the most direct speed improvement.
- **Generic encoder:** CLIP ViT-B/32 is not fine-tuned on this dataset or domain. A larger backbone (e.g. ViT-L/14) or domain-specific fine-tuning would likely improve both similarity scores and recall, at the cost of encoding time.
- **Evaluation:** recall was measured on internal subsets rather than the standard Flickr30k test partition, so results aren't directly comparable to published CLIP benchmarks.

## Author

**Dragoș-Daniel Degerat**
Technical University of Cluj-Napoca, Faculty of Electronics, Telecommunications and Information Technology
Supervisor: Sl.dr.ing. Ștefania-Ramona Benea

