# Denoising-and-4x-Super-Resolution-of-Low-Light-Images

# Low-Light Image Enhancement Pipeline: Denoising + 4× Super-Resolution using Real-ESRGAN

## Overview

This repository implements an end-to-end **image enhancement pipeline** designed for low-light and degraded images, combining classical image processing techniques with state-of-the-art deep learning based super-resolution. The system performs **noise suppression followed by 4× spatial resolution upscaling**, producing high-fidelity outputs suitable for downstream vision tasks or visual inspection.

The pipeline is built around a **pretrained Real-ESRGAN model (anime fine-tuned variant)** for super-resolution, integrated with **non-local means denoising** from OpenCV to handle low-light noise artifacts before upscaling. The implementation is optimized for GPU acceleration and structured for batch inference on large image sets.

This project was developed and executed in a Kaggle GPU environment (NVIDIA Tesla T4) and includes a complete inference-to-submission workflow.

## Dataset
The dataset will consist of three key components:

Training Set: Contains low-resolution noisy images along with their corresponding high-resolution clean ground truth images.

Evaluation Set: Similar to the training set, it includes low-resolution noisy images and their corresponding high-resolution clean images, allowing participants to validate the performance of their models during development.

Test Set: Comprises only low-resolution noisy images without any corresponding high-resolution clean images. The performance of the models will be evaluated on this set.
---

## Problem Framing

Low-light images typically suffer from:

* High sensor noise
* Loss of fine-grained textures
* Blurring and low contrast
* Resolution degradation

Applying super-resolution directly on noisy inputs often amplifies noise and compression artifacts. This pipeline explicitly separates the enhancement task into two stages:

1. **Noise suppression (preprocessing)**
2. **Learned super-resolution (reconstruction)**

This staged design ensures that the super-resolution model operates on cleaner feature distributions, improving perceptual quality and structural consistency.

---

## Pipeline Architecture

The system is composed of the following sequential stages:

### 1. Input Acquisition

* Images are loaded from a structured test directory.
* Supported formats: `.png`, `.jpg`, `.jpeg`
* Images are converted to RGB and normalized through PIL and NumPy pipelines.

---

### 2. Denoising Module (Preprocessing)

**Technique:**
OpenCV `fastNlMeansDenoisingColored`

**Rationale:**
Non-local means denoising is effective at removing low-light chromatic and luminance noise while preserving edge structures and textures.

**Operation:**

* Input images are converted from PIL (RGB) → NumPy → BGR (OpenCV format).
* Denoising parameters are tuned to balance noise suppression and detail retention.
* Output is converted back to PIL RGB format for model compatibility.

This step reduces high-frequency noise components that would otherwise be amplified during super-resolution.

---

### 3. Super-Resolution Module

**Model:**
Real-ESRGAN (Anime Fine-Tuned Variant)

**Source:**
`danhtran2mind/Real-ESRGAN-Anime-finetuning` via Hugging Face Hub

**Framework:**
Loaded using `stablepy` for simplified inference integration.

**Key Characteristics:**

* GAN-based architecture trained for perceptual quality
* Enhanced edge sharpness and texture reconstruction
* 4× spatial upscaling factor

**Inference Strategy:**

* Model is loaded onto GPU (CUDA) if available.
* Half precision (FP16) is enabled on GPU to optimize memory and speed.
* Tiling is disabled (`tile=0`) to preserve global context, with overlap padding for stability.

Each denoised image is passed through the upscaler to produce a 4× super-resolved output.

---

### 4. Output Management

* Enhanced images are saved to a dedicated output directory.
* Original filenames are preserved for traceability.
* Batch processing is handled using `tqdm` for progress tracking.

---

### 5. Submission Serialization

The pipeline includes a custom **image-to-CSV serialization layer** to support competition-style submissions:

* Images are converted to grayscale.
* Pixel arrays are flattened and subsampled.
* Image IDs are mapped from `test_` to `gt_` format.
* Final output is written as `submission.csv` with structured pixel columns.

This step transforms visual outputs into a machine-readable tabular format suitable for automated evaluation pipelines.

---

## Model and Algorithmic Choices

### Why Real-ESRGAN?

* Trained on diverse degradation patterns
* Strong performance on texture hallucination and edge reconstruction
* Robust to compression artifacts and low-quality inputs
* GAN-based perceptual loss improves visual realism

### Why Pre-Denoising?

* Super-resolution models tend to amplify noise
* Pre-cleaning stabilizes feature distributions
* Improves downstream reconstruction fidelity
* Reduces hallucinated artifacts

### Why Anime Fine-Tuned Variant?

* Strong edge priors
* High-frequency detail preservation
* Performs well even on natural images when texture recovery is critical

---

## Performance Considerations

* **GPU Acceleration:** CUDA-enabled inference for Real-ESRGAN
* **Batch Processing:** Iterative loop optimized for dataset-scale inference
* **Mixed Precision:** FP16 inference on GPU for reduced memory footprint
* **Minimal Overhead:** Direct PIL → model → PIL pipeline to avoid unnecessary tensor conversions

---

## Dependency Stack

* **PyTorch** – model inference and device management
* **OpenCV** – denoising and color space transformations
* **PIL (Pillow)** – image I/O and format handling
* **stablepy** – model loading and abstraction layer
* **huggingface_hub** – pretrained weight retrieval
* **NumPy** – pixel-level manipulation
* **tqdm** – progress tracking

---

## Execution Environment

* **Platform:** Kaggle Notebook
* **GPU:** NVIDIA Tesla T4
* **Python:** 3.11
* **Frameworks:** PyTorch, OpenCV, Hugging Face ecosystem

---

## Design Philosophy

This project is intentionally structured as a **modular enhancement pipeline** rather than a monolithic black-box model. Each stage is independently replaceable:

* Denoiser can be swapped with learned denoising models (DnCNN, U-Net, etc.)
* Super-resolution model can be replaced with ESRGAN, SwinIR, or diffusion-based SR models
* Serialization logic can be adapted for different evaluation formats

This makes the pipeline suitable as a **baseline architecture** for:

* Low-light image restoration
* Super-resolution research
* Preprocessing stage for downstream vision models
* Competition benchmarking

---

## Summary

This repository demonstrates a practical, production-style image enhancement pipeline that combines:

* Classical signal processing (non-local means denoising)
* Deep learning based reconstruction (Real-ESRGAN)
* GPU-optimized inference
* Structured output serialization

The design emphasizes **data quality, modularity, and inference stability**, making it suitable for both research experimentation and real-world deployment scenarios.


# References:
Kaggle Competition Link: https://www.kaggle.com/competitions/dlp-may-2025-nppe-3/overview
