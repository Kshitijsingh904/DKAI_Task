# Preliminary Clinical Imaging Assistant using Qwen2-VL

## Objective

This project implements an open-source Vision-Language Model (VLM) as a preliminary clinical imaging assistant, built to run end-to-end on the free Google Colab environment.

> **Important:** This project is an educational demonstration only. It is **not** intended for clinical diagnosis or treatment decisions.

## Selected Model

| | |
|---|---|
| Model | Qwen2-VL-2B-Instruct |
| Source | Hugging Face (`Qwen/Qwen2-VL-2B-Instruct`) |
| Hardware target | Google Colab NVIDIA T4 GPU |
| Quantization | 4-bit NF4 (BitsAndBytes) with double quantization |

**Why this model:** Qwen2-VL-2B-Instruct handles both images and natural-language instructions, is small enough (2B parameters) to fit a free Colab T4 GPU once 4-bit quantized, and is openly available on Hugging Face — giving a practical balance between multimodal capability and resource constraints versus larger VLMs.

## Requirements Covered

1. Select and justify an open-source Vision-Language Model.
2. Load the model using memory-efficient 4-bit quantization.
3. Create a reusable multimodal inference function.
4. Design a clinical imaging system prompt.
5. Run inference on at least two publicly available medical images.
6. Document sample outputs, design choices, and limitations.

## How the Notebook Works

1. **Hardware check** — confirms a CUDA GPU is available and prints GPU name/memory (prompts to switch to a T4 runtime if not).
2. **Package installation** — installs pinned versions of `transformers`, `accelerate`, `bitsandbytes`, `pillow`, and `requests`.
3. **Model loading** — configures 4-bit NF4 quantization via `BitsAndBytesConfig` and loads `Qwen2VLForConditionalGeneration` with `device_map="auto"`.
4. **Clinical system prompt** — defines a structured prompt instructing the model to act as a cautious, non-diagnostic imaging assistant that reports only what is visible in the image (image type/quality, key observations, differential considerations, limitations).
5. **Test images** — downloads two public medical images from Wikimedia Commons: a chest X-ray and a brain MRI.
6. **Reusable inference pipeline**:
   - `load_image()` — loads an image from a path or URL and resizes it (max 1024px) to control GPU memory/token usage.
   - `run_clinical_inference()` — takes an image and a clinical query, builds the multimodal prompt, runs generation, and returns the model's structured response.
7. **Inference runs** — the pipeline is run once on the brain MRI and once on the chest X-ray, with GPU memory checked before and cleared after.
8. **Design choices, limitations, and conclusion** — markdown sections documenting why the model/quantization/prompt approach was chosen, and the known limitations of the system.

## Design Choices

- **4-bit quantization (NF4 + double quant)** to fit the model into a free-tier Colab GPU.
- **`device_map="auto"`** so Hugging Face Accelerate handles hardware placement automatically across compatible environments.
- **Reusable inference function** that cleanly separates image loading, prompt construction, generation, and memory cleanup, making it easy to run on additional images or queries.
- **Image resizing** before inference to avoid excessive visual tokens and GPU memory blowups from large medical images.

## Limitations & Safety

- **General-purpose model:** Qwen2-VL-2B-Instruct is not a dedicated medical/radiology model — its outputs shouldn't be assumed to match specialized systems.
- **Hallucination risk:** VLMs can produce plausible but unsupported statements; outputs need independent verification.
- **No patient context:** The model sees only the image and text prompt — no history, symptoms, labs, or prior imaging.
- **Image quality dependence:** Poor resolution, compression, positioning, or artifacts can degrade output quality.
- **Not a diagnostic tool:** All outputs are preliminary, educational, and require qualified human review.

## Requirements

```
transformers>=4.45.0
accelerate>=0.34.0
bitsandbytes>=0.43.0
pillow>=10.0.0
requests>=2.31.0
```

Run in **Google Colab** with runtime type set to **T4 GPU** (Runtime → Change runtime type → T4 GPU).

## Usage

1. Open `VLM_Inference.ipynb` in Google Colab and select a T4 GPU runtime.
2. Run all cells top to bottom — this installs dependencies, loads the quantized model, downloads the two sample images, and runs inference on both.
3. To analyze a new image, call:

```python
result = run_clinical_inference(
    image_input="path_or_url_to_image",
    user_query="Please provide a systematic preliminary assessment of this image...",
    max_image_size=1024,
    max_new_tokens=256
)
print(result)
```

## Conclusion

This project demonstrates an end-to-end preliminary clinical imaging workflow using an open-source, 4-bit quantized VLM on a free Colab GPU — with a reusable inference function, a structured clinical system prompt, and explicit communication of uncertainty and limitations. Outputs are for educational demonstration only and must not be treated as medical diagnoses.
