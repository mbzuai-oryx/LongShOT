# LongShOT: A Benchmark and Agentic Framework for Omni-Modal Reasoning and Tool Use in Long Videos

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="LongShOT">
</p>

#### Mohammed Irfan K\*, Jaseel Muhammad Kaithakkodan\*, Jinxing Zhou, Sahal Shaji Mullappilly, Mohammad Almansoori, Noor Ahsan, Beknur Kalmakhanbet, Sambal Shikhar, Rishabh Lalla, Jean Lahoud, Mariette Awad, Fahad Shahbaz Khan, Salman Khan, Rao Muhammad Anwer, and Hisham Cholakkal

\**Equally contributing first authors*

#### **Mohamed Bin Zayed University of Artificial Intelligence (MBZUAI), UAE**
[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://github.com/mbzuai-oryx/longshot)
[![Paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2412.07769)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Dataset-F9D371)](https://huggingface.co/collections/MBZUAI/longshot_bench)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey)](https://github.com/mbzuai-oryx/BiMediX/blob/main/LICENSE.txt)

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
  - [Setup YouTube Cookie Authentication](#setup-youtube-cookie-authentication)
  - [Download Videos](#download-videos)
  - [Process Videos with Vision-Language Models (VLM)](#process-videos-with-vision-language-models-vlm)
  - [Process Videos with Language Models (LLM)](#process-videos-with-language-models-llm)
- [Dataset Generation](#dataset-generation)
- [Upload to Hugging Face](#upload-to-hugging-face)
- [Evaluation](#evaluation)
  - [Generate Model Responses](#generate-model-responses)
  - [Evaluate Responses](#evaluate-responses)

## Overview

LongShOT introduces a diagnostic benchmark and agentic framework for long-form multimodal video understanding. **LongShOTBench** features open-ended questions, multi-turn dialogues, and tasks requiring vision, speech, and audio reasoning with tool use. Each sample includes reference answers and graded rubrics for interpretable evaluation. **LongShOTAgent** employs preprocessing, search, and iterative refinement to analyze long videos. Current state-of-the-art models show significant gaps: Gemini-2.5-Flash achieves 52.95%, open-source models remain below 30%, and LongShOTAgent attains 44.66%, highlighting the challenge of real-world long video understanding.

<p align="center">
    <img src="figures/generation_pipeline_overview.pdf" alt="Generation Pipeline">
</p>

**Figure 1:** Construction pipeline of LongShOTBench. The pipeline begins with raw video data where speech, visuals, and audio cues are extracted. These are passed into multimodal processing to generate segment-wise aligned and fused metadata. Only the distilled information flows to question design, where scenarios and question types are mapped, followed by the generation of questions and conversational answers. Next, verifiable rubrics are created to evaluate correctness and difficulty. Finally, the core dataset, comprising Q&A pairs and tailored evaluation rubrics, is manually reviewed and corrected by human validators, ensuring a clean, reliable benchmark.

## Installation

Create and activate a conda environment with Python 3.11:

```bash
conda create -n longshot python=3.11 -y
conda activate longshot
pip install -r requirements.txt
```

> **Note:** If you encounter CUDNN issues, install it separately for your version:
> ```bash
> conda install -c conda-forge cudnn=8
> ```

## Data Preprocessing

The preprocessing pipeline downloads videos from YouTube and processes them to extract video, audio, and speech metadata/transcriptions.

### Setup YouTube Cookie Authentication

To download videos from YouTube, you need to provide authentication cookies:

1. Install the [Get cookies.txt LOCALLY](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc) Chrome extension
2. Navigate to YouTube and refresh the page
3. Click on the extension icon and copy the cookies
4. Paste the cookies into `preprocess/cookies.txt`

### Download Videos

Navigate to the preprocess directory and download videos:

```bash
cd preprocess
python download_youtube_videos.py
```

> **Note:** Some videos may fail to download through the script. Please manually download any missing videos and place them in the appropriate directory (preprocess/dataset/videos).

### Process Videos with Vision-Language Models (VLM)

Start the VLM server in a separate terminal:

```bash
./vllm_start.sh vlm
```

In another terminal, run the VLM processing stages:

```bash
python run.py vlm-stages
```

### Process Videos with Language Models (LLM)

Start the LLM server in a separate terminal:

```bash
./vllm_start.sh llm
```

In another terminal, run the LLM processing stages:

```bash
python run.py llm-stages
```

Once complete, the raw dataset with video, audio, and speech descriptions will be ready in the `preprocess/dataset/` directory.

## Dataset Generation

Generate the LongShOTBench dataset from the preprocessed videos:

```bash
cd ../datagen
python main.py
```

The final dataset will be saved at `datagen/results/final_dataset.jsonl`.

## Upload to Hugging Face

To share the dataset on Hugging Face Hub:

```python
from datasets import Dataset
dataset = Dataset.from_json("results/clean_dataset.jsonl")
dataset.push_to_hub("your-org/longshot-bench", config_name="postvalid")
```

## Evaluation

Evaluate model performance on the LongShOTBench dataset.

### Generate Model Responses

To generate responses from candidate models:

```bash
cd ../eval
bash generate.sh
```

### Evaluate Responses

To evaluate the generated responses:

```bash
bash eval.sh
```

Results will be saved in the `eval/results_postvalid/` directory.

