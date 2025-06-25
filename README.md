
# MuDiT-MuSiT: Aligning Generative Music Models with Professional and Amateur Users' Expectations

This repository contains the official implementation and resources for our paper, **"Generative Music Models' Alignment with Professional and Amateur Users' Expectations"**. Our work addresses a critical and underexplored challenge in text-to-music generation: understanding users with diverse musical expertise and generating music that meets their varied expectations.

To tackle this, we introduce the novel task of **Professional and Amateur Description-to-Song Generation**. The core of our research is a single-stage generation framework, **MuDiT/MuSiT**, designed to bridge the semantic gap between human expression and machine understanding.

---

### Core Contributions

-   **MuDiT/MuSiT Framework**: We propose a novel, end-to-end, single-stage framework based on Diffusion/Interpolant Transformers (DiT/SiT). This framework is designed to generate complete songs (including vocals and accompaniment) that align with user descriptions.
-   **ChinMu Cross-Modal Encoder**: To better understand colloquial Chinese descriptions, we developed **ChinMu**, a cross-modal encoder based on CLAP and MuLan architectures, trained on our unique dataset.
-   **The MuChin Dataset**: Our work is built upon the **MuChin dataset**, which uniquely contains paired annotations from both professional musicians and amateur listeners for identical songs, providing a rich source for studying the "semantic divide".
-   **New Evaluation Metrics**: We introduce novel metrics to evaluate the alignment between generated audio and user intent, including Semantic Similarity and Acoustic Similarity .

---

### Repository Structure Overview

This repository is organized to reflect the key components of our research pipeline. The folder provided maps to the following experimental areas:

-   **Model Architectures & Training (`dit_*`, `vae_*`, `hifigan_*`, etc.)**:
    -   `dit_training_on_vae/`: Our core **MuDiT** implementation, a Diffusion Transformer that operates in the VAE latent space to generate WAV audio.
    -   `sit_prompt_finetuning/`: Our core **MuSiT** implementation, which uses a Scalable Interpolant Transformer, offering a more flexible and efficient generation process.
    -   `vae_vocal_training/` & `direct_wav_vae_distributed/`: Scripts for training the foundational VAE (Variational Autoencoder) models that create the latent space for our DiT/SiT models.
    -   `hifigan_vocoder/`: Contains the HiFiGAN model used as a high-fidelity vocoder to convert spectral representations (like Mel spectrograms) back into audio waveforms.
    -   `music_understanding_model/`: Contains various tools and models for music understanding, including **NVIDIA Apex** for high-performance mixed-precision training.

-   **Data Processing & Management (`data_*`)**:
    -   `data_prep_demucs/`: Scripts using Demucs for music source separation (e.g., separating vocals from accompaniment).
    -   `data_synthesis_tool/`: Tools for creating synthesized training data by combining WAV, CLAP vectors, and VAE outputs.
    -   `large_scale_asr_annotation/`: Workflow for annotating our 1.5 million song dataset with lyrics using Whisper ASR, including post-processing scripts to remove model hallucinations.

-   **Cross-Modal & RAG Systems (`clap_*`, `rag_*`)**:
    -   `clap_finetuning_zh_en/`: Scripts for fine-tuning the CLAP model, which forms the basis of our **ChinMu** encoder.
    -   `data_prep_muchin_to_clap_vectors/`: The pipeline for processing the MuChin dataset descriptions into CLAP-based vector embeddings.
    -   `clap_retrieval_system/` & `rag_search_frontend/`: A CLAP-based vector retrieval system and its frontend, demonstrating the RAG (Retrieval-Augmented Generation) concept.

-   **Evaluation & Metrics (`intent_*`, `lyric_*`, `Semantic-Analysis-Test-Data`)**:
    -   `intent_fidelity_metrics/`: Code for calculating our proposed (Semantic Audio Alignment) and (Acoustic Reference Alignment) metrics.
    -   `lyric_evaluation_metrics/`: A detailed framework for objectively evaluating the quality of LLM-generated lyrics based on structure, word count, and rhyme.
    -   `Semantic-Analysis-Test-Data/`: Code and data for assessing the semantic similarity between text-based tags, a key part of analyzing the amateur-professional divide.

---

### Installation & Setup

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/CarlWangChina/MuDiT-MuSiT.git](https://github.com/CarlWangChina/MuDiT-MuSiT.git)
    cd MuDiT-MuSiT
    ```

2.  **Create a Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Core Dependencies**:
    A `requirements.txt` should be provided for the main components.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install Specific Tool Dependencies**:
    - **Semantic Analysis**:
        ```bash
        pip install -U FlagEmbedding transformers==4.34.0
        # Download the embedding model
        git clone [https://huggingface.co/BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5) ./Semantic-Analysis-Test-Data/models/bge-large-zh-v1.5
        ```
    - **NVIDIA Apex** (for high-performance training):
        ```bash
        cd ./music_understanding_model/apex
        pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
        cd ../.. 
        ```

---

### How to Use: A General Workflow

Our framework follows a three-stage process as detailed in the paper: Module Preparation, Pre-training, and Fine-tuning.

#### 1. Module Preparation

-   **VAE Training**: First, train a VAE on a large audio dataset (like our 1.5M song collection) to obtain a robust audio encoder/decoder. See `vae_vocal_training/`.
-   **ChinMu Encoder Training**: Fine-tune a CLAP-based model on the MuChin dataset (audio and text descriptions) to create the ChinMu cross-modal encoder. See `clap_finetuning_zh_en/`.
-   **Lyric LLM Fine-tuning**: Fine-tune a large language model (e.g., Qwen) on structured lyric data to enable it to generate lyrics with structural tags like `<verse>` and `<chorus>`. See `Lyric-Generation-LLMs/`.

#### 2. Large-Scale Pre-training

-   Use the large, untagged song dataset (audio + ASR-generated lyrics) to pre-train the DiT/SiT model. This teaches the model the general structure of music and lyrics.
    ```bash
    # Example command (conceptual)
    # This would be a large-scale distributed training job
    deepspeed --num_gpus=8 scripts/train.py --config configs/dit_pretrain_config.yaml
    ```

#### 3. Fine-tuning on MuChin

-   Fine-tune the pre-trained DiT/SiT model using the MuChin dataset. In this stage, the model learns to associate the professional and amateur text descriptions (encoded by ChinMu) with specific musical outcomes.
    ```bash
    # Example command (conceptual)
    deepspeed --num_gpus=8 scripts/train.py --config configs/mudit_finetune_config.yaml --resume_from_checkpoint /path/to/pretrained_model.pth
    ```

#### Component Models

For the reproducibility of specific modules discussed in our paper, we provide the fine-tuned weights for our CLAP-based retrieval model and other models. These are available at our Hugging Face Model repository: 
`huggingface.co/karl-wang/ama-prof-divi`

However, please note that the weights for certain models cannot be open-sourced due to commercial confidentiality agreements with our partner companies.

Additionally, the dataset used for pre-training, which contains the recordings and lyrics of over 1.5 million songs, is not publicly available due to copyright restrictions. Researchers interested in using this dataset for academic research may request access by contacting us via email at carlwang1212@gmail.com.