# Multimodal Emotion-Cause Pair Extraction in Conversations

This repository aims to provide the working code for the paper: [Multimodal Emotion-Cause Pair Extraction in Conversations](https://arxiv.org/pdf/2110.08020).

## Introduction

The goal of this project is to extract emotion-cause pairs in conversations using multimodal data including text, audio, and video. The code is based on the methodologies described in the paper.
Emotions included are Anger, Disgust, Sadness, Joy, Neutral, Surprise and Fear.

## Dataset

We use the ECF dataset for this project, which includes the Emotion-Cause Pairs for utterances from "Friends" TV show.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/nagajas/NVDL.git
    cd NVDL
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

## Usage

### Preprocessing

1. Extract features from the dataset using OpenSMILE for audio, 3D-CNN for video, and BERT for text.
2. Preprocess the data and generate the embeddings.

## Citation

If you use this code, please cite the following paper:
