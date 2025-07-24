# Semantic Chunking Script

A Python script that performs intelligent text chunking on JSON files using semantic similarity. The script groups semantically similar text segments together while respecting character limits, making it ideal for preparing text data for large language models or information retrieval systems.

## Overview

This script takes a JSON file containing a list of text strings and chunks them based on semantic similarity using sentence transformer models. It ensures that:

- Semantically similar segments are grouped together
- Chunks don't exceed a maximum character limit
- Optionally, chunks meet a minimum character threshold
- Split decisions are logged for transparency

## Features

- **Semantic Chunking**: Uses sentence transformer models to compute cosine similarity between text segments
- **Flexible Text Splitting**: Supports multiple sentence splitting methods (spaCy, PySBD, or none)
- **Character Limits**: Configurable maximum and minimum character limits per chunk
- **GPU Support**: Automatic CUDA detection with optional FP16 precision
- **Detailed Logging**: Comprehensive logging of split decisions and statistics
- **Batch Processing**: Configurable batch sizes for efficient encoding

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yigittuncer07/semantic_chunking
cd semantic_chunking
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python semantically_chunk.py input_file.json --log-file log.log
```

This will:
- Process `input_file.json` using default settings
- Use the BAAI/bge-m3 model for embeddings
- Set similarity threshold to 0.5
- Limit chunks to 8000 characters maximum
- Save output to `./output/input_file.json`
- Save logs to log.txt

### Advanced Usage

```bash
python semantically_chunk.py input_file.json \
    --output_dir ./my_output \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --threshold 0.6 \
    --max-chars 4000 \
    --min-chars 500 \
    --batch-size 32 \
    --splitter spacy \
    --log-file splits.log \
    --fp16
```

## Command Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `input_file` | - | Required | Path to JSON file containing list of strings |
| `--output_dir` | `-o` | `./output` | Directory to save chunked JSON |
| `--model` | `-m` | `BAAI/bge-m3` | SentenceTransformer model name |
| `--device` | `-d` | Auto-detect | Device for model (cuda or cpu) |
| `--fp16` | - | False | Load model in half precision |
| `--threshold` | `-t` | 0.5 | Similarity threshold for merging segments |
| `--max-chars` | `-c` | 8000 | Maximum characters per chunk |
| `--min-chars` | `-n` | None | Minimum characters per chunk |
| `--batch-size` | `-b` | 1 | Batch size for encoding |
| `--splitter` | - | `none` | Sentence splitter: none, spacy, or pysbd |
| `--log-file` | - | None | Path to write split logs (prints to console if not set) |

## Input Format

The input JSON file should contain a list of strings:

```json
[
    "This is the first text segment.",
    "This is another segment that might be related.",
    "Here's a completely different topic.",
    "This continues the different topic."
]
```

## Output Format

The script outputs a JSON file with semantically grouped chunks:

```json
[
    "This is the first text segment.\nThis is another segment that might be related.",
    "Here's a different topic.\nThis continues the different topic."
]
```

## Algorithm Details

### 1. Text Preprocessing
- **Sentence Splitting**: Optionally splits input texts into sentences using:
  - **spaCy**: Language-aware sentence segmentation
  - **PySBD**: Rule-based sentence boundary detection
  - **None**: Uses input texts as-is
- **Short Sentence Merging**: Merges sentences shorter than 16 characters with the previous sentence

### 2. Semantic Chunking Process
The [`semantic_chunk`](semantically_chunk.py) function:

1. **Encoding**: Converts all text segments to embeddings using the specified model
2. **Similarity Calculation**: Computes cosine similarity between consecutive segments
3. **Chunking Decision**: Merges segments if:
   - Cosine similarity > threshold AND
   - Combined length ≤ max_chars
4. **Split Logging**: Records reasons for splits (similarity or character limits)

### 3. Post-processing
The [`apply_min_chars`](semantically_chunk.py) function merges chunks shorter than `min_chars` with the previous chunk.

## Performance Tips

1. **GPU Usage**: The script automatically detects CUDA. Use `--fp16` for faster processing with minimal quality loss
2. **Batch Size**: Increase `--batch-size` for faster encoding on powerful GPUs
3. **Model Selection**: Choose models based on your language and quality requirements:

## Output Statistics

The script provides detailed statistics:
- Number of segments processed
- Similarity statistics (mean, standard deviation)
- Split counts by reason (threshold vs character limits)
- Merge counts for minimum character enforcement
