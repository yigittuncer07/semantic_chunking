#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import sys
import statistics

import torch
from sentence_transformers import SentenceTransformer, util

try:
    import spacy
except ImportError:
    spacy = None

try:
    import pysbd
except ImportError:
    pysbd = None


def load_spacy_sentencizer(lang: str = "tr"):
    print(f"[+] Initializing spaCy sentencizer (lang={lang})…")
    if not spacy:
        raise RuntimeError("spaCy is not installed. Install with `pip install spacy`.")
    nlp = spacy.blank(lang)
    nlp.add_pipe("sentencizer")
    return nlp


def load_pysbd_segmenter(lang: str = "kk"): # no turkish so next best thing
    print(f"[+] Initializing PySBD segmenter (lang={lang})…")
    if not pysbd:
        raise RuntimeError("pysbd is not installed. Install with `pip install pysbd`.")
    return pysbd.Segmenter(language=lang)


def merge_short_sentences(sentences, min_length=16):
    merged = []
    for sent in sentences:
        if len(sent) < min_length and merged:
            # merge into previous
            merged[-1] = merged[-1].rstrip() + " " + sent.lstrip()
        else:
            merged.append(sent)
    return merged


def semantic_chunk(texts, model, threshold: float, max_chars: int, batch_size: int, split_log_file=None):
    print(f"[+] Encoding {len(texts)} segments (batch_size={batch_size})…")
    embeddings = model.encode(
        texts,
        convert_to_tensor=True,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    raw_chunks = []
    sims = []
    current = texts[0]
    curr_len = len(current)
    max_char_splits = 0
    threshold_splits = 0

    for i in range(1, len(texts)):
        sim = util.cos_sim(embeddings[i], embeddings[i - 1]).item()
        sims.append(sim)
        seg = texts[i]
        seg_len = len(seg)

        if not (sim > threshold and (curr_len + seg_len) <= max_chars):
            reasons = []
            if sim <= threshold:
                reasons.append(f"sim {sim:.4f} ≤ threshold {threshold}")
                threshold_splits += 1
            if curr_len + seg_len > max_chars:
                reasons.append(f"len({curr_len}+{seg_len}) > {max_chars}")
                max_char_splits += 1
            reason_str = " & ".join(reasons)
            split_log_msg = f"[!] split from {i}-{i+1} since {reason_str}"

            if split_log_file:
                with open(split_log_file, "a", encoding="utf-8") as f:
                    f.write(split_log_msg + "\n")
            else:
                print(split_log_msg)

            raw_chunks.append(current)
            current = seg
            curr_len = seg_len
        else:
            current += "\n" + seg
            curr_len += seg_len

    raw_chunks.append(current)
    print("[+] Added final chunk.")

    # Log the total number of max_char splits
    max_char_summary = f"[+] Performed {max_char_splits} max-char splits"
    print(max_char_summary)

    # Log the total number of threshold splits
    threshold_summary = f"[+] Performed {threshold_splits} threshold splits"
    print(threshold_summary)

    if sims:
        avg = statistics.mean(sims)
        std = statistics.pstdev(sims)
        print(f"[+] Cosine similarity — mean: {avg:.4f}, stddev: {std:.4f}")
    else:
        print("[+] No cosine similarities computed (too few segments).")

    return raw_chunks


def apply_min_chars(chunks, min_chars: int, split_log_file=None):
    merged = []
    merge_count = 0
    for idx, chunk in enumerate(chunks, start=1):
        length = len(chunk)
        if not merged:
            merged.append(chunk)
        elif length < min_chars:
            merge_log_msg = f"[!] merged chunk {idx} into previous since {length} < min_chars {min_chars}"
            
            if split_log_file:
                with open(split_log_file, "a", encoding="utf-8") as f:
                    f.write(merge_log_msg + "\n")
            else:
                print(merge_log_msg)
            
            merged[-1] += "\n" + chunk
            merge_count += 1
        else:
            merged.append(chunk)
    
    # Log the total number of merges
    merge_summary = f"[+] Performed {merge_count} min-char merges"
    print(merge_summary)
    
    return merged


def process_file(
    file_path,
    out_dir,
    model,
    threshold,
    max_chars,
    min_chars,
    batch_size,
    splitter,
    sentencizer=None,
    split_log_file=None,
):
    print(f"[+] Starting chunking on file: {file_path}")
    if not os.path.isfile(file_path):
        raise ValueError(f"Input path must be a file, not a directory: {file_path}")

    if split_log_file:
        with open(split_log_file, "w", encoding="utf-8") as f:
            f.write(f"Split log for file: {file_path}\n{'='*50}\n")
        print(f"[+] Split logs will be written to: {split_log_file}")

    print("[+] Loading JSON…")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[+] Loaded {len(data)} entries from JSON.")

    # sentence splitting
    sentences = []
    if splitter == "spacy":
        if sentencizer is None:
            sentencizer = load_spacy_sentencizer()
        print("[+] Splitting into sentences with spaCy…")
        for txt in data:
            for sent in sentencizer(txt).sents:
                s = sent.text.strip()
                if s:
                    sentences.append(s)

    elif splitter == "pysbd":
        if sentencizer is None:
            sentencizer = load_pysbd_segmenter()
        print("[+] Splitting into sentences with PySBD…")
        for txt in data:
            for s in sentencizer.segment(txt):
                s = s.strip()
                if s:
                    sentences.append(s)

    else:  # none
        sentences = [t.strip() for t in data if isinstance(t, str) and t.strip()]

    print(f"[+] Produced {len(sentences)} raw “sentences.”")

    # merge any too‐short sentences back into the previous one
    sentences = merge_short_sentences(sentences, min_length=16)
    print(f"[+] {len(sentences)} sentences after merging <16‑char fragments.")

    raw_chunks = semantic_chunk(
        sentences,
        model=model,
        threshold=threshold,
        max_chars=max_chars,
        batch_size=batch_size,
        split_log_file=split_log_file,
    )
    print(f"[+] Raw chunks count: {len(raw_chunks)}")

    if min_chars is not None:
        final_chunks = apply_min_chars(raw_chunks, min_chars, split_log_file=split_log_file)
        print(f"[+] Final chunks count after min_chars: {len(final_chunks)}")
    else:
        final_chunks = raw_chunks
        print("[+] Skipping min_chars merging (not specified)")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(file_path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_chunks, f, ensure_ascii=False, indent=2)
    print(f"[+] Wrote output to {out_path}")


def main():
    p = argparse.ArgumentParser(
        description="Semantic chunking of a single JSON list of text segments"
    )
    p.add_argument("input_file", help="Path to a JSON file containing a list of strings")
    p.add_argument("-o", "--output_dir", default="./", help="Directory to save chunked JSON")
    p.add_argument("-m", "--model", default="BAAI/bge-m3", help="SentenceTransformer model")
    p.add_argument("-d", "--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device for model (cuda or cpu)")
    p.add_argument("--fp16", action="store_true",
                   help="Load model in fp16 (half precision) if set")
    p.add_argument("-t", "--threshold", type=float, default=0.5,
                   help="Similarity threshold for merging segments")
    p.add_argument("-c", "--max-chars", type=int, default=8000,
                   help="Maximum characters per chunk")
    p.add_argument("-n", "--min-chars", type=int, default=None,
                   help="Minimum characters per chunk; shorter ones get merged back")
    p.add_argument("-b", "--batch-size", type=int, default=1,
                   help="Batch size for encoding")
    p.add_argument("--splitter", choices=["none", "spacy", "pysbd"], default="none",
                   help="Sentence‑splitter: none, spacy, or pysbd")
    p.add_argument("--log-file", help="Path to write split logs instead of printing")

    args = p.parse_args()

    print(f"[+] Loading model '{args.model}'…")
    if args.fp16:
        print("[+] Using fp16 with automatic device mapping")
        model = SentenceTransformer(
            args.model,
            trust_remote_code=True,
            use_auth_token=os.getenv("HF_TOKEN", None),
            model_kwargs={"torch_dtype": torch.float16, "device_map": "auto"},
            # tokenizer_kwargs={"truncation": True},
        )
    else:
        model = SentenceTransformer(
            args.model,
            device=args.device,
            use_auth_token=os.getenv("HF_TOKEN", None),
        )

    process_file(
        file_path=args.input_file,
        out_dir=args.output_dir,
        model=model,
        threshold=args.threshold,
        max_chars=args.max_chars,
        min_chars=args.min_chars,
        batch_size=args.batch_size,
        splitter=args.splitter,
        sentencizer=None,
        split_log_file=args.log_file,
    )


if __name__ == "__main__":
    main()
