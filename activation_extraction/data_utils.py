from datasets import load_dataset
from typing import List, Tuple, Dict
import json

from moe.path_config import DATASET_CACHE_DIR

def load_and_format_dataset(
    dataset_name: str,
    subset: str = None,
    cache_dir: str = DATASET_CACHE_DIR,
    question_only: bool = False,
    word_limit: int = 250,
    splits: List[str] = ["train", "test"]
) -> Dict[str, List[str]]:
    """
    Load and format supported datasets.

    Args:
        dataset_name (str): Dataset key.
        subset (str): Optional Hugging Face subset
        cache_dir (str): HF dataset cache path
        question_only (bool): If True, return only the question/prompt
        word_limit (int): Max number of words in final formatted string
        splits (List[str]): train/test

    Returns:
        dict: {
            'split_name': [List of formatted strings]
        }
    """
    # === Dataset mapping ===
    dataset_map = {
        "gsm8k": ("openai/gsm8k", "main"),
        "arc-easy": ("allenai/ai2_arc", "ARC-Easy"),
        "arc-challenge": ("allenai/ai2_arc", "ARC-Challenge"),
        "sciq": ("allenai/sciq", "default"),
        "ag_news": ("fancyzhx/ag_news", "default"),
        "imdb_pos": ("stanfordnlp/imdb", "plain_text"),
        "imdb_neg": ("stanfordnlp/imdb", "plain_text"),
        "mbpp": ("mbpp", "full"),
        "poetry": ("suayptalha/Poetry-Foundation-Poems", "default"),
        "lex_glue": ("coastalcph/lex_glue", "case_hold"),
        "arxiv": ("etanios/arxiv-abstracts-full", "default"),
        "personas": ("allenai/tulu-3-sft-personas-instruction-following", "default")
    }

    assert dataset_name in dataset_map, f"Unsupported dataset: {dataset_name}"
    hf_name, hf_subset = dataset_map[dataset_name]
    dataset = load_dataset(hf_name, hf_subset if subset is None else subset, cache_dir=cache_dir)

    # === Helper: Truncate ===
    def truncate(text: str, limit: int) -> str:
        if limit is None:
            return text
        words = text.split()
        if len(words) <= limit:
            return text
        return " ".join(words[:limit]) + "..."

    # === Formatters ===
    def format_gsm8k(example, q_only=False):
        out = example['question'].strip() if q_only else f"Question: {example['question'].strip()}\nAnswer: {example['answer'].strip()}"
        return truncate(out, word_limit)

    def format_arc(example, q_only=False):
        choices = example['choices']
        choice_str = "\n".join(f"{label}. {text}" for label, text in zip(choices["label"], choices["text"]))
        out = example['question'].strip() if q_only else f"Question: {example['question'].strip()}\nChoices:\n{choice_str}\nAnswer: {example['answerKey']}"
        return truncate(out, word_limit)

    def format_sciq(example, q_only=False):
        out = example['question'].strip() if q_only else f"Question: {example['question'].strip()}\nReasoning: {example['support'].strip()}\nAnswer: {example['correct_answer']}"
        return truncate(out, word_limit)

    def format_imdb(example, q_only=False):
        review = example['text'].strip()
        out = f"{review}\nWhat is the sentiment of this review?" if q_only else review
        return truncate(out, word_limit)

    def format_ag_news(example, q_only=False):
        article = example['text'].strip()
        out = f"{article}\nWhat is the topic of this article?" if q_only else article
        return truncate(out, word_limit)

    def format_mbpp(example, q_only=False):
        text = example["text"].strip()
        code = example["code"].strip()
        out = text if q_only else f"Problem: {text}\n\nSolution:\n{code}"
        return truncate(out, word_limit)

    def format_poetry(example, q_only=False):
        return truncate(example["Poem"].strip(), word_limit)

    def format_lex_glue(example, q_only=False):
        return truncate(example["context"].strip(), word_limit)

    def format_arxiv(example, q_only=False):
        return truncate(example["abstract"].strip(), word_limit)

    def format_arxiv(example, q_only=False):
        return truncate(example["abstract"].strip(), word_limit)

    def format_personas(example, q_only=False):
        return truncate(example["prompt"].strip(), word_limit)

    # === Formatter selection ===
    if dataset_name == "gsm8k":
        formatter = lambda ex: format_gsm8k(ex, q_only=question_only)
    elif dataset_name in ["arc-easy", "arc-challenge"]:
        formatter = lambda ex: format_arc(ex, q_only=question_only)
    elif dataset_name == "sciq":
        formatter = lambda ex: format_sciq(ex, q_only=question_only)
    elif dataset_name == "ag_news":
        formatter = lambda ex: format_ag_news(ex, q_only=question_only)
    elif dataset_name in ["imdb_pos", "imdb_neg"]:
        target_label = 1 if dataset_name == "imdb_pos" else 0
        formatter = lambda ex: format_imdb(ex, q_only=question_only)
        return {
            split: [formatter(ex) for ex in dataset[split] if ex["label"] == target_label]
            for split in dataset.keys() if split in splits
        }
    elif dataset_name == "mbpp":
        formatter = lambda ex: format_mbpp(ex, q_only=question_only)
    elif dataset_name == "poetry":
        formatter = lambda ex: format_poetry(ex)
    elif dataset_name == "lex_glue":
        formatter = lambda ex: format_lex_glue(ex)
    elif dataset_name == "arxiv":
        formatter = lambda ex: format_arxiv(ex)
    elif dataset_name == "personas":
        formatter = lambda ex: format_personas(ex)
    else:
        raise ValueError(f"No formatter defined for dataset {dataset_name}")

    return {
        split: [formatter(ex) for ex in dataset[split]]
        for split in dataset.keys() if split in splits
    }

    