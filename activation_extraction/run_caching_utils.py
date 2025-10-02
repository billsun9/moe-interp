import os
from typing import Dict, List, Optional, Tuple, Union
from moe.activation_extraction.data_utils import load_and_format_dataset
from moe.activation_extraction.activation_extractor_utils import OLMoEActivationExtractor
from moe.path_config import MODEL_CACHE_DIR, SAVE_ACTS_PATH_DIR, DATASET_CACHE_DIR


class ActivationProcessor:
    def __init__(
        self,
        cache_dir: str = MODEL_CACHE_DIR,
        dataset_dir: str = DATASET_CACHE_DIR,
        base_save_dir: str = SAVE_ACTS_PATH_DIR,
        batch_size: int = 4,
        layer_idx: int = 10,
        extract_fn: str = "extract_activations_mlp_batch",
    ):
        """
        Args:
            cache_dir: Path to model cache
            dataset_dir: Path to dataset cache
            base_save_dir: Base directory for saving .pt files
            batch_size: Batch size for processing
            layer_idx: Model layer to extract from
            extract_fn: Extraction function to use
                        ("extract_activations_mlp_batch", 
                        "extract_topk_routing_batch", 
                        "extract_prerouting_and_combined_output_batch")
        """
        self.cache_dir = cache_dir
        self.dataset_dir = dataset_dir
        self.base_save_dir = base_save_dir
        self.batch_size = batch_size
        self.layer_idx = layer_idx
        self.extract_fn = extract_fn

        self.extractor = OLMoEActivationExtractor(cache_dir=cache_dir)
        if not self.extractor.load_model_and_tokenizer():
            raise RuntimeError("❌ Failed to load model/tokenizer.")

        if not hasattr(self.extractor, extract_fn):
            raise ValueError(f"Unknown extract_fn: {extract_fn}")
        self.extract_func = getattr(self.extractor, extract_fn)

    def _init_batch_container(
        self,
        dataset_name: str,
        split: str,
        question_only: Union[str, bool],
        metadata: Optional[Union[dict, str]],
    ) -> Dict:
        """Initialize the container for batch results depending on extract_fn."""
        container = {
            "dataset_name": dataset_name,
            "split": split,
            "question_only": question_only,
            "texts": [],
            "tokens": [],
            "metadata": metadata if metadata else "",
        }

        if self.extract_fn == "extract_activations_mlp_batch":
            container.update({
                "prerouting_activations": [],
                "router_logits": [],
                "router_probs": [],
                "topk_indices": [],
                "expert_outputs": [],
                "combined_output": []
            })
        elif self.extract_fn == "extract_topk_routing_batch":
            container.update({"topk_indices": [], "topk_scores": []})
        elif self.extract_fn == "extract_prerouting_and_combined_output_batch":
            container.update({"prerouting_activations": [], "combined_output": []})
        else:
            raise ValueError(f"Unsupported extract_fn: {self.extract_fn}")

        return container


    def process_sample_texts(
        self,
        sample_texts: List[str],
        dataset_name: str,
        metadata: Optional[Union[dict, str]] = None,
    ) -> str:
        """Process a list of custom sample texts."""
        num_samples = len(sample_texts)
        print(f"\n{'='*60}")
        print(f"Processing custom texts for {dataset_name} with {self.extract_fn}...")
        print(f"Total samples: {num_samples}, Batch size: {self.batch_size}")

        save_filename = f"L{self.layer_idx}_{dataset_name}_custom_{num_samples}samples.pt"
        save_path = os.path.join(self.base_save_dir, self.extract_fn, dataset_name, save_filename)

        all_batch_data = self._init_batch_container(
            dataset_name=dataset_name, split="", question_only="", metadata=metadata
        )

        try:
            batch_data = self.extract_func(
                sample_texts,
                layer_idx=self.layer_idx,
                batch_size=self.batch_size
            )
            for k in all_batch_data.keys():
                if k in batch_data:
                    all_batch_data[k].extend(batch_data[k])

            if not self.extractor.save_activations(all_batch_data, save_path):
                return ""
        except Exception as e:
            print(f"❌ Error processing custom texts: {e}")
            return ""

        return save_path

    def process_dataset(
        self,
        dataset_name: str,
        num_samples: int = 25,
        question_only: bool = False,
        metadata: Optional[Union[dict, str]] = None,
        splits: List[str] = ["train", "test"],
    ) -> Dict[str, str]:
        """Process a dataset split (train/test)."""
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name} dataset with {self.extract_fn}...")
        print(f"Samples per split: {num_samples}, Question only: {question_only}, Splits: {splits}")

        try:
            dataset_dict = load_and_format_dataset(
                dataset_name=dataset_name,
                question_only=question_only,
                cache_dir=self.dataset_dir,
                splits=splits,
            )
            print(f"✅ Loaded {dataset_name}")
        except Exception as e:
            print(f"❌ Failed to load dataset {dataset_name}: {e}")
            return {}

        save_paths = {}
        for split_name, texts in dataset_dict.items():
            print(f"\n--- {dataset_name} {split_name} split ---")
            texts_subset = texts[:num_samples]
            if not texts_subset:
                print(f"⚠️ No samples in {split_name}")
                continue

            suffix = "_questions" if question_only else "_full"
            save_filename = f"L{self.layer_idx}_{dataset_name}_{split_name}{suffix}_{len(texts_subset)}samples.pt"
            save_path = os.path.join(self.base_save_dir, self.extract_fn, dataset_name, save_filename)

            all_batch_data = self._init_batch_container(
                dataset_name=dataset_name,
                split=split_name,
                question_only=question_only,
                metadata=metadata,
            )

            batch_data = self.extract_func(texts_subset, self.layer_idx, batch_size=self.batch_size)
            for k in all_batch_data.keys():
                if k in batch_data:
                    all_batch_data[k].extend(batch_data[k])
            
            if self.extractor.save_activations(all_batch_data, save_path):
                save_paths[split_name] = save_path

        return save_paths

    def run_all(
        self,
        datasets_to_process: List[Tuple[str, bool]],
        num_samples: int = 100,
        splits: List[str] = ["train", "test"],
    ) -> Dict[str, Dict[str, str]]:
        """Run activation extraction across multiple datasets."""
        print("🚀 Running activation extraction on all datasets...")
        all_results = {}

        for dataset_name, question_only in datasets_to_process:
            try:
                save_paths = self.process_dataset(
                    dataset_name=dataset_name,
                    num_samples=num_samples,
                    question_only=question_only,
                    splits=splits,
                )
                key = f"{dataset_name}_{'questions' if question_only else 'full'}"
                all_results[key] = save_paths
            except Exception as e:
                print(f"❌ Failed {dataset_name} (question_only={question_only}): {e}")

        print(f"\n{'='*60}\n📊 EXTRACTION SUMMARY\n{'='*60}")
        for key, save_paths in all_results.items():
            print(f"\n{key}:")
            if save_paths:
                for split, path in save_paths.items():
                    print(f"  {split}: {path}")
            else:
                print("  ❌ No files saved")

        print(f"\n🎉 Completed {len(all_results)} dataset configs!")
        return all_results
