import os
import torch

from moe.activation_extraction.activation_extractor_utils import OLMoEActivationExtractor
from moe.eda.eda_utils import *
from moe.path_config import SAVE_FIGS_PATH_DIR, SAVE_ACTS_PATH_DIR

if __name__ == "__main__":
    extractor = OLMoEActivationExtractor()
    extractor.load_model_and_tokenizer()
    for param in extractor.model.parameters():
        print(param.device)
        break  # only need to check the first param usually
    gsm8k_train = extractor.load_activations(os.path.join(SAVE_ACTS_PATH_DIR + "gsm8k/gsm8k_train_questions_20samples.pt"))
    gsm8k_test = extractor.load_activations(os.path.join(SAVE_ACTS_PATH_DIR + "gsm8k/gsm8k_test_questions_20samples.pt"))

    dataset = gsm8k_train
    # Save directory
    save_dir = SAVE_FIGS_PATH_DIR

    plot_routing_histogram(dataset, [(8, 0, 5), (8, 0, 10), (8, 0, 11)], save_dir)
    plot_routing_histogram(dataset, [(8, 0, 5), (8, 0, 10), (8, 0, 11)], save_dir, top_k=8)
    plot_routing_histogram(dataset, [(8, 0, 5), (8, 0, 10), (8, 0, 11)], save_dir, softmax=False, top_k=8)

    plot_routing_histogram(dataset, [(8, 0, 5), (8, 0, 10), (8, 0, 11)], save_dir, stacked=True)
    plot_routing_histogram(dataset, [(8, 0, 5), (8, 0, 10), (8, 0, 11)], save_dir, top_k=8, stacked=True)
    plot_routing_histogram(dataset, [(8, 0, 5), (8, 0, 10), (8, 0, 11)], save_dir, softmax=False, top_k=8, stacked=True)
    # layer_idx = 6
    # d = get_top_embeddings(dataset, layer_idx, list(range(len(dataset['texts']))))
    # print(d.keys())
    # print(len(d.keys()))
    # hydrated = []
    # for i in [18, 37, 54, 63]:
    #     hydrated_samples = hydrate_batch(dataset, d[i][:100])
    #     hydrated.extend(hydrated_samples)
    # plot_expert_embeddings(hydrated, d, save_path = "/local/bys2107/research/data/figs/pca_L6_E18E37E54E63.png")
    # plot_expert_embeddings(hydrated, d, save_path = "/local/bys2107/research/data/figs/tsne_L6_E18E37E54E63.png", method="tsne")
    # datasets = {
    #     "gsm8k": gsm8k_train,
    #     "sciq_train": sciq_train,
    #     "sciq_test": sciq_test,
    # }

    # sample_indices = {
    #     "gsm8k": [0,1,2,3],
    #     "sciq_train": [0,1,2,3],
    #     "sciq_test": [0,1,2,3]
    # }
    
    # expert_map = get_top_embeddings_multiple(datasets, layer_idx=5, sample_indices=sample_indices, top_k=8)
    
    # to_hydrate = []
    # for i in [18, 37, 54, 63]:
    #     to_hydrate.extend(expert_map[i][:25])
    
    # hydrated = hydrate_multiple(datasets, to_hydrate)
    # plot_expert_embeddings(hydrated, expert_map, save_path="experts_multi.png", method="pca")