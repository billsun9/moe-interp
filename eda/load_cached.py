from moe.activation_caching.cache_utils import OLMoEActivationExtractor
from eda_utils import plot_routing_softmax

extractor = OLMoEActivationExtractor()
ROOT_DIR = "/pmglocal/bys2107/research/data/OLMoE-acts/"
gsm8k_train = extractor.load_activations(ROOT_DIR + "gsm8k/gsm8k_train_questions_100samples.pt")
print(gsm8k_train.keys())
dataset = gsm8k_train['routing_weights']

print(len(dataset))
SAVE_DIR = "/pmglocal/bys2107/research/data/figs/"
# Single token raw distribution
plot_routing_softmax(dataset, layer_idx=1, sample_idx=0, token_idx=3, mode="single", save_path=SAVE_DIR + "single_token_all.png")

# Average over tokens in one sample (softmax weights)
plot_routing_softmax(dataset, layer_idx=1, sample_idx=0, mode="avg_sample", save_path=SAVE_DIR + "avg_over_one_sample.png")

# Average over tokens in one sample (softmax weights)
plot_routing_softmax(dataset, layer_idx=1, sample_idx=0, mode="avg_sample", top_k=1, save_path=SAVE_DIR + "avg_over_one_sample_topk=1.png")

# Average over tokens in one sample (softmax weights)
plot_routing_softmax(dataset, layer_idx=1, sample_idx=0, mode="avg_sample", top_k=8, save_path=SAVE_DIR + "avg_over_one_sample_topk=8.png")

# Single token, top-1 mask
plot_routing_softmax(
    dataset,
    layer_idx=1,
    sample_idx=0,
    token_idx=3,
    mode="single",
    top_k=1,
    save_path=SAVE_DIR + "single_token_top1.png"
)

# Single token, top-1 mask
plot_routing_softmax(
    dataset,
    layer_idx=1,
    sample_idx=0,
    token_idx=3,
    mode="single",
    top_k=8,
    save_path=SAVE_DIR + "single_token_top8.png"
)

print("Done")