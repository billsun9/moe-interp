# Core Objectives

Various Mechanistic Interpretability (MI) methods have gained traction as a means of understanding the internals of dense large language models (LLMs) like GPT2 and Gemma2. However, production LLMs like [GPT-OSS](https://openai.com/index/introducing-gpt-oss/) often adopt a Mixture-of-Experts (MoE) architecture, which has received significantly less attention from the MI community than their dense counterpart. 

We perform MI analysis on [OLMoE](https://arxiv.org/pdf/2409.02060), a 7b MoE model which achieves competitive performance with industry-standard dense models. We focus on the following question: **How does OLMoE’s routing mechanism decide which 8 of 64 experts to activate, and what are the dynamics of these decisions on a per-layer and per-pre-router embedding basis**?

To support data analysis, we create a [website](https://moe-server-u963.onrender.com/) showing the top activations for specific expert index sets. For instance, we can aggregate all tokens which have as the first 2 experts [{25, 32}](https://moe-server-u963.onrender.com/expert/25/32) then see the top tokens + context across 12 diverse datasets. [Website PRD](https://docs.google.com/document/d/1hzaWTL745JdpI3Lh8M8vFudS0Z-bYrsGsMGw8Vqedrk/edit?usp=sharing)

# File Structure
 - `activation_extraction`
   - `activation_extractor_utils.py` - pytorch hooks to save activations in OLMoE model
   - `data_utils.py` - load and format popular datasets (e.g. GSM8k, ARC)
   - `run_caching_utils.py` - utility functions to save activations on popular datasets or a custom list of texts
 - `activation_steering`  
   - `steer.py` - inject steering vector at specified token position or in all newly generated tokens, or perform activation patching. WIP
 - `annotator`
   - `annotator.py` - Auto-annotation using OLMoE to describe what a set of tokens semantically corresponds to. This doesn't work very well
 - `eda`
   - `eda_utils.py` - (1) plot topK expert routing decisions for specified (layer_idx, sample_idx, token_idx) (although this plot is only meaningful within the same layer_idx), (2) for each expert, get the top prerouting activation embeddings mapping to it, and plot these using TSNE/PCA
   - `utils.py` - Other utility functions: load_all_activations, compute_per_dataset_token_counts, compute_expert_set_counts, find_expert_coactivations, format_expert_coactivations
   - `feature_finder_utils.py` - find_dataset_specific_expert_sets
 - `snmf` - Copy of "Semi-Nonnegative Matix Factorization" repo by Or Sharfan
 - `tests` - test each *_utils.py file