# Core Objectives

Various Mechanistic Interpretability (MI) methods have gained traction as a means of understanding the internals of dense large language models (LLMs) like GPT2 and Gemma2. However, production LLMs like [GPT-OSS](https://openai.com/index/introducing-gpt-oss/) often adopt a Mixture-of-Experts (MoE) architecture, which has received significantly less attention from the MI community than their dense counterpart. 

We perform MI analysis on [OLMoE](https://arxiv.org/pdf/2409.02060), a 7b model which achieves competitive performance with industry-standard dense models. We focus on the following question: **How does OLMoE’s routing mechanism decide which 8 of 64 experts to activate, and what are the dynamics of these decisions on a per-layer and per-pre-router embedding basis**?

# File Structure
 - `activation_extraction`
   - `activation_extractor_utils.py` - pytorch hooks to save activations in OLMoE model
   - `data_utils.py` - load and format popular datasets (e.g. GSM8k, ARC)
   - `run_caching_utils.py` - utility functions to save activations on popular datasets or a custom list of texts
 - `activation_steering`  
   - `steer.py` - inject steering vector at specified token position or in all newly generated tokens, or perform activation patching
 - `eda`
   - `eda_utils.py` - (1) plot topK expert routing decisions for specified (layer_idx, sample_idx, token_idx) (although this plot is only meaningful within the same layer_idx), (2) for each expert, get the top prerouting activation embeddings mapping to it, and plot these using TSNE/PCA
 - `tests` - test each *_utils.py file