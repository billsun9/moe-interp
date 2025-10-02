# Semi Non-Negative Matrix Factorization on OLMoE Activations

### What's this?
> Current methods rely on dictionary learning with sparse autoencoders (SAEs), commonly trained over residual stream activations to learn directions from scratch. However, SAEs often struggle in causal evaluations and lack intrinsic interpretability, as their learning is not explicitly tied to the computations of the model. Here, we tackle these limitations by directly decomposing MLP activations with semi-nonnegative matrix factorization (SNMF), such that the learned features are (a) sparse linear combinations of co-activated neurons, and (b) mapped to their activating inputs, making them directly interpretable.

This is a copy of [Or Shafran's repo](https://github.com/ordavid-s/snmf-mlp-decomposition) and replicates some of their experiments on an MoE block output