import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from collections import defaultdict, Counter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class MoERoutingAnalyzer:
    """
    Comprehensive analysis toolkit for MoE routing dynamics.
    """
    
    def __init__(self, 
                 num_experts: int = 64, 
                 top_k: int = 8,
                 num_layers: int = 16):
        """
        Initialize the analyzer.
        
        Args:
            num_experts: Total number of experts per layer
            top_k: Number of activated experts per token
            num_layers: Number of transformer layers
        """
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_layers = num_layers
        self.datasets = {}
        
    def load_dataset(self, name: str, data: Dict) -> None:
        """Load a dataset for analysis."""
        self.datasets[name] = data
        print(f"✅ Loaded {name}: {len(data['texts'])} samples, {len(data['prerouting_logits'][0])} layers")
    
    def extract_top_k_experts(self, routing_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract top-k expert indices and their probabilities from routing weights.
        
        Args:
            routing_weights: Tensor of shape [batch, seq_len, num_experts]
            
        Returns:
            top_k_indices: Indices of top-k experts [batch, seq_len, top_k]
            top_k_probs: Probabilities of top-k experts [batch, seq_len, top_k]
        """
        # Apply softmax to get probabilities
        routing_probs = torch.softmax(routing_weights, dim=-1)
        
        # Get top-k experts
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        
        return top_k_indices, top_k_probs
    
    def compute_expert_usage_stats(self, dataset_name: str) -> Dict:
        """
        Compute comprehensive expert usage statistics for a dataset.
        
        Args:
            dataset_name: Name of the dataset to analyze
            
        Returns:
            Dictionary containing various usage statistics
        """
        data = self.datasets[dataset_name]
        
        stats = {
            'expert_usage_by_layer': defaultdict(Counter),
            'expert_usage_total': Counter(),
            'expert_selection_entropy': [],
            'layer_entropy': [],
            'position_entropy': [],
            'expert_probability_distributions': defaultdict(list)
        }
        
        for sample_idx, routing_weights_sample in enumerate(data['routing_weights']):
            for layer_idx, routing_weights in enumerate(routing_weights_sample):
                # routing_weights shape: [1, seq_len, num_experts]
                routing_weights = routing_weights.squeeze(0)  # [seq_len, num_experts]
                
                top_k_indices, top_k_probs = self.extract_top_k_experts(routing_weights.unsqueeze(0))
                top_k_indices = top_k_indices.squeeze(0)  # [seq_len, top_k]
                top_k_probs = top_k_probs.squeeze(0)      # [seq_len, top_k]
                
                # Count expert usage
                for pos in range(top_k_indices.shape[0]):
                    for k in range(self.top_k):
                        expert_id = top_k_indices[pos, k].item()
                        prob = top_k_probs[pos, k].item()
                        
                        stats['expert_usage_by_layer'][layer_idx][expert_id] += 1
                        stats['expert_usage_total'][expert_id] += 1
                        stats['expert_probability_distributions'][expert_id].append(prob)
                
                # Compute entropy measures
                routing_probs = torch.softmax(routing_weights, dim=-1)
                
                # Expert selection entropy (per position)
                pos_entropies = []
                for pos in range(routing_probs.shape[0]):
                    entropy = -torch.sum(routing_probs[pos] * torch.log(routing_probs[pos] + 1e-10))
                    pos_entropies.append(entropy.item())
                
                stats['expert_selection_entropy'].extend(pos_entropies)
                stats['layer_entropy'].append(np.mean(pos_entropies))
        
        # Compute summary statistics
        stats['expert_usage_variance'] = np.var(list(stats['expert_usage_total'].values()))
        stats['expert_usage_gini'] = self._compute_gini_coefficient(list(stats['expert_usage_total'].values()))
        
        return stats
    
    def _compute_gini_coefficient(self, values: List[float]) -> float:
        """Compute Gini coefficient for measuring inequality in expert usage."""
        values = np.array(values)
        n = len(values)
        if n == 0:
            return 0
        values = np.sort(values)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n
    
    def analyze_expert_specialization(self, dataset_name: str) -> Dict:
        """
        Analyze whether experts specialize in certain types of content.
        
        Returns:
            Dictionary containing specialization metrics
        """
        data = self.datasets[dataset_name]
        
        # Group samples by content characteristics
        content_groups = self._categorize_content(data['texts'])
        
        specialization_stats = {}
        
        for layer_idx in range(self.num_layers):
            layer_specialization = defaultdict(lambda: defaultdict(int))
            
            for sample_idx, content_category in enumerate(content_groups):
                routing_weights = data['routing_weights'][sample_idx][layer_idx].squeeze(0)
                top_k_indices, _ = self.extract_top_k_experts(routing_weights.unsqueeze(0))
                top_k_indices = top_k_indices.squeeze(0)
                
                # Count expert usage by content category
                for pos in range(top_k_indices.shape[0]):
                    for k in range(self.top_k):
                        expert_id = top_k_indices[pos, k].item()
                        layer_specialization[content_category][expert_id] += 1
            
            specialization_stats[layer_idx] = dict(layer_specialization)
        
        return specialization_stats
    
    def _categorize_content(self, texts: List[str]) -> List[str]:
        """
        Categorize text content (basic implementation - can be enhanced).
        """
        categories = []
        for text in texts:
            text_lower = text.lower()
            if any(word in text_lower for word in ['calculate', 'solve', 'number', 'math', '+', '-', '*', '/']):
                categories.append('mathematical')
            elif any(word in text_lower for word in ['which', 'what', 'where', 'who', 'choice']):
                categories.append('factual_qa')
            elif len(text.split()) > 20:
                categories.append('long_form')
            else:
                categories.append('short_form')
        return categories
    
    def analyze_positional_routing(self, dataset_name: str) -> Dict:
        """
        Analyze how routing patterns change across token positions.
        """
        data = self.datasets[dataset_name]
        
        position_stats = defaultdict(lambda: defaultdict(Counter))
        
        for sample_idx, routing_weights_sample in enumerate(data['routing_weights']):
            seq_len = routing_weights_sample[0].shape[1]
            
            for layer_idx, routing_weights in enumerate(routing_weights_sample):
                routing_weights = routing_weights.squeeze(0)
                top_k_indices, _ = self.extract_top_k_experts(routing_weights.unsqueeze(0))
                top_k_indices = top_k_indices.squeeze(0)
                
                for pos in range(seq_len):
                    # Normalize position (0-1 scale)
                    normalized_pos = pos / (seq_len - 1) if seq_len > 1 else 0
                    pos_bucket = int(normalized_pos * 10) / 10  # Bucket to 0.0, 0.1, 0.2, ...
                    
                    for k in range(self.top_k):
                        expert_id = top_k_indices[pos, k].item()
                        position_stats[layer_idx][pos_bucket][expert_id] += 1
        
        return dict(position_stats)
    
    def compare_datasets_routing(self) -> Dict:
        """
        Compare routing patterns across different datasets.
        """
        if len(self.datasets) < 2:
            print("⚠️ Need at least 2 datasets for comparison")
            return {}
        
        comparison = {}
        
        for dataset1 in self.datasets.keys():
            for dataset2 in self.datasets.keys():
                if dataset1 >= dataset2:  # Avoid duplicate comparisons
                    continue
                
                comparison_key = f"{dataset1}_vs_{dataset2}"
                
                # Compute expert usage overlap
                stats1 = self.compute_expert_usage_stats(dataset1)
                stats2 = self.compute_expert_usage_stats(dataset2)
                
                # Jaccard similarity of most used experts
                top_experts1 = set([expert for expert, _ in stats1['expert_usage_total'].most_common(20)])
                top_experts2 = set([expert for expert, _ in stats2['expert_usage_total'].most_common(20)])
                
                jaccard_sim = len(top_experts1 & top_experts2) / len(top_experts1 | top_experts2)
                
                # KL divergence between expert usage distributions
                usage1 = np.array([stats1['expert_usage_total'][i] for i in range(self.num_experts)])
                usage2 = np.array([stats2['expert_usage_total'][i] for i in range(self.num_experts)])
                
                # Normalize to probabilities
                usage1 = usage1 / usage1.sum() + 1e-10
                usage2 = usage2 / usage2.sum() + 1e-10
                
                kl_div = np.sum(usage1 * np.log(usage1 / usage2))
                
                comparison[comparison_key] = {
                    'jaccard_similarity': jaccard_sim,
                    'kl_divergence': kl_div,
                    'entropy_diff': np.mean(stats1['expert_selection_entropy']) - np.mean(stats2['expert_selection_entropy'])
                }
        
        return comparison
    
    def visualize_expert_usage_heatmap(self, dataset_name: str, save_path: Optional[str] = None):
        """
        Create a heatmap showing expert usage across layers.
        """
        stats = self.compute_expert_usage_stats(dataset_name)
        
        # Create usage matrix [layers x experts]
        usage_matrix = np.zeros((self.num_layers, self.num_experts))
        
        for layer_idx in range(self.num_layers):
            for expert_id, count in stats['expert_usage_by_layer'][layer_idx].items():
                usage_matrix[layer_idx, expert_id] = count
        
        # Normalize by row (layer) for better visualization
        usage_matrix_norm = usage_matrix / (usage_matrix.sum(axis=1, keepdims=True) + 1e-10)
        
        plt.figure(figsize=(15, 8))
        sns.heatmap(usage_matrix_norm, 
                   xticklabels=range(self.num_experts),
                   yticklabels=range(self.num_layers),
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Normalized Usage Frequency'})
        
        plt.title(f'Expert Usage Heatmap - {dataset_name}')
        plt.xlabel('Expert ID')
        plt.ylabel('Layer')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return usage_matrix_norm
    
    def visualize_routing_entropy_evolution(self, save_path: Optional[str] = None):
        """
        Visualize how routing entropy changes across layers for different datasets.
        """
        plt.figure(figsize=(12, 6))
        
        for dataset_name in self.datasets.keys():
            stats = self.compute_expert_usage_stats(dataset_name)
            
            # Compute average entropy per layer
            layer_entropies = []
            for layer_idx in range(self.num_layers):
                layer_entropy = np.mean([entropy for i, entropy in enumerate(stats['expert_selection_entropy']) 
                                       if i % self.num_layers == layer_idx])
                layer_entropies.append(layer_entropy)
            
            plt.plot(range(self.num_layers), layer_entropies, 
                    marker='o', label=dataset_name, linewidth=2)
        
        plt.xlabel('Layer Index')
        plt.ylabel('Average Routing Entropy')
        plt.title('Routing Entropy Evolution Across Layers')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_expert_specialization_interactive(self, dataset_name: str, layer_idx: int = 8):
        """
        Create an interactive visualization of expert specialization.
        """
        specialization = self.analyze_expert_specialization(dataset_name)
        
        if layer_idx not in specialization:
            print(f"⚠️ Layer {layer_idx} not found in specialization data")
            return
        
        # Prepare data for plotting
        categories = list(specialization[layer_idx].keys())
        experts = list(range(self.num_experts))
        
        data = []
        for category in categories:
            for expert_id in experts:
                count = specialization[layer_idx][category].get(expert_id, 0)
                data.append({
                    'Category': category,
                    'Expert': expert_id,
                    'Usage Count': count
                })
        
        df = pd.DataFrame(data)
        
        # Create pivot table for heatmap
        pivot_df = df.pivot(index='Category', columns='Expert', values='Usage Count').fillna(0)
        
        # Create interactive heatmap
        fig = px.imshow(pivot_df.values, 
                       x=pivot_df.columns, 
                       y=pivot_df.index,
                       color_continuous_scale='Viridis',
                       title=f'Expert Specialization by Content Category (Layer {layer_idx})')
        
        fig.update_layout(
            xaxis_title='Expert ID',
            yaxis_title='Content Category',
            width=1000,
            height=400
        )
        
        fig.show()
        
        return pivot_df
    
    def analyze_routing_consistency(self, dataset_name: str, similarity_threshold: float = 0.8) -> Dict:
        """
        Analyze consistency of routing decisions for similar inputs.
        """
        data = self.datasets[dataset_name]
        
        # Find pairs of similar texts (basic similarity based on length and word overlap)
        similar_pairs = []
        texts = data['texts']
        
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                # Simple similarity metric (can be enhanced with embeddings)
                words1 = set(texts[i].lower().split())
                words2 = set(texts[j].lower().split())
                
                if len(words1) > 0 and len(words2) > 0:
                    jaccard_sim = len(words1 & words2) / len(words1 | words2)
                    if jaccard_sim > similarity_threshold:
                        similar_pairs.append((i, j, jaccard_sim))
        
        consistency_stats = {
            'similar_pairs_found': len(similar_pairs),
            'routing_similarities': [],
            'layer_consistency': defaultdict(list)
        }
        
        for i, j, text_sim in similar_pairs:
            for layer_idx in range(self.num_layers):
                # Get routing weights for both samples
                routing1 = data['routing_weights'][i][layer_idx].squeeze(0)
                routing2 = data['routing_weights'][j][layer_idx].squeeze(0)
                
                # Compute average routing similarity across positions
                routing_probs1 = torch.softmax(routing1, dim=-1)
                routing_probs2 = torch.softmax(routing2, dim=-1)
                
                # Cosine similarity between routing distributions
                cos_sim = torch.nn.functional.cosine_similarity(
                    routing_probs1.mean(dim=0),
                    routing_probs2.mean(dim=0),
                    dim=0
                ).item()
                
                consistency_stats['routing_similarities'].append(cos_sim)
                consistency_stats['layer_consistency'][layer_idx].append(cos_sim)
        
        # Compute summary statistics
        if consistency_stats['routing_similarities']:
            consistency_stats['mean_consistency'] = np.mean(consistency_stats['routing_similarities'])
            consistency_stats['std_consistency'] = np.std(consistency_stats['routing_similarities'])
        
        return consistency_stats
    
    def generate_comprehensive_report(self, save_dir: str = "./moe_analysis_report/"):
        """
        Generate a comprehensive analysis report with all visualizations.
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("📊 Generating comprehensive MoE routing analysis report...")
        
        # 1. Expert usage analysis
        print("1. Analyzing expert usage patterns...")
        for dataset_name in self.datasets.keys():
            stats = self.compute_expert_usage_stats(dataset_name)
            
            # Save usage statistics
            with open(f"{save_dir}/{dataset_name}_usage_stats.txt", 'w') as f:
                f.write(f"Expert Usage Statistics for {dataset_name}\n")
                f.write("="*50 + "\n\n")
                f.write(f"Expert Usage Variance: {stats['expert_usage_variance']:.4f}\n")
                f.write(f"Expert Usage Gini Coefficient: {stats['expert_usage_gini']:.4f}\n")
                f.write(f"Mean Selection Entropy: {np.mean(stats['expert_selection_entropy']):.4f}\n\n")
                
                f.write("Top 10 Most Used Experts:\n")
                for expert_id, count in stats['expert_usage_total'].most_common(10):
                    f.write(f"Expert {expert_id}: {count} activations\n")
            
            # Generate heatmap
            self.visualize_expert_usage_heatmap(
                dataset_name, 
                save_path=f"{save_dir}/{dataset_name}_usage_heatmap.png"
            )
        
        # 2. Cross-dataset comparison
        print("2. Comparing routing patterns across datasets...")
        comparison = self.compare_datasets_routing()
        
        with open(f"{save_dir}/dataset_comparison.txt", 'w') as f:
            f.write("Dataset Routing Pattern Comparison\n")
            f.write("="*40 + "\n\n")
            
            for comparison_key, metrics in comparison.items():
                f.write(f"{comparison_key}:\n")
                f.write(f"  Jaccard Similarity: {metrics['jaccard_similarity']:.4f}\n")
                f.write(f"  KL Divergence: {metrics['kl_divergence']:.4f}\n")
                f.write(f"  Entropy Difference: {metrics['entropy_diff']:.4f}\n\n")
        
        # 3. Entropy evolution
        print("3. Analyzing entropy evolution...")
        self.visualize_routing_entropy_evolution(
            save_path=f"{save_dir}/entropy_evolution.png"
        )
        
        # 4. Consistency analysis
        print("4. Analyzing routing consistency...")
        for dataset_name in self.datasets.keys():
            consistency = self.analyze_routing_consistency(dataset_name)
            
            with open(f"{save_dir}/{dataset_name}_consistency.txt", 'w') as f:
                f.write(f"Routing Consistency Analysis for {dataset_name}\n")
                f.write("="*50 + "\n\n")
                f.write(f"Similar pairs found: {consistency['similar_pairs_found']}\n")
                if 'mean_consistency' in consistency:
                    f.write(f"Mean routing consistency: {consistency['mean_consistency']:.4f}\n")
                    f.write(f"Std routing consistency: {consistency['std_consistency']:.4f}\n")
        
        print(f"✅ Comprehensive report generated in {save_dir}")

# Usage example and utility functions
def quick_analysis_pipeline(datasets_dict: Dict[str, Dict]) -> MoERoutingAnalyzer:
    """
    Quick analysis pipeline for loaded datasets.
    
    Args:
        datasets_dict: Dictionary mapping dataset names to data dictionaries
        
    Returns:
        Configured analyzer with loaded datasets
    """
    analyzer = MoERoutingAnalyzer()
    
    # Load all datasets
    for name, data in datasets_dict.items():
        analyzer.load_dataset(name, data)
    
    # Run basic analysis
    print("🔍 Running expert usage analysis...")
    for dataset_name in datasets_dict.keys():
        stats = analyzer.compute_expert_usage_stats(dataset_name)
        print(f"\n{dataset_name} Statistics:")
        print(f"  Expert usage Gini coefficient: {stats['expert_usage_gini']:.3f}")
        print(f"  Mean routing entropy: {np.mean(stats['expert_selection_entropy']):.3f}")
        print(f"  Most used expert: {stats['expert_usage_total'].most_common(1)[0]}")
    
    print("\n🎯 Running cross-dataset comparison...")
    comparison = analyzer.compare_datasets_routing()
    for comp_name, metrics in comparison.items():
        print(f"{comp_name}: Jaccard={metrics['jaccard_similarity']:.3f}, KL={metrics['kl_divergence']:.3f}")
    
    return analyzer