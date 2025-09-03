import os
import matplotlib.pyplot as plt
import numpy as np
from moe.activation_caching.cache_utils import OLMoEActivationExtractor
from moe_routing_analysis import MoERoutingAnalyzer, quick_analysis_pipeline

def load_your_datasets():
    """
    Load your existing datasets following your current code pattern.
    """
    extractor = OLMoEActivationExtractor()
    ROOT_DIR = "/pmglocal/bys2107/research/data/OLMoE-acts/"
    
    print("📂 Loading datasets...")
    print("Available directories:", os.listdir(ROOT_DIR))
    
    # Load the three datasets
    datasets = {}
    
    try:
        datasets['gsm8k'] = extractor.load_activations(
            ROOT_DIR + "gsm8k/gsm8k_train_questions_50samples.pt"
        )
        print("✅ Loaded GSM8K")
    except Exception as e:
        print(f"❌ Failed to load GSM8K: {e}")
    
    try:
        datasets['arc_easy'] = extractor.load_activations(
            ROOT_DIR + "arc-easy/arc-easy_train_questions_50samples.pt"
        )
        print("✅ Loaded ARC-Easy")
    except Exception as e:
        print(f"❌ Failed to load ARC-Easy: {e}")
    
    try:
        datasets['arc_challenge'] = extractor.load_activations(
            ROOT_DIR + "arc-challenge/arc-challenge_train_questions_50samples.pt"
        )
        print("✅ Loaded ARC-Challenge")
    except Exception as e:
        print(f"❌ Failed to load ARC-Challenge: {e}")
    
    return datasets

def run_comprehensive_analysis():
    """
    Run the comprehensive analysis pipeline on your datasets.
    """
    print("🚀 Starting comprehensive MoE routing analysis...")
    
    # Load datasets
    datasets = load_your_datasets()
    
    if not datasets:
        print("❌ No datasets loaded. Exiting.")
        return
    
    # Quick analysis pipeline
    analyzer = quick_analysis_pipeline(datasets)
    
    # Generate detailed visualizations
    print("\n📊 Generating detailed visualizations...")
    
    # 1. Expert usage heatmaps for each dataset
    for dataset_name in datasets.keys():
        print(f"Creating heatmap for {dataset_name}...")
        analyzer.visualize_expert_usage_heatmap(dataset_name)
    
    # 2. Entropy evolution across layers
    print("Analyzing entropy evolution...")
    analyzer.visualize_routing_entropy_evolution()
    
    # 3. Expert specialization analysis (interactive - will show in browser/notebook)
    for dataset_name in datasets.keys():
        print(f"Analyzing specialization for {dataset_name}...")
        specialization_data = analyzer.visualize_expert_specialization_interactive(
            dataset_name, layer_idx=8  # Middle layer
        )
        print(f"Specialization matrix shape: {specialization_data.shape}")
    
    # 4. Detailed consistency analysis
    print("\n🔍 Running detailed consistency analysis...")
    for dataset_name in datasets.keys():
        consistency = analyzer.analyze_routing_consistency(dataset_name, similarity_threshold=0.3)
        print(f"\n{dataset_name} Consistency:")
        print(f"  Similar pairs found: {consistency['similar_pairs_found']}")
        if 'mean_consistency' in consistency:
            print(f"  Mean consistency: {consistency['mean_consistency']:.4f}")
    
    # 5. Generate comprehensive report
    print("\n📋 Generating comprehensive report...")
    analyzer.generate_comprehensive_report(save_dir="./moe_routing_report/")
    
    return analyzer

def focused_expert_analysis(datasets, expert_id: int = 32, layer_id: int = 8):
    """
    Deep dive analysis of a specific expert across datasets.
    
    Args:
        datasets: Loaded dataset dictionary
        expert_id: Specific expert to analyze (0-63)
        layer_id: Specific layer to focus on (0-15)
    """
    print(f"🎯 Deep dive analysis of Expert {expert_id} at Layer {layer_id}")
    
    analyzer = MoERoutingAnalyzer()
    for name, data in datasets.items():
        analyzer.load_dataset(name, data)
    
    # Analyze this specific expert
    expert_analysis = {}
    
    for dataset_name, data in datasets.items():
        expert_activations = []
        expert_probabilities = []
        
        for sample_idx, routing_weights_sample in enumerate(data['routing_weights']):
            if layer_id < len(routing_weights_sample):
                routing_weights = routing_weights_sample[layer_id].squeeze(0)
                
                # Get top-k experts and probabilities
                top_k_indices, top_k_probs = analyzer.extract_top_k_experts(routing_weights.unsqueeze(0))
                top_k_indices = top_k_indices.squeeze(0)
                top_k_probs = top_k_probs.squeeze(0)
                
                # Check if our expert was selected
                for pos in range(top_k_indices.shape[0]):
                    expert_mask = (top_k_indices[pos] == expert_id)
                    if expert_mask.any():
                        expert_prob = top_k_probs[pos][expert_mask].max().item()
                        expert_probabilities.append(expert_prob)
                        expert_activations.append((sample_idx, pos, expert_prob))
        
        expert_analysis[dataset_name] = {
            'activation_count': len(expert_activations),
            'mean_probability': np.mean(expert_probabilities) if expert_probabilities else 0,
            'std_probability': np.std(expert_probabilities) if expert_probabilities else 0,
            'activations': expert_activations
        }
    
    # Visualize expert comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    datasets_names = list(expert_analysis.keys())
    activation_counts = [expert_analysis[name]['activation_count'] for name in datasets_names]
    plt.bar(datasets_names, activation_counts)
    plt.title(f'Expert {expert_id} Activation Count (Layer {layer_id})')
    plt.ylabel('Number of Activations')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    mean_probs = [expert_analysis[name]['mean_probability'] for name in datasets_names]
    std_probs = [expert_analysis[name]['std_probability'] for name in datasets_names]
    plt.bar(datasets_names, mean_probs, yerr=std_probs, capsize=5)
    plt.title(f'Expert {expert_id} Mean Probability (Layer {layer_id})')
    plt.ylabel('Mean Routing Probability')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis
    print(f"\nDetailed Analysis for Expert {expert_id}:")
    print("=" * 50)
    
    for dataset_name, analysis in expert_analysis.items():
        print(f"\n{dataset_name}:")
        print(f"  Activations: {analysis['activation_count']}")
        print(f"  Mean probability: {analysis['mean_probability']:.4f}")
        print(f"  Std probability: {analysis['std_probability']:.4f}")
        
        if analysis['activations']:
            print(f"  Sample activations (first 5):")
            for i, (sample_idx, pos, prob) in enumerate(analysis['activations'][:5]):
                text_preview = datasets[dataset_name]['texts'][sample_idx][:50] + "..."
                print(f"    Sample {sample_idx}, Pos {pos}: {prob:.3f} - '{text_preview}'")
    
    return expert_analysis

def routing_pattern_clustering(datasets, layer_id: int = 8):
    """
    Cluster samples based on their routing patterns to find similar routing behaviors.
    
    Args:
        datasets: Loaded dataset dictionary
        layer_id: Layer to analyze routing patterns
    """
    print(f"🔬 Clustering routing patterns at Layer {layer_id}")
    
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    analyzer = MoERoutingAnalyzer()
    
    # Collect all routing patterns
    all_routing_patterns = []
    all_labels = []
    all_texts = []
    
    for dataset_name, data in datasets.items():
        analyzer.load_dataset(dataset_name, data)
        
        for sample_idx, routing_weights_sample in enumerate(data['routing_weights']):
            if layer_id < len(routing_weights_sample):
                routing_weights = routing_weights_sample[layer_id].squeeze(0)
                
                # Average routing pattern across sequence
                routing_probs = torch.softmax(routing_weights, dim=-1)
                avg_routing = routing_probs.mean(dim=0).numpy()  # [num_experts]
                
                all_routing_patterns.append(avg_routing)
                all_labels.append(dataset_name)
                all_texts.append(data['texts'][sample_idx])
    
    # Convert to numpy array
    routing_patterns = np.array(all_routing_patterns)  # [n_samples, num_experts]
    
    print(f"Collected {routing_patterns.shape[0]} routing patterns")
    
    # Perform clustering
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(routing_patterns)
    
    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    routing_2d = pca.fit_transform(routing_patterns)
    
    # Visualize clusters
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Clusters colored by dataset
    plt.subplot(1, 3, 1)
    colors = {'gsm8k': 'red', 'arc_easy': 'blue', 'arc_challenge': 'green'}
    for dataset_name in datasets.keys():
        mask = np.array(all_labels) == dataset_name
        if mask.any():
            plt.scatter(routing_2d[mask, 0], routing_2d[mask, 1], 
                       c=colors.get(dataset_name, 'gray'), 
                       label=dataset_name, alpha=0.6)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
    plt.title('Routing Patterns by Dataset')
    plt.legend()
    
    # Plot 2: Clusters colored by cluster assignment
    plt.subplot(1, 3, 2)
    scatter = plt.scatter(routing_2d[:, 0], routing_2d[:, 1], 
                         c=cluster_labels, cmap='tab10', alpha=0.6)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
    plt.title(f'K-means Clusters (k={n_clusters})')
    plt.colorbar(scatter)
    
    # Plot 3: Cluster centers in expert space
    plt.subplot(1, 3, 3)
    cluster_centers = kmeans.cluster_centers_  # [n_clusters, num_experts]
    
    for i, center in enumerate(cluster_centers):
        plt.plot(center, label=f'Cluster {i}', marker='o', markersize=3)
    
    plt.xlabel('Expert ID')
    plt.ylabel('Average Routing Probability')
    plt.title('Cluster Centers (Expert Preferences)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Analyze cluster composition
    print("\nCluster Analysis:")
    print("=" * 40)
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_datasets = np.array(all_labels)[cluster_mask]
        cluster_texts = np.array(all_texts)[cluster_mask]
        
        print(f"\nCluster {cluster_id} ({cluster_mask.sum()} samples):")
        
        # Dataset distribution
        dataset_counts = {name: (cluster_datasets == name).sum() for name in datasets.keys()}
        for dataset_name, count in dataset_counts.items():
            percentage = count / cluster_mask.sum() * 100
            print(f"  {dataset_name}: {count} samples ({percentage:.1f}%)")
        
        # Top experts for this cluster
        center = cluster_centers[cluster_id]
        top_experts = np.argsort(center)[-5:][::-1]  # Top 5 experts
        print(f"  Top experts: {top_experts} (probs: {center[top_experts]})")
        
        # Sample texts
        print(f"  Sample texts:")
        for i, text in enumerate(cluster_texts[:3]):
            print(f"    {i+1}. {text[:80]}...")
    
    return {
        'cluster_labels': cluster_labels,
        'cluster_centers': cluster_centers,
        'routing_2d': routing_2d,
        'pca': pca,
        'texts': all_texts,
        'dataset_labels': all_labels
    }

def expert_co_activation_analysis(datasets, layer_id: int = 8):
    """
    Analyze which experts tend to be activated together.
    
    Args:
        datasets: Loaded dataset dictionary
        layer_id: Layer to analyze co-activations
    """
    print(f"🤝 Analyzing expert co-activation patterns at Layer {layer_id}")
    
    analyzer = MoERoutingAnalyzer()
    
    # Initialize co-activation matrix
    co_activation_matrix = np.zeros((analyzer.num_experts, analyzer.num_experts))
    total_activations = np.zeros(analyzer.num_experts)
    
    for dataset_name, data in datasets.items():
        print(f"Processing {dataset_name}...")
        
        for sample_idx, routing_weights_sample in enumerate(data['routing_weights']):
            if layer_id < len(routing_weights_sample):
                routing_weights = routing_weights_sample[layer_id].squeeze(0)
                
                # Get top-k experts for each position
                top_k_indices, top_k_probs = analyzer.extract_top_k_experts(routing_weights.unsqueeze(0))
                top_k_indices = top_k_indices.squeeze(0)  # [seq_len, top_k]
                
                for pos in range(top_k_indices.shape[0]):
                    activated_experts = top_k_indices[pos].numpy()
                    
                    # Count individual activations
                    for expert in activated_experts:
                        total_activations[expert] += 1
                    
                    # Count co-activations
                    for i, expert_i in enumerate(activated_experts):
                        for j, expert_j in enumerate(activated_experts):
                            if i != j:  # Don't count self-activation
                                co_activation_matrix[expert_i, expert_j] += 1
    
    # Normalize co-activation matrix
    # Convert to conditional probabilities: P(expert_j | expert_i)
    co_activation_probs = np.zeros_like(co_activation_matrix)
    for i in range(analyzer.num_experts):
        if total_activations[i] > 0:
            co_activation_probs[i, :] = co_activation_matrix[i, :] / total_activations[i]
    
    # Visualize co-activation matrix
    plt.figure(figsize=(12, 10))
    im = plt.imshow(co_activation_probs, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, label='P(Expert j | Expert i)')
    plt.xlabel('Expert j (Co-activated)')
    plt.ylabel('Expert i (Primary)')
    plt.title(f'Expert Co-activation Probabilities (Layer {layer_id})')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Highlight strong co-activations
    threshold = 0.15  # Adjust based on your data
    strong_pairs = np.where(co_activation_probs > threshold)
    
    for i, j in zip(strong_pairs[0], strong_pairs[1]):
        if i != j:  # Skip diagonal
            plt.plot(j, i, 'w*', markersize=8, alpha=0.7)
    
    plt.show()
    
    # Find strongest co-activation pairs
    print(f"\nStrongest Co-activation Pairs (P > {threshold}):")
    print("=" * 50)
    
    strong_pairs_info = []
    for i in range(analyzer.num_experts):
        for j in range(analyzer.num_experts):
            if i != j and co_activation_probs[i, j] > threshold:
                strong_pairs_info.append((i, j, co_activation_probs[i, j]))
    
    # Sort by co-activation probability
    strong_pairs_info.sort(key=lambda x: x[2], reverse=True)
    
    for expert_i, expert_j, prob in strong_pairs_info[:20]:  # Top 20 pairs
        print(f"Expert {expert_i:2d} -> Expert {expert_j:2d}: {prob:.3f}")
    
    # Find expert communities (simple clustering based on co-activation)
    from sklearn.cluster import SpectralClustering
    
    # Use co-activation as similarity matrix
    similarity_matrix = (co_activation_probs + co_activation_probs.T) / 2
    
    n_communities = 8
    spectral = SpectralClustering(n_clusters=n_communities, affinity='precomputed', random_state=42)
    communities = spectral.fit_predict(similarity_matrix)
    
    print(f"\nExpert Communities (Spectral Clustering, k={n_communities}):")
    print("=" * 60)
    
    for community_id in range(n_communities):
        community_experts = np.where(communities == community_id)[0]
        print(f"Community {community_id}: Experts {list(community_experts)}")
        
        # Calculate internal co-activation strength
        if len(community_experts) > 1:
            internal_coactivation = []
            for i in community_experts:
                for j in community_experts:
                    if i != j:
                        internal_coactivation.append(co_activation_probs[i, j])
            
            if internal_coactivation:
                mean_internal = np.mean(internal_coactivation)
                print(f"  Mean internal co-activation: {mean_internal:.3f}")
    
    return {
        'co_activation_matrix': co_activation_matrix,
        'co_activation_probs': co_activation_probs,
        'total_activations': total_activations,
        'communities': communities,
        'strong_pairs': strong_pairs_info
    }

def temporal_routing_analysis(datasets, dataset_name: str = 'gsm8k'):
    """
    Analyze how routing patterns evolve within individual sequences.
    
    Args:
        datasets: Loaded dataset dictionary
        dataset_name: Which dataset to analyze
    """
    print(f"⏱️ Analyzing temporal routing patterns for {dataset_name}")
    
    if dataset_name not in datasets:
        print(f"❌ Dataset {dataset_name} not found")
        return
    
    data = datasets[dataset_name]
    analyzer = MoERoutingAnalyzer()
    
    # Select a few representative samples with different lengths
    sample_indices = []
    lengths = []
    for i, tokens in enumerate(data['tokens']):
        lengths.append(len(tokens))
    
    # Get samples with short, medium, and long sequences
    sorted_by_length = sorted(enumerate(lengths), key=lambda x: x[1])
    
    # Pick representative samples
    n_samples = len(sorted_by_length)
    representative_indices = [
        sorted_by_length[n_samples // 4][0],   # Short sequence
        sorted_by_length[n_samples // 2][0],   # Medium sequence  
        sorted_by_length[3 * n_samples // 4][0]  # Long sequence
    ]
    
    plt.figure(figsize=(15, 10))
    
    for plot_idx, sample_idx in enumerate(representative_indices):
        seq_len = lengths[sample_idx]
        text = data['texts'][sample_idx]
        tokens = data['tokens'][sample_idx]
        
        print(f"\nAnalyzing Sample {sample_idx} (length {seq_len}):")
        print(f"Text: {text}")
        
        # Analyze routing evolution across layers for this sample
        routing_evolution = []  # [layer, position, expert]
        
        for layer_idx in range(analyzer.num_layers):
            routing_weights = data['routing_weights'][sample_idx][layer_idx].squeeze(0)
            top_k_indices, top_k_probs = analyzer.extract_top_k_experts(routing_weights.unsqueeze(0))
            top_k_indices = top_k_indices.squeeze(0)
            top_k_probs = top_k_probs.squeeze(0)
            
            layer_routing = []
            for pos in range(seq_len):
                pos_experts = []
                for k in range(analyzer.top_k):
                    expert_id = top_k_indices[pos, k].item()
                    prob = top_k_probs[pos, k].item()
                    pos_experts.append((expert_id, prob))
                layer_routing.append(pos_experts)
            routing_evolution.append(layer_routing)
        
        # Plot routing heatmap for this sample
        plt.subplot(3, 1, plot_idx + 1)
        
        # Create heatmap showing most probable expert at each (layer, position)
        heatmap_data = np.zeros((analyzer.num_layers, seq_len))
        
        for layer_idx in range(analyzer.num_layers):
            for pos in range(seq_len):
                # Use the most probable expert
                most_probable_expert = routing_evolution[layer_idx][pos][0][0]
                heatmap_data[layer_idx, pos] = most_probable_expert
        
        im = plt.imshow(heatmap_data, aspect='auto', cmap='tab20', 
                       extent=[0, seq_len, analyzer.num_layers, 0])
        plt.colorbar(im, label='Expert ID')
        plt.ylabel('Layer')
        plt.xlabel('Token Position')
        plt.title(f'Sample {sample_idx}: Routing Evolution (Length={seq_len})')
        
        # Add token labels on x-axis if sequence is not too long
        if seq_len <= 15:
            plt.xticks(range(seq_len), 
                      [token[:8] for token in tokens], 
                      rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    # Analyze routing stability across positions
    print(f"\n📊 Routing Stability Analysis:")
    print("=" * 40)
    
    position_stability = []  # How much routing changes between adjacent positions
    layer_stability = []     # How much routing changes between adjacent layers
    
    for sample_idx in representative_indices:
        seq_len = lengths[sample_idx]
        
        # Position stability (within each layer)
        sample_pos_stability = []
        for layer_idx in range(analyzer.num_layers):
            routing_weights = data['routing_weights'][sample_idx][layer_idx].squeeze(0)
            routing_probs = torch.softmax(routing_weights, dim=-1)
            
            pos_changes = []
            for pos in range(1, seq_len):
                # Cosine similarity between adjacent positions
                cos_sim = torch.nn.functional.cosine_similarity(
                    routing_probs[pos-1], routing_probs[pos], dim=0
                ).item()
                pos_changes.append(1 - cos_sim)  # Convert similarity to change
            
            if pos_changes:
                sample_pos_stability.append(np.mean(pos_changes))
        
        position_stability.extend(sample_pos_stability)
        
        # Layer stability (within each position)
        sample_layer_stability = []
        for pos in range(seq_len):
            layer_changes = []
            for layer_idx in range(1, analyzer.num_layers):
                routing_weights_prev = data['routing_weights'][sample_idx][layer_idx-1].squeeze(0)
                routing_weights_curr = data['routing_weights'][sample_idx][layer_idx].squeeze(0)
                
                routing_probs_prev = torch.softmax(routing_weights_prev[pos], dim=-1)
                routing_probs_curr = torch.softmax(routing_weights_curr[pos], dim=-1)
                
                cos_sim = torch.nn.functional.cosine_similarity(
                    routing_probs_prev, routing_probs_curr, dim=0
                ).item()
                layer_changes.append(1 - cos_sim)
            
            if layer_changes:
                sample_layer_stability.append(np.mean(layer_changes))
        
        layer_stability.extend(sample_layer_stability)
    
    print(f"Mean positional routing change: {np.mean(position_stability):.4f}")
    print(f"Mean layer-wise routing change: {np.mean(layer_stability):.4f}")
    
    # Plot stability metrics
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(position_stability, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Routing Change (Adjacent Positions)')
    plt.ylabel('Frequency')
    plt.title('Positional Routing Stability')
    plt.axvline(np.mean(position_stability), color='red', linestyle='--', 
               label=f'Mean: {np.mean(position_stability):.3f}')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(layer_stability, bins=20, alpha=0.7, color='green')
    plt.xlabel('Routing Change (Adjacent Layers)')
    plt.ylabel('Frequency')
    plt.title('Layer-wise Routing Stability')
    plt.axvline(np.mean(layer_stability), color='red', linestyle='--', 
               label=f'Mean: {np.mean(layer_stability):.3f}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'position_stability': position_stability,
        'layer_stability': layer_stability,
        'representative_samples': representative_indices
    }

if __name__ == "__main__":
    # Example usage - uncomment the analysis you want to run
    
    print("🔬 MoE Routing Analysis Toolkit")
    print("="*50)
    
    # Load your datasets
    datasets = load_your_datasets()
    
    if datasets:
        print(f"\n✅ Loaded {len(datasets)} datasets")
        
        # Option 1: Quick comprehensive analysis
        # analyzer = run_comprehensive_analysis()
        
        # Option 2: Focus on specific expert
        # focused_expert_analysis(datasets, expert_id=32, layer_id=8)
        
        # Option 3: Routing pattern clustering
        # clustering_results = routing_pattern_clustering(datasets, layer_id=8)
        
        # Option 4: Expert co-activation analysis
        # coactivation_results = expert_co_activation_analysis(datasets, layer_id=8)
        
        # Option 5: Temporal routing analysis
        # temporal_results = temporal_routing_analysis(datasets, 'gsm8k')
        
        print("\n💡 Uncomment the desired analysis function above to run specific analyses!")
        print("📊 For a complete analysis, run: run_comprehensive_analysis()")
    else:
        print("❌ No datasets loaded. Please check your data paths.")