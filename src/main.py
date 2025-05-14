import pandas as pd
from pgmpy.models import BayesianNetwork
from data_utils import load_dataset
from structure_learning import generate_initial_dag, compute_score, optimize_structure
from evaluation import evaluate_structure, compare_with_ground_truth
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import networkx as nx

def generate_synthetic_dataset(num_rows=1000, seed=None):
    """
    Generates a synthetic dataset with binary variables that have 
    conditional dependencies for Bayesian Network structure learning.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create empty dataframe
    data = pd.DataFrame()
    
    # Generate first feature randomly (no parents)
    data['feature1'] = np.random.binomial(1, 0.3, size=num_rows)
    
    # Generate feature2 with dependency on feature1
    # P(feature2=1|feature1=0) = 0.2
    # P(feature2=1|feature1=1) = 0.7
    prob_feature2 = np.where(data['feature1'] == 1, 0.7, 0.2)
    data['feature2'] = np.random.binomial(1, prob_feature2, size=num_rows)
    
    # Generate feature3 with dependency on feature1
    # P(feature3=1|feature1=0) = 0.1
    # P(feature3=1|feature1=1) = 0.8
    prob_feature3 = np.where(data['feature1'] == 1, 0.8, 0.1)
    data['feature3'] = np.random.binomial(1, prob_feature3, size=num_rows)
    
    # Generate feature4 with dependency on feature2 and feature3
    # Using a logical OR-like relationship
    prob_feature4 = 0.1 + 0.4*data['feature2'] + 0.4*data['feature3']
    prob_feature4 = np.clip(prob_feature4, 0, 0.9)  # Ensure probability doesn't exceed 0.9
    data['feature4'] = np.random.binomial(1, prob_feature4, size=num_rows)
    
    # Generate feature5 independent of others
    data['feature5'] = np.random.binomial(1, 0.5, size=num_rows)
    
    # Generate target with dependencies on feature3 and feature4
    # P(target=1) increases with both feature3 and feature4
    prob_target = 0.1 + 0.3*data['feature3'] + 0.5*data['feature4']
    prob_target = np.clip(prob_target, 0, 0.9)  # Ensure probability doesn't exceed 0.9
    data['target'] = np.random.binomial(1, prob_target, size=num_rows)
    
    return data

def get_ground_truth_dag():
    """
    Returns the ground truth DAG for our synthetic data generation process.
    """
    # Create an empty Bayesian Network
    
    # Define the edges based on how we generated the data
    edges = [
        ('feature1', 'feature2'),
        ('feature1', 'feature3'),
        ('feature2', 'feature4'),
        ('feature3', 'feature4'),
        ('feature3', 'target'),
        ('feature4', 'target')
    ]
    
    return BayesianNetwork(edges)

def load_txt_dataset(file_path):
    """
    Loads data from a text file with T/F values and converts it to a pandas DataFrame
    with proper binary encoding (1 for True, 0 for False).
    """
    # Read the space-separated text file
    df = pd.read_csv(file_path, sep=r'\s+')
    
    # Convert T/F to 1/0
    for col in df.columns:
        df[col] = df[col].map({'T': 1, 'F': 0})
    
    # Map generic column names to the expected feature names
    column_mapping = {
        'A': 'feature1',
        'B': 'feature2', 
        'C': 'feature3',
        'D': 'feature4',
        'E': 'feature5',
        'F': 'feature6',
        'G': 'target'
    }
    
    df = df.rename(columns=column_mapping)
    return df

def generate_report(datasets_info, experiment_results, report_path='report'):
    """
    Generate a comprehensive report of the structure learning experiments.
    
    Args:
        datasets_info: Dictionary with dataset information
        experiment_results: Dictionary with experiment results
        report_path: Directory to save the report
    """
    # Create report directory if it doesn't exist
    os.makedirs(report_path, exist_ok=True)
    
    # Generate timestamp for unique report filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(report_path, f"structure_learning_report_{timestamp}.md")
    
    with open(report_file, "w") as f:
        # Write report header
        f.write("# Bayesian Network Structure Learning Report\n\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write dataset information
        f.write("## Dataset Information\n\n")
        for name, info in datasets_info.items():
            f.write(f"### {name}\n")
            f.write(f"- Shape: {info['shape']}\n")
            f.write(f"- Columns: {info['columns']}\n")
            f.write(f"- Sample:\n\n```\n{info['sample']}\n```\n\n")
        
        # Write ground truth information
        f.write("## Ground Truth DAG\n\n")
        f.write(f"- Edges: {experiment_results['ground_truth_edges']}\n\n")
        
        # Write experiment results for each dataset
        f.write("## Structure Learning Experiments\n\n")
        for dataset_name, results in experiment_results.items():
            if dataset_name == 'ground_truth_edges':
                continue
                
            f.write(f"### {dataset_name}\n\n")
            
            # Initial DAG information
            f.write("#### Initial DAG\n")
            f.write(f"- Edges: {results['initial_edges']}\n")
            f.write(f"- Score: {results['initial_score']}\n\n")
            
            # Optimized DAG information
            f.write("#### Optimized DAG\n")
            f.write(f"- Edges: {results['optimized_edges']}\n")
            f.write(f"- Score: {results['final_score']}\n\n")
            
            # Comparison with ground truth
            f.write("#### Comparison with Ground Truth\n")
            f.write(f"- True Positive Edges: {results['comparison']['true_positives']}\n")
            f.write(f"- False Positive Edges: {results['comparison']['false_positives']}\n")
            f.write(f"- False Negative Edges: {results['comparison']['false_negatives']}\n")
            f.write(f"- Reversed Edges: {results['comparison']['reversed_edges']}\n")
            f.write(f"- Precision: {results['comparison']['precision']:.4f}\n")
            f.write(f"- Recall: {results['comparison']['recall']:.4f}\n")
            f.write(f"- F1 Score: {results['comparison']['f1_score']:.4f}\n")
            f.write(f"- Structural Hamming Distance: {results['comparison']['structural_hamming_distance']}\n\n")
            
            # Add any other relevant information
            if 'notes' in results:
                f.write(f"#### Additional Notes\n{results['notes']}\n\n")
        
        # Add conclusions or observations
        f.write("## Conclusions\n\n")
        f.write("- Summary of findings\n")
        f.write("- Best performing approach\n")
        f.write("- Recommendations for future work\n\n")
        
    print(f"Report generated and saved to {report_file}")
    return report_file

def visualize_network(dag, filename):
    """
    Create a visualization of the Bayesian Network and save it to a file.
    """
    plt.figure(figsize=(10, 8))
    nx_graph = nx.DiGraph()
    
    # Add edges to the networkx graph
    for edge in dag.edges():
        nx_graph.add_edge(edge[0], edge[1])
    
    # Draw the graph
    pos = nx.spring_layout(nx_graph)
    nx.draw(nx_graph, pos, with_labels=True, node_color='skyblue', 
            node_size=1500, arrowsize=20, font_size=12, font_weight='bold')
    
    # Save the figure
    plt.savefig(filename)
    plt.close()

def create_custom_initial_dag():
    """
    Creates a custom initial DAG structure with the specified edges.
    
    The structure follows:
    A to C and D
    B to C and E
    C to D and E
    D to F
    E to F and G
    F to G
    
    Using our feature naming:
    feature1 → feature3, feature1 → feature4
    feature2 → feature3, feature2 → feature5
    feature3 → feature4, feature3 → feature5
    feature4 → feature6
    feature5 → feature6, feature5 → target
    feature6 → target
    """
    # Define the edges based on the specified structure
    edges = [
        ('feature1', 'feature3'),  # A to C
        ('feature1', 'feature4'),  # A to D
        ('feature2', 'feature3'),  # B to C
        ('feature2', 'feature5'),  # B to E
        ('feature3', 'feature4'),  # C to D
        ('feature3', 'feature5'),  # C to E
        ('feature4', 'feature6'),  # D to F
        ('feature5', 'feature6'),  # E to F
        ('feature5', 'target'),    # E to G
        ('feature6', 'target')     # F to G
    ]
    
    return BayesianNetwork(edges)

def main():
    # Create dictionaries to store information for the report
    datasets_info = {}
    experiment_results = {}
    
    # Load datasets from text files
    dataset_1000 = load_txt_dataset('1000_recs.txt')
    dataset_50000 = load_txt_dataset('50000_recs.txt')
    print("Text datasets loaded.")
    
    # Store dataset information for the report
    datasets_info['dataset_1000'] = {
        'shape': dataset_1000.shape,
        'columns': dataset_1000.columns.tolist(),
        'sample': dataset_1000.head(3).to_string()
    }
    
    datasets_info['dataset_50000'] = {
        'shape': dataset_50000.shape,
        'columns': dataset_50000.columns.tolist(),
        'sample': dataset_50000.head(3).to_string()
    }
    
    print(f"Dataset 1000 shape: {dataset_1000.shape}")
    print(f"Dataset 50000 shape: {dataset_50000.shape}")
    print(f"Dataset 1000 columns: {dataset_1000.columns.tolist()}")
    print(f"Sample of dataset 1000:\n{dataset_1000.head(3)}")

    # Get ground truth DAG
    ground_truth = get_ground_truth_dag()
    print("Ground truth DAG generated.")
    print(f"Ground truth edges: {ground_truth.edges()}")
    
    # Store ground truth information for the report
    experiment_results['ground_truth_edges'] = ground_truth.edges()
    
    # Create a directory for visualizations
    os.makedirs('visualizations', exist_ok=True)
    
    # Visualize ground truth DAG
    visualize_network(ground_truth, 'visualizations/ground_truth_dag.png')

    # Process each dataset
    for dataset, name in zip([dataset_1000, dataset_50000], ['dataset_1000', 'dataset_50000']):
        print(f"\n===== Processing {name} =====")
        
        # Initialize dictionary for this dataset's results
        experiment_results[name] = {}

        # Use custom initial DAG instead of generating one
        initial_dag = create_custom_initial_dag()
        print(f"Initial DAG edges: {initial_dag.edges()}")
        
        # Store initial DAG information for the report
        experiment_results[name]['initial_edges'] = initial_dag.edges()
        
        # Visualize initial DAG
        visualize_network(initial_dag, f'visualizations/{name}_initial_dag.png')

        # Compute score of the initial DAG
        initial_score = compute_score(initial_dag, dataset)
        print(f"Initial score: {initial_score}")
        
        # Store initial score for the report
        experiment_results[name]['initial_score'] = initial_score

        # Optimize the structure using hill climbing or other strategies
        optimized_dag = optimize_structure(initial_dag, dataset)
        print(f"Optimized DAG edges: {optimized_dag.edges()}")
        
        # Store optimized DAG information for the report
        experiment_results[name]['optimized_edges'] = optimized_dag.edges()
        
        # Visualize optimized DAG
        visualize_network(optimized_dag, f'visualizations/{name}_optimized_dag.png')
        
        # Compute final score
        final_score = compute_score(optimized_dag, dataset)
        print(f"Final score: {final_score}")
        
        # Store final score for the report
        experiment_results[name]['final_score'] = final_score

        # Compare with ground truth
        comparison = compare_with_ground_truth(optimized_dag, ground_truth)
        
        # Store comparison results for the report
        experiment_results[name]['comparison'] = comparison
        
        # Print comparison results
        print("\n----- Comparison with Ground Truth -----")
        print(f"True Positive Edges: {comparison['true_positives']}")
        print(f"False Positive Edges: {comparison['false_positives']}")
        print(f"False Negative Edges: {comparison['false_negatives']}")
        print(f"Reversed Edges: {comparison['reversed_edges']}")
        print(f"Precision: {comparison['precision']:.4f}")
        print(f"Recall: {comparison['recall']:.4f}")
        print(f"F1 Score: {comparison['f1_score']:.4f}")
        print(f"Structural Hamming Distance (SHD): {comparison['structural_hamming_distance']}")
        
        # Optionally, still run your original evaluation
        print("\n----- Comparison with Initial DAG -----")
        evaluate_structure(optimized_dag, initial_dag)
    
    # Generate the report
    report_file = generate_report(datasets_info, experiment_results)
    
    print(f"\nExperiment completed. Report saved to: {report_file}")

if __name__ == "__main__":
    main()