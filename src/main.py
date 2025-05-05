import pandas as pd
from data_utils import load_dataset
from structure_learning import generate_initial_dag, compute_score, optimize_structure
from evaluation import evaluate_structure, compare_with_ground_truth
import numpy as np
import os

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
    from pgmpy.models import BayesianNetwork
    
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

def main():
    # Generate and save synthetic datasets if they don't exist
    os.makedirs('datasets', exist_ok=True)
    if not os.path.exists('datasets/dataset_1000.csv'):
        df_1000 = generate_synthetic_dataset(1000, seed=42)
        df_1000.to_csv('datasets/dataset_1000.csv', index=False)
        print("Synthetic 1000 records datasets generated.")
    if not os.path.exists('datasets/dataset_50000.csv'):
        df_50000 = generate_synthetic_dataset(50000, seed=42)
        df_50000.to_csv('datasets/dataset_50000.csv', index=False)
        print("Synthetic 50000 records datasets generated.")

    # Load datasets
    dataset_1000 = load_dataset('datasets/dataset_1000.csv')
    dataset_50000 = load_dataset('datasets/dataset_50000.csv')
    print("Datasets loaded.")

    # Get ground truth DAG
    ground_truth = get_ground_truth_dag()
    print("Ground truth DAG generated.")
    print(f"Ground truth edges: {ground_truth.edges()}")

    # Process each dataset
    for dataset, name in zip([dataset_1000, dataset_50000], ['dataset_1000', 'dataset_50000']):
        print(f"\n===== Processing {name} =====")

        # Generate initial DAG
        initial_dag = generate_initial_dag(dataset)
        print(f"Initial DAG edges: {initial_dag.edges()}")

        # Compute score of the initial DAG
        initial_score = compute_score(initial_dag, dataset)
        print(f"Initial score: {initial_score}")

        # Optimize the structure using hill climbing or other strategies
        optimized_dag = optimize_structure(initial_dag, dataset)
        print(f"Optimized DAG edges: {optimized_dag.edges()}")
        
        # Compute final score
        final_score = compute_score(optimized_dag, dataset)
        print(f"Final score: {final_score}")

        # Compare with ground truth
        comparison = compare_with_ground_truth(optimized_dag, ground_truth)
        
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

if __name__ == "__main__":
    main()