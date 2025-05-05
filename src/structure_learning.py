from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BicScore
import random
import pandas as pd

def generate_initial_dag(dataset):
    """
    Generate an initial Bayesian Network DAG using the actual column names from the dataset.
    Ensures that no cycles are formed.
    """
    # Get column names from the dataset
    nodes = list(dataset.columns)
    
    # Create edges (only allowing edges from earlier to later columns)
    # This automatically prevents cycles
    edges = []
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            # Add edge with 50% probability
            if random.random() > 0.5:
                edges.append((nodes[i], nodes[j]))
    
    return BayesianNetwork(edges)

def compute_score(dag, dataset):
    """
    Compute the BIC score for a given DAG structure on the dataset.
    BIC balances goodness of fit with model complexity.
    """
    try:
        # Make a copy to avoid modifying the original dataset
        dataset = dataset.copy()
        
        # Check if the DAG has any nodes and edges
        if len(dag.nodes()) == 0:
            print("Warning: Empty DAG provided for scoring")
            return float('-inf')
            
        # Ensure all node names in DAG exist as columns in the dataset
        dag_nodes = set(dag.nodes())
        dataset_columns = set(dataset.columns)
        
        # Print debugging info
        print(f"DAG nodes: {dag_nodes}")
        print(f"Dataset columns: {dataset_columns}")
        
        if not dag_nodes.issubset(dataset_columns):
            missing = dag_nodes - dataset_columns
            print(f"Error: DAG contains nodes that don't exist in dataset: {missing}")
            return float('-inf')
        
        # Ensure all values are discrete - BIC requires discrete data
        for col in dataset.columns:
            # Convert any non-categorical/non-integer data to categorical
            if dataset[col].dtype.kind not in 'iub':  # integer, unsigned int, or boolean
                # For float columns, discretize into 4 bins
                if dataset[col].dtype.kind == 'f':
                    dataset[col] = pd.qcut(dataset[col], 4, labels=[0,1,2,3], duplicates='drop')
                # For other types, just convert to categorical codes
                else:
                    dataset[col] = dataset[col].astype('category').cat.codes
                    
            # Ensure no NaN values
            if dataset[col].isna().any():
                dataset[col] = dataset[col].fillna(dataset[col].mode()[0])
                
        # Initialize the BIC scorer
        scoring_method = BicScore(dataset)
        
        # Calculate the score
        score = scoring_method.score(dag)
        return score
    except Exception as e:
        print(f"Error computing score: {e}")
        print(f"Exception type: {type(e)}")
        # Print stack trace for more detailed debugging
        import traceback
        traceback.print_exc()
        return float('-inf')  # Return negative infinity for invalid models

def optimize_structure(initial_dag, dataset):
    """
    Optimize the structure using hill climbing algorithm.
    Tries to maximize the BIC score by adding, removing, or reversing edges.
    """
    # Create a copy of the initial DAG
    current_dag = BayesianNetwork(initial_dag.edges())  # Changed from BayesianModel
    current_score = compute_score(current_dag, dataset)
    
    # Convert column names to strings
    dataset = dataset.copy()
    dataset.columns = [str(col) for col in dataset.columns]
    
    # Get all possible nodes
    nodes = list(dataset.columns)
    
    improved = True
    iterations = 0
    max_iterations = 100  # Prevent infinite loops
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Try adding edges
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i != j:
                    # Skip if edge already exists
                    if (nodes[i], nodes[j]) in current_dag.edges():
                        continue
                    
                    # Try adding edge if it doesn't create a cycle
                    try:
                        new_dag = BayesianNetwork(current_dag.edges())  # Changed from BayesianModel
                        new_dag.add_edge(nodes[i], nodes[j])
                        new_score = compute_score(new_dag, dataset)
                        
                        if new_score > current_score:
                            current_dag = new_dag
                            current_score = new_score
                            improved = True
                            print(f"Added edge {nodes[i]} -> {nodes[j]}, new score: {new_score}")
                    except Exception:
                        # Edge would create a cycle, skip it
                        pass
        
        # Try removing edges
        for u, v in list(current_dag.edges()):
            new_dag = BayesianNetwork(current_dag.edges())  # Changed from BayesianModel
            new_dag.remove_edge(u, v)
            new_score = compute_score(new_dag, dataset)
            
            if new_score > current_score:
                current_dag = new_dag
                current_score = new_score
                improved = True
                print(f"Removed edge {u} -> {v}, new score: {new_score}")
        
        # Try reversing edges
        for u, v in list(current_dag.edges()):
            try:
                new_dag = BayesianNetwork(current_dag.edges())  # Changed from BayesianModel
                new_dag.remove_edge(u, v)
                new_dag.add_edge(v, u)
                new_score = compute_score(new_dag, dataset)
                
                if new_score > current_score:
                    current_dag = new_dag
                    current_score = new_score
                    improved = True
                    print(f"Reversed edge {u} -> {v} to {v} -> {u}, new score: {new_score}")
            except Exception:
                # Edge reversal would create a cycle, skip it
                pass
    
    print(f"Structure optimization completed after {iterations} iterations.")
    return current_dag

def learn_structure(dataset_path):
    print(f"Learning structure from dataset: {dataset_path}")
    # Read the dataset
    data = pd.read_csv(dataset_path)
    print(f"Dataset shape: {data.shape}")
    # Generate initial DAG
    initial_dag = generate_initial_dag(data)
    print(f"Initial DAG: {initial_dag.edges()}")

    # Compute the score of the initial DAG
    initial_score = compute_score(initial_dag, data)
    print(f"Initial score: {initial_score}")

    # Optimize the DAG structure
    final_model = optimize_structure(initial_dag, data)
    print(f"Final model: {final_model.edges()}")
    
    # Compute the final score
    final_score = compute_score(final_model, data)
    print(f"Final score: {final_score}")

    return initial_dag, initial_score, final_model, final_score

# Example usage:
# initial_dag, initial_score, final_model, final_score = learn_structure('datasets/dataset_1000.csv')