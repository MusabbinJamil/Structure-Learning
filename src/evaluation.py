def compare_structures(learned_structure, original_structure):
    # Compare the learned structure with the original structure
    learned_edges = set(learned_structure.edges())
    original_edges = set(original_structure.edges())
    
    added_edges = learned_edges - original_edges
    removed_edges = original_edges - learned_edges
    
    return added_edges, removed_edges

def report_comparison(added_edges, removed_edges):
    # Report the differences between the learned structure and the original structure
    print("Comparison of Learned Structure with Original Structure:")
    if added_edges:
        print("Added Edges:")
        for edge in added_edges:
            print(edge)
    else:
        print("No edges added.")
    
    if removed_edges:
        print("Removed Edges:")
        for edge in removed_edges:
            print(edge)
    else:
        print("No edges removed.")

def evaluate_structure(learned_structure, original_structure):
    added_edges, removed_edges = compare_structures(learned_structure, original_structure)
    report_comparison(added_edges, removed_edges)

def compare_with_ground_truth(learned_dag, ground_truth_dag):
    """
    Compare the learned DAG structure with the ground truth DAG.
    
    Args:
        learned_dag: The DAG learned by the structure learning algorithm
        ground_truth_dag: The ground truth DAG
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Get edges from both DAGs
    learned_edges = set(learned_dag.edges())
    true_edges = set(ground_truth_dag.edges())
    
    # Calculate metrics
    true_positives = learned_edges.intersection(true_edges)
    false_positives = learned_edges - true_edges
    false_negatives = true_edges - learned_edges
    
    # Calculate precision, recall, and F1 score
    precision = len(true_positives) / max(len(learned_edges), 1)
    recall = len(true_positives) / max(len(true_edges), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    
    # Calculate structural Hamming distance (SHD)
    # SHD is the number of operations (add/delete/reverse) needed to transform one graph to another
    shd = len(false_positives) + len(false_negatives)
    
    # Check for reversed edges (count as partial matches)
    reversed_edges = set()
    for u, v in learned_edges:
        if (v, u) in true_edges:
            reversed_edges.add((u, v))
            shd += 1  # Each reversal counts as an additional operation
    
    # Remove reversed edges from false positives as they're counted separately
    false_positives = false_positives - reversed_edges
    
    results = {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'reversed_edges': reversed_edges,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'structural_hamming_distance': shd
    }
    
    return results