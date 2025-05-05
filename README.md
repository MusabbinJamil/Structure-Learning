# Bayesian Network Structure Learning

This project implements a program that learns the structure of a Bayesian network from synthetic datasets using the `pgmpy` library. The program follows a structured approach: generating synthetic data (if needed), defining a ground truth DAG, generating an initial Directed Acyclic Graph (DAG) from data, computing its score, optimizing the structure using search strategies, and evaluating the learned structure against the ground truth.

## Project Structure

- `src/main.py`: Entry point of the application. Orchestrates the workflow including dataset generation/loading, defining the ground truth, generating the initial DAG, scoring, structure learning, and evaluation.
- `src/data_utils.py`: Contains utility functions for loading datasets from CSV files into pandas DataFrames.
- `src/structure_learning.py`: Implements structure learning algorithms, including the generation of the initial DAG and optimization strategies such as hill climbing (`optimize_structure`). Also includes functions for scoring (`compute_score`).
- `src/evaluation.py`: Provides functions for evaluating the learned structure against the ground truth DAG (`compare_with_ground_truth`) and comparing it with the initial DAG (`evaluate_structure`).
- `src/datasets/dataset_1000.csv`: Synthetic dataset with 1000 records generated from the Bayesian network (created automatically if not present).
- `src/datasets/dataset_50000.csv`: Synthetic dataset with 50000 records generated from the Bayesian network (created automatically if not present).
- `requirements.txt`: Lists the dependencies required for the project, including `pgmpy` and other necessary libraries.

## Getting Started

### Prerequisites

Make sure you have Python installed on your machine. You will also need to install the required libraries listed in `requirements.txt`.

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd bayesian-network-structure-learning
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Running the Program

To run the program, execute the following command in your terminal:

```
python src/main.py
```

This will:
1. Check if the synthetic datasets exist in the `datasets/` folder. If not, it will generate them (`dataset_1000.csv` and `dataset_50000.csv`).
2. Load the datasets.
3. Define the ground truth DAG structure.
4. For each dataset:
    - Generate an initial DAG.
    - Compute the initial score.
    - Optimize the DAG structure using a search algorithm (e.g., Hill Climbing).
    - Compute the final score.
    - Compare the optimized DAG with the ground truth DAG and print evaluation metrics.
    - Compare the optimized DAG with the initial DAG.

## Comparison with Ground Truth

After the structure learning process for each dataset, the learned (optimized) structure is compared with the predefined ground truth structure. The evaluation results printed to the console include:

- True Positive Edges
- False Positive Edges
- False Negative Edges
- Reversed Edges
- Precision
- Recall
- F1 Score
- Structural Hamming Distance (SHD)

These metrics help assess the accuracy of the learned network structure.

## Acknowledgments

This project utilizes the `pgmpy` library for probabilistic graphical models and structure learning. For more information on `pgmpy`, visit the official documentation at [pgmpy.org](https://pgmpy.org).