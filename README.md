# Bayesian Network Structure Learning

This project implements a program that learns the structure of a Bayesian network from synthetic datasets using the `pgmpy` library. The program follows a structured approach to read datasets, generate an initial Directed Acyclic Graph (DAG), compute its score, and optimize the structure using search strategies.

## Project Structure

- `src/main.py`: Entry point of the application. Orchestrates the workflow of reading datasets, generating the initial DAG, scoring it, and applying structure learning algorithms.
- `src/data_utils.py`: Contains utility functions for loading datasets from CSV files into pandas DataFrames.
- `src/structure_learning.py`: Implements structure learning algorithms, including the generation of the initial DAG and optimization strategies such as hill climbing.
- `src/evaluation.py`: Provides functions for evaluating the learned structure against the original structure obtained from GeNIe.
- `datasets/dataset_1000.csv`: Synthetic dataset with 1000 records generated from the Bayesian network.
- `datasets/dataset_50000.csv`: Synthetic dataset with 50000 records generated from the Bayesian network.
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

This will initiate the process of reading the datasets, generating the initial DAG, scoring it, and applying the structure learning algorithms.

## Comparison with GeNIe

After the structure learning process, the learned structure will be compared with the original structure obtained from GeNIe. The evaluation results will highlight the differences and similarities between the two structures.

## Acknowledgments

This project utilizes the `pgmpy` library for probabilistic graphical models and structure learning. For more information on `pgmpy`, visit the official documentation at [pgmpy.org](https://pgmpy.org).

## License

This project is licensed under the MIT License - see the LICENSE file for details.