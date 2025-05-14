# Bayesian Network Structure Learning Report

Generated: 2025-05-14 23:25:44

## Dataset Information

### dataset_1000
- Shape: (1000, 7)
- Columns: ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'target']
- Sample:

```
   feature1  feature2  feature3  feature4  feature5  feature6  target
0         0         0         0         0         0         1       1
1         1         0         0         1         1         0       1
2         0         0         0         0         0         1       0
```

### dataset_50000
- Shape: (50000, 7)
- Columns: ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'target']
- Sample:

```
   feature1  feature2  feature3  feature4  feature5  feature6  target
0         0         1         0         0         1         0       1
1         0         1         0         0         0         1       1
2         0         0         0         0         0         1       1
```

## Ground Truth DAG

- Edges: [('feature1', 'feature2'), ('feature1', 'feature3'), ('feature2', 'feature4'), ('feature3', 'feature4'), ('feature3', 'target'), ('feature4', 'target')]

## Structure Learning Experiments

### dataset_1000

#### Initial DAG
- Edges: [('feature1', 'feature3'), ('feature1', 'feature4'), ('feature3', 'feature4'), ('feature3', 'feature5'), ('feature4', 'feature6'), ('feature2', 'feature3'), ('feature2', 'feature5'), ('feature5', 'feature6'), ('feature5', 'target'), ('feature6', 'target')]
- Score: -3137.647878956002

#### Optimized DAG
- Edges: [('feature1', 'feature3'), ('feature1', 'feature4'), ('feature3', 'feature4'), ('feature3', 'feature5'), ('feature4', 'feature6'), ('feature5', 'feature6'), ('feature5', 'target'), ('feature6', 'target'), ('feature2', 'feature3'), ('feature2', 'feature5')]
- Score: -3137.6478789560024

#### Comparison with Ground Truth
- True Positive Edges: {('feature1', 'feature3'), ('feature3', 'feature4')}
- False Positive Edges: {('feature1', 'feature4'), ('feature5', 'feature6'), ('feature3', 'feature5'), ('feature5', 'target'), ('feature6', 'target'), ('feature2', 'feature3'), ('feature4', 'feature6'), ('feature2', 'feature5')}
- False Negative Edges: {('feature3', 'target'), ('feature2', 'feature4'), ('feature1', 'feature2'), ('feature4', 'target')}
- Reversed Edges: set()
- Precision: 0.2000
- Recall: 0.3333
- F1 Score: 0.2500
- Structural Hamming Distance: 12

### dataset_50000

#### Initial DAG
- Edges: [('feature1', 'feature3'), ('feature1', 'feature4'), ('feature3', 'feature4'), ('feature3', 'feature5'), ('feature4', 'feature6'), ('feature2', 'feature3'), ('feature2', 'feature5'), ('feature5', 'feature6'), ('feature5', 'target'), ('feature6', 'target')]
- Score: -153531.50793444988

#### Optimized DAG
- Edges: [('feature1', 'feature3'), ('feature1', 'feature4'), ('feature3', 'feature4'), ('feature3', 'feature5'), ('feature4', 'feature6'), ('feature5', 'feature6'), ('feature5', 'target'), ('feature6', 'target'), ('feature2', 'feature3'), ('feature2', 'feature5')]
- Score: -153531.50793444988

#### Comparison with Ground Truth
- True Positive Edges: {('feature1', 'feature3'), ('feature3', 'feature4')}
- False Positive Edges: {('feature1', 'feature4'), ('feature5', 'feature6'), ('feature3', 'feature5'), ('feature5', 'target'), ('feature6', 'target'), ('feature2', 'feature3'), ('feature4', 'feature6'), ('feature2', 'feature5')}
- False Negative Edges: {('feature3', 'target'), ('feature2', 'feature4'), ('feature1', 'feature2'), ('feature4', 'target')}
- Reversed Edges: set()
- Precision: 0.2000
- Recall: 0.3333
- F1 Score: 0.2500
- Structural Hamming Distance: 12

## Conclusions

- Summary of findings
- Best performing approach
- Recommendations for future work

