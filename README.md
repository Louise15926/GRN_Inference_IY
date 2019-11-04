# Gene Regulatory Network (GRN) Inference Project

In this project, we attempt to infer gene regulatory networks from expression data. We obtained the expression data from the [MERLIN+P project](https://github.com/Roy-lab/merlin-p_inferred_networks) (Citation needed). 
<br>

## Inference Methods
### Lasso Regression
We attempt to infer gene regulatory networks by treating it as a feature selection problem:
1. Treat each gene as target (y) and all other genes as predictors (X).
2. Use Lasso Regression method to predict y based on X.
3. Get predictor genes in X that has non-zero weight in the model

### Regression Forest
Like Lasso, we also treat grn inference as a feature selection problem:
1. Treat each gene as target (y) and all other genes as predictors (X).
2. Use Regression Forrest method to predict y based on X.
3. Get predictor genes in X that has non-zero feature importance
## Evaluation Metrics
### Intersection Over Union (IOU)
Using IOU, we are able to swiftly score how well our inference method by having the proportion of intersection over union between the predicted edges and gold-standard edges.
<br>
(Insert Venn Diagram Image here)
<br>
In the case of a perfect prediction, the prediction set and the gold-standard edges are completely overlapping. In the case of entirely incorrect prediction, there is no intersection between predicted edges and gold-standard edges.

## Notes:
For next steps:
Inference Methods:
- Setting hyperparameters for Lasso
- Other non-linear methods such as regression trees
- Hyperparameter search for both lasso and regression trees.

Evaluation:
- IOU with statistical adjustment (Probability of seeing IOU with the case of random adjustment)
- Regression evaluation, R^2 between observed vs predictions for train and validation data
