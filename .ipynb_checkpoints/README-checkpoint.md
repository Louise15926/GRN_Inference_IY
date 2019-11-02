# Gene Regulatory Network (GRN) Inference Project

In this project, we attempt to infer gene regulatory networks from expression data. We obtained the expression data from the [MERLIN+P project](https://github.com/Roy-lab/merlin-p_inferred_networks) (Citation needed). 
<br>

## Inference Methods
### Lasso Regression
We attempt to infer gene regulatory networks by treating it as a feature selection problem:
1. Treat each gene as target (y) and all other genes as predictors (X).
2. Use Lasso Regression method to predict y based on X.
3. Get predictor genes in X that has non-zero weight in the model

## Evaluation Metrics
### Intersection Over Union (IOU)
Using IOU, we are able to swiftly score how well our inference method by having the proportion of intersection over union between the predicted edges and gold-standard edges.
(Insert Venn Diagram Image here)
In the case of a perfect prediction, the prediction set and the gold-standard edges are completely overlapping. In the other hand, with the case of entirely incorrect prediction, there is no intersection between predicted edges and gold-standard edges.