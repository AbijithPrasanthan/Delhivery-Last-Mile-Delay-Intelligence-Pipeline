# Delhivery Last Mile Delay Intelligence Pipeline

## Part A - Findings

1. The Express service type is delayed the most with a delay rate of 48.97% (119/243 shipments are delayed).
2. The delay rate is the least for tier 1 at 4% and increases to 21.5% in tier 2 and 44% in tier 3 hubs
3. The delay rate also increases from 0-200 to 601+ with values ranging from 2.19% to 85.23%
4. Missed hours, Distance KM and n_stops are the top 3 columns that are most correlated with delivery_hours with values 0.84, 0.40 and 0.32 respectively.
5. declared_value_inr has very large numbers that can skew the model and the data. Log transformation brings it down to usable or normalized values.

## Model Results (regression)

1. The baseline R2 score is ***-0.0212***. This means that the predictions are worse than a naive guess such as predicting the validation dataset's own mean. 
2. The winning model is **Linear Regression** with a validation RMSE of **4.8912 hours**.
3. The depths with the highest bias are 1 and 2 and the depths with the highest variance are 12 and unlimited. The best depth is **6** with a validation RMSE of **6.2796**
4. 0.1 is the best learning rate with the highest validation R2 score of **0.9306**. With a learning rate of 0.01, the model is learning too slowly and with a learning rate of 0.1, the model overshoots.
5. Although Gradient Boosting achieved better cross-validation results, the difference was relatively small. This suggests that both models generalize well, but the final choice remains Linear Regression based on the established validation selection rule.