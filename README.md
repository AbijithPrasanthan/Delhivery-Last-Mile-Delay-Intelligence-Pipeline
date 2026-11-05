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

## Model Results (classification)

1. The blind baseline accuracy score is **0.7667**. This is useless because this model did not predict any delayed shipments. 
2. Combined, the importance of the noise features comes out to be about **22.41%**. All 5 of these fall outside the top 7 we selected.
3. Feature selection improved the model performance. Removing less relevant features and sticking to the top 7 improved the performance.
4. The hyperparameter tuning had no effect on the Random Forest's performance. The untuned model had an **F1 score** of **0.83** compared to the **0.77** from **GridSearch** and **0.78** from **Randomized Search**. This tells us that tuning on a 180 row validation set may be useless.
5. The **chosen threshold** is **0.5**. This gave us a **precision** of **0.7955** and a **recall** of **0.8333**. Increasing this threshold by 0.1 increased the precision but dropped the recall to below the required 0.8.
6. The final model achieved a **precision** of **0.6889**, **recall** of **0.7561**, and **F1 score** of **0.7209**. The recall was lower than the validation recall of 0.8333, and with only 41 delayed shipments in the test set, even a few missed predictions can make a noticeable difference.

## How to run

To reproduce the project, run the below commands in the same order:
1. git clone https://github.com/AbijithPrasanthan/Delhivery-Last-Mile-Delay-Intelligence-Pipeline.git
2. cd Delhivery-Last-Mile-Delay-Intelligence-Pipeline
3. pip install -r requirements.txt
4. python generate_data.py
5. python part_a_clean_eda.py
6. python part_b_regression.py
7. python part_c_classification.py
8. python part_d_segments.py
9. python package_model.py - 
10. python predict_delay.py

All code runs on the python package versions mentioned in the requirements file. In case of version mismatches, you can regenerate the model by running package_model.py again.

## Operations Recommendation

Below are the hubs in the Escalate segment with their respective delay rates:

- Nagpur - 46%
- Kochi - 36%
- Patna - 46%
- Ranchi - 38%
- Guwahati - 62%
- Raipur - 48%
- Bhubaneswar - 40%
- Siliguri - 44%
- Agartala - 54%

The hubs with tier and segment mismatches are:
- Kolkata - HUB07
- Nagpur - HUB13
- Kochi - HUB15
- Dehradun - HUB22

This means that the original tier label should not be the only data source for decisions. The actual operation data provides a better indication of where intervention is required.

The alert threshold is **0.5**. The precision on the test was **0.6889** and the recall was **0.7561**. This means about 7 out of 10 flagged shipments were actually delayed and about 8 out of 10 delayed shipments were correctly identified. 

From the 48 new shipments, 8 were flagged. Of these, 6 required escalation to hub manager and 2 required proactive customer calls.

K = 3 was retained because operations requires exactly three playbooks: Standard, Watch and Escalate. Although the across-to-within separation ratio increased from 5.3928 (K = 3) to 6.3047 (K = 4) and 6.8866 (K = 5), the higher K values produced clusters containing fewer than six hubs. K = 4 produced cluster sizes of 8-7-5-4, while K = 5 produced 8-7-4-3-2. These smaller groups would make separate operational playbooks impractical. Therefore, K = 3 provides a reasonable balance between cluster separation and operational requirements.

## Production Plan

Before deployment, the pipeline should satisfy the following production-readiness considerations:

- Accuracy: Classification performance is evaluated using precision, recall and F1 in part_c_final_test.csv.

- Reliability: part_d_reload_check.csv verifies that the saved and reloaded models produce identical probabilities.

- Reproducibility: Fixed random seeds and pinned package versions allow consistent results.

- Maintainability: Shared preprocessing functions in src/shipment_prep.py reduce duplicated code.

- Consistency: Training and prediction use the same cleaning and feature-engineering functions, with stored historical medians.

- Monitorability: The generated prediction and action CSVs provide a record of flagged shipments and operational decisions.

In production, I would monitor these quantities weekly:

- Actual delay rate: Investigate if more than 30% of delivered shipments are delayed.

- Flagging rate: Investigate if more than 30% of shipments are flagged.

- Model recall: Review and retrain the model if recall falls below 0.70 on shipments with known delivery outcomes.

I would retrain the classifier monthly using newly completed deliveries, with additional retraining if monitoring detects significant performance deterioration.

The September 15–28 sale period should be monitored separately using sale_event_flag. Sale-period performance should be compared against similar promotional periods rather than ordinary weeks, since delivery patterns differ during high demand.

Data drift could appear as changes in shipment distances, parcel weights, hub workloads, service-type distribution or delay rates. Such changes may reduce the accuracy of a model trained on earlier data.

## Limitations

1. Synthetic data: This is not real-life data and hence it may not express the complexity of real logistics operations.
2. Small test sample: The classification test set contains only 41 delayed shipments, making metric estimates sensitive to individual prediction errors.
3. Threshold selection: The alert threshold was selected using a single validation split. This may not be perfect for future shipments
4. Limited hub history: Each hub is represented by only 50 historical shipments. This may not accurately capture long term variations in the operations.
5. Static hub segments: The K-Means segments reflect historical performance and do not automatically adapt when hub conditions change. Periodic re-segmentation would be necessary.