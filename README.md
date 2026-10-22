# Delhivery Last Mile Delay Intelligence Pipeline

## Part A - Findings

1. The Express service type is delayed the most with a delay rate of 48.97% (119/243 shipments are delayed).
2. The delay rate is the least for tier 1 at 4% and increases to 21.5% in tier 2 and 44% in tier 3 hubs
3. The delay rate also increases from 0-200 to 601+ with values ranging from 2.19% to 85.23%
4. Missed hours, Distance KM and n_stops are the top 3 columns that are most correlated with delivery_hours with values 0.84, 0.40 and 0.32 respectively.
5. declared_value_inr has very large numbers that can skew the model and the data. Log transformation brings it down to usable or normalized values.