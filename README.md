
# Battery Manufacturing Yield Analytics

## Project Overview
An end-to-end manufacturing analytics project focused on identifying production bottlenecks, predicting battery cell coating defects and improving overall manufacturing yield through data-driven process optimisation.

The project analyses simulated lithium-ion battery production data across six manufacturing stages: mixing, coating, calendering, assembly, formation and aging.

## Objectives
- Identify production bottlenecks and stage-wise yield losses.
- Predict coating batch failures using machine learning.
- Identify critical process parameters contributing to manufacturing defects.
- Evaluate risk-based inspection strategies to improve production yield.

## Tech Stack
- **Programming:** Python
- **Big Data Processing:** PySpark, Spark SQL
- **Machine Learning:** Scikit-learn, Gradient Boosting
- **Quality Analysis:** Statistical Process Control (SPC), Process Capability (Cpk)
- **Visualisation:** Power BI, Matplotlib, Seaborn

## Methodology

### 1. Data Processing
- Processed 7,200 production records using PySpark.
- Performed data cleaning, transformation and validation.
- Analysed stage-wise production yield and rolled throughput yield.
- Identified coating as the primary manufacturing bottleneck.

### 2. Defect Prediction
- Developed a Gradient Boosting classification model.
- Predicted coating batch failures using process parameters.
- Evaluated model performance using AUC and recall.
- Applied feature importance to identify critical defect drivers.
- Used SPC control charts and Cpk analysis to assess process stability and capability.

### 3. Process Optimisation
- Developed a risk-based batch inspection strategy.
- Identified high-risk batches for targeted inspection and reprocessing.
- Simulated improvements in scrap rates and production yield.
- Evaluated the impact of process interventions on manufacturing performance.

## Key Results

| Performance Metric | Result |
|---|---:|
| Production records analysed | 7,200 |
| Gradient Boosting AUC | 0.80 |
| Failures captured | 46% |
| Coating scrap reduction | 33% |
| Initial rolled throughput yield | 95.2% |
| Improved rolled throughput yield | 96.0% |

## Business Impact
The project demonstrates how manufacturing analytics and predictive modelling can support early defect detection, reduce production scrap, identify process bottlenecks and improve battery manufacturing yield.

## Limitations
- Production data is simulated.
- Model performance reflects the assumptions used during data generation.
- Yield improvements are based on simulated inspection and reprocessing scenarios, not actual factory trials.
