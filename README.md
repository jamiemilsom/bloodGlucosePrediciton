# BrisT1D Blood Glucose Prediction Competition

Using historical blood glucose readings, insulin dosage, carbohydrate intake, and smartwatch activity data to predict future blood glucose.
## Overview

Predicting blood glucose fluctuations is crucial for managing type 1 diabetes. Developing effective algorithms for this can alleviate some of the challenges faced by individuals with the condition.

## Goal: Forecast blood glucose levels one hour ahead using the previous six hours of participant data.

Type 1 diabetes is a chronic condition in which the body no longer produces the hormone insulin and therefore cannot regulate the amount of glucose (sugar) in the bloodstream. Without careful management, this can be life-threatening and so those with the condition need to inject insulin to manage their blood glucose levels themselves. There are many different factors that impact blood glucose levels, including eating, physical activity, stress, illness, sleep, alcohol, and many more, so calculating how much insulin to give is complex. The continuous need to think about how an action may impact blood glucose levels and what to do to counteract it is a significant burden for those with type 1 diabetes.

An important part of type 1 diabetes management is working out how blood glucose levels are going to change in the future. Algorithms of varying levels of complexity have been developed that perform this prediction but the messy nature of health data and the numerous unmeasured factors mean there is a limit to how effective they can be. This competition aims to build on this work by challenging people to predict future blood glucose on a newly collected dataset.
The Dataset

The data used in this competition is part of a newly collected dataset of real-world data collected from young adults in the UK who have type 1 diabetes. All participants used continuous glucose monitors, and insulin pumps and were given a smartwatch as part of the study to collect activity data. The complete dataset will be published after the competition for research purposes. Some more details about the study can be found in this blog post.
Evaluation

Submissions are evaluated on Root Mean Square Error (RMSE) between the predicted blood glucose levels an hour into the future and the actual values that were then collected.

RMSE is defined as
$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

where y^i is the ith predicted value, yi is the ith true value and n is the number of samples.

The RMSE value is calculated from the bg+1:00(future blood glucose) prediction values in the submission file against the true future blood glucose values. The RMSE values for the public and private leaderboards are calculated from unknown and non-overlapping samples from the submission file across all of the participants.
Submission File

For each ID in the test set, you must predict a blood glucose value an hour into the future. The file should contain a header and have the following format:
```
id,bg+1:00
p01_0,6.3
p01_1,6.3
p01_2,6.3
etc.
```
Prizes

Prizes will be awarded to the five top performing participants (according to the leaderboard ranking of the private leaderboard), who deliver the final model’s software code and associated documentation. The prizes comprise of five Love2Shop Global Rewards electronic voucher codes for the following amounts:
1st Prize: £600
2nd Prize: £400
3rd Prize: £300
4th Prize: £200
5th Prize: £100
Details of what electronic voucher codes can be used for in different countries can be found at on the Love2Shop website (https://business.love2shop.co.uk/rewards/global-rewards/) and their terms of use at https://app.g.codes/terms-of-use.
Timeline

    Opening Date: 18th September 2024 at 17:00 UTC.
    Final Submission Deadline: 29th November 2024 at 23:55 UTC


# Initial Ideas for Glucose Prediction

## 1. Preprocessing & Feature Engineering

- **Handle Missing Values**: Use imputation strategies or drop missing data if necessary.
- **Feature Scaling**: Normalise or standardise features for models like LSTMs and transformers.
- **Create Time-Based Features**: Add features such as hour, day of the week, month, etc.
- **Lag Features**: Add past values as features (e.g., previous day's glucose value).
- **Rolling Statistics**: Calculate moving averages and rolling standard deviations for smoothing.

## 2. Model Training

- **Generative Adversarial Recurrent Neural Network (GARNN)**:
    - Experiment with GANs for generating glucose sequences.
    - Train and tune hyperparameters.
  
- **LSTM (Long Short-Term Memory)**:
    - Build LSTM architecture with multiple layers.
    - Hyperparameter tuning (e.g., hidden units, learning rate).
  
- **GRU (Gated Recurrent Unit)**:
    - Train GRU as a simpler alternative to LSTM.
    - Compare performance with LSTM.

- **Transformer-Based Model**:
    - Implement a time-series transformer model (e.g., Time-Series Transformer).
    - Focus on tuning attention heads, layers, and learning rate.

## 3. Hyperparameter Tuning & Cross-Validation

- **5-Fold Cross-Validation**: Use 5-fold cross-validation for each model to find the best hyperparameters.
- **Hyperopt/Bayesian Optimisation**: Use for efficient hyperparameter search.
- **Train 10 Best Models**: Train 10 different hyperparameter configurations for each model.

## 4. Model Evaluation

- **Time-Series Cross-Validation**: Use TimeSeriesSplit or other time-series specific cross-validation techniques.
- **Overfitting Monitoring**: Use early stopping, regularisation, and dropout to avoid overfitting.

## 5. Model Averaging & Stacking

- **Model Averaging**:
    - Average the predictions of 10 best models from each model type.

- **Stacking**:
    - Use model predictions as features for a meta-model (e.g., logistic regression or simple NN).

## 6. Feature Importances

- **Gradient Boosted Models**: Analyse feature importance from models like LightGBM and CatBoost.
- **Integrate Important Features**: Incorporate important features into deep learning models.

## 7. Model Ensembling

- **Train Base Models**: Train different base models (e.g., LSTM, GRU, GARNN, transformer).
- **Combine Outputs**: Combine model outputs via stacking, bagging, or boosting.
  
- **Final Stage NN**: 
    - Train a neural network to combine predictions from all models.

## 8. Model Deployment (Optional)

- **Save Models**: Save models in a standardised format (e.g., ONNX, Pickle).
- **Deploy on Cloud**: Deploy models on cloud platforms (e.g., AWS, GCP) for scalable prediction.

## 9. Monitoring & Iteration

- **Monitor Model Performance**: Track performance on public leaderboard and private test set.
- **Iterate Models**: Fine-tune based on performance and adjust strategy if needed.

## 10. Final Evaluation & Submission

- **Evaluate Final Model**: Evaluate final model on holdout/test data.
- **Generate Predictions**: Generate final predictions and submit.
