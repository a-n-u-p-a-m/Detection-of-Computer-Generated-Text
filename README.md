# Detection-of-Computer-Generated-Text
Drive link: [https://googledrive.com/host/1OKB9eZXvUNouIlmOBhsNCZo7po68xNvI](url)

The folder corressponding to above link contains code and dataset for the project "Detection of Computer-generated text".

## FILES:
/data/splits: contains the training(train.csv), validation(validation.csv), testing(test.csv) and custom(custom.csv) datasets

/data: contains the .csv files displaying the output

DL_MODELS.py: definition of architecture of the deep learning models

Datahandler.py: contains function to read csv files and function for training and testing machine learning models

1.MINI_PROJECT_ML.ipynb: implementation of Machine Learning classifiers using TF-IDF vectorizer for the task on Tweepfake dataset

2.MINI_PROJECT_BERT_ML.ipynb: implementation of Machine Learning classifiers using pre-trained BERT embeddings

3.MINI_PROJECT_DL.ipynb: implementation of Deep Learning models for the task on Tweepfake dataset

4.Architecture.ipynb: plotting the architecture of the Deep Learning models

5.MINI_PROJECT_BERT.ipynb: implementation of BERT pre-trained transformer model for the task on Tweepfake dataset

6.MINI_PROJECT_RobertA.ipynb: implemenation of RobertA pre-trained transformer model for the task on Tweepfake dataset

7.MINI_PROJECT_COMPARISON.ipynb: plotting and comparing the results of all the models

## RUN ON CUSTOM DATASET:
To run on custom inputs, add the custom tweets to the custom.csv file and then run the .ipynb notebook of the corresponding model on the custom.csv dataset and check the prediction in custom_predictions.<model>.csv. Add the following snippet after the model.

# Use the trained model to predict labels for custom tweets
```
y_pred_custom_prob = model.predict(custom_features)
y_pred_custom = (y_pred_custom_prob > 0.5).astype(int)
def toBoolValue(v):
  if v == True:
    return 1
  else:
    return 0
y_pred_custom = [toBoolValue(t) for t in y_pred_custom]

# Map predictions to label names
custom_prediction_labels = [dictLabelsReverse[t] for t in y_pred_custom]

# Add the predicted labels to the custom tweets DataFrame
customTweetsDataset["prediction"] = custom_prediction_labels

# Save the custom tweets DataFrame to a CSV file
customTweetsDataset.to_csv("/content/drive/MyDrive/MINI_PROJECT/data/custom_predictions_charCNN.csv", sep='\t', encoding='utf-8')

```
