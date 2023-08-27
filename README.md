# Sales prediction project
Author: Mariana Taglio
This repository contains scripts and classes for a machine learning project that focuses on transforming raw data, building a linear regression model, and making predictions. The project is organized into several scripts and classes to facilitate modularity and reusability.

## Project Structure
All the source code is located in the src folder.
The project is structured as follows:

- `transformers.py`: This script defines a seveeral classes that inherit from sklearn BaseEstimator and TransformerMixin that encapsulate a set of data transformation methods. These methods can be applied to preprocess raw data before feeding it into the model. The transformations included are:
* PriceBucketTransformer
* YearTransformer
* ModeImputationByGroup
* Null Imputer
* OrdinalEncodingTransformer
* OneHotEncoder
* RemoveColumns

- `feature_engineering.py`: This script defines a class named `FeatureEngineeringPipeline` that utilizes the transformation methods from `transformers.py`. It performs data preprocessing using these transformations and saves the resulting transformed dataset in a CSV file. Additionally, it saves the transformation pipeline as a pickle (`.pkl`) file for later use.

- `train.py`: This script contains a class named `ModelTrainingPipeline` that reads the transformed data, splits it into training and testing sets, fits a linear regression model to the training data, evaluates the model's performance on the test data, and appends the trained model to the saved transformation pipeline in the `.pkl` file.

- `predict.py`: This script defines a class named `MakePredictionPipeline` that reads raw data, loads a trained model from the saved pipeline in the `.pkl` file, makes predictions for new data, and saves the predictions in a new file.

- `train_pipeline.py`: This script streamlines the machine learning pipeline for training a model. It takes as input an input raw data file, a directory for storing transformed data files, a model file path, and a log file path. This script orchestrates the following steps:

1. **Feature Engineering**: It invokes the `feature_engineering.py` script to preprocess the raw data. The transformed dataset is saved in the specified directory for further use.

2. **Model Training**: The script then runs the `train.py` script to train a model using the transformed data. The trained model is saved in the specified model file.

3. **Logging**: Throughout the pipeline execution, detailed logs are written to the specified log file, aiding in tracking the progress and diagnosing issues.

- `inference_pipeline.py`: This script automates the process of making predictions on new, unseen data using a trained machine learning model. It takes as input the paths to the transformed data, a trained model file, an output path for saving prediction results, and a log file to store execution logs. This script orchestrates the following steps:

1. **Prediction Generation**: It calls the `predict.py` script, providing the path to the transformed data, the trained model file, and the output path for saving prediction results. The script loads the model, makes predictions on the new data, and saves the predictions in the specified output path.

2. **Logging**: Throughout the prediction process, detailed execution logs are written to the specified log file.

## Usage

1. Begin by placing your raw data in `data directory`
2. To use the train pipeline for preprocessing data and training a model, follow these steps:

    b) Run the `train_pipeline.py` script with the necessary command-line arguments:

   ```bash
     python train_pipeline.py --input-file <input_file_name> --tfmd-dir <transformed_files_dir> --model-file <model_file_path> --log-file <log_file_path>
     ```

    - input-file: Name of the input raw data file located in the DATA_PATH.
    - tfmd-dir: Path to the directory where transformed CSV and pipeline pickle (pkl) files will be saved.
    - model-file: Path where the trained model will be saved as a pickle (pkl) file.
    - log-file: Path to the log file where execution logs will be stored.


3. To utilize the inference pipeline for making predictions, follow these steps:

    a. Ensure you have the transformed data, a trained model file, and an output path for predictions ready.

    b. Run the `inference_pipeline.py` script with the necessary command-line arguments:

   ```bash
   python inference_pipeline.py --input-path <transformed_data_path> --models-path <model_file_path> --output-path <predictions_output_path> --log-file <log_file_path>
   ```

    - input-path: Path to the transformed data that you want to make predictions on.
    - models-path: Path to the trained model pickle (pkl) file.
    - output-path: Path where the prediction results will be saved.
    - log-file: Path to the log file where execution logs will be stored.

## Dependencies

- Python (>=3.6)
- Required libraries are listed in the `requirements.txt` file. You can install them using the command: pip install -r requirements.txt

## Hyperparameter optimization

The repository also includes a notebook named hyperparameter_optimization.ipynb that focuses on training several XGBoost models and performing hyperparameter optimization using grid search. The notebook then proceeds to plot various graphics for visualizing the results of the optimization process.

## Conclusion

This machine learning project provides a structured framework for data transformation, feature engineering, model training, and prediction. It promotes modularity, reusability, and easy experimentation with different transformations and models. By following the provided guidelines, users can preprocess data, train models, and make predictions efficiently.


