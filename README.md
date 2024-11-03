# PhishingDetectionProject
#Project Summary: Phishing URL Detection Web Application
Overview: This project implements a web application to detect phishing URLs using machine learning models. It comprises several components, including data extraction, model training, a database to store blacklisted URLs, a prediction module, and a Flask web application for user interaction.

## Components
### Data Collection:

File: dataset.csv
Description: This CSV file contains features extracted from URLs along with their classifications as either legitimate or phishing.

### Feature Extraction:

File: featuresExt.py
Functionality: This module includes functions that extract relevant features from a given URL, which will be used for classification. The details of the features extracted are determined by the specific implementation of the module.

### Model Training:

File: model.py
Libraries Used: pickle, pandas, numpy, matplotlib, seaborn, sklearn, xgboost.
Functionality: This script:
Loads the dataset and preprocesses it.
Trains three machine learning models: Random Forest, XGBoost, and Decision Tree.
Utilizes Grid Search for hyperparameter tuning.
Evaluates model performance based on accuracy, precision, recall, F1 score, and confusion matrices.
Saves the best-performing model using pickle.

### Database Management:

File: database.py
Libraries Used: flask_sqlalchemy, os.
Functionality:
Defines a Blacklist model to manage URLs identified as phishing.
Initializes a SQLite database and provides functions to add new URLs to the blacklist and check for existing URLs.

### Prediction:

File: predictor.py
Libraries Used: pickle, featuresExt, database.
Functionality:
Loads the pre-trained model.
Provides a function classifyURL(url) that checks if a URL is in the blacklist and, if not, extracts features from the URL and classifies it as either "Phishy" or "Legitimate".
Adds the URL to the blacklist if classified as "Phishy".

### Web Application:

File: app.py
Libraries Used: flask, predictor, database.
Functionality:
Creates a Flask web application with routes for:
The homepage (index.html).
URL submission (check.html).
Displaying classification results (results.html).
Allows users to submit URLs and view whether they are classified as phishing or legitimate.

### Visualization:

File: visual.py
Libraries Used: pandas, numpy, matplotlib, seaborn, warnings, os.
Functionality:
Performs exploratory data analysis (EDA) by generating various visualizations, including scatter plots, count plots, a correlation heatmap, and distribution plots.
Saves generated plots to a "diagrams" folder for later use.

## Required Libraries

### The following Python libraries are required to run this project:

Flask
Flask-SQLAlchemy
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
XGBoost
BeautifulSoup
Requests
Whois
Pickle

## Steps to Run the Web Application

### Set Up Your Environment:

Make sure you have Python installed on your machine (Python 3.7 or later recommended).

### Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
or just use command prompt (locate the dir to where your current code is)

### Install Required Libraries:

Use pip to install the required libraries:

pip install -r requirements.txt

### Prepare the Dataset:

Ensure the dataset.csv file is available in the project directory.

### Feature Extraction:

Run the featuresExt.py script (if necessary) to ensure all features are correctly extracted and ready for model training.

### Train the Model:

Execute the model.py script to train the models and save the best model as model.pkl.

### Initialize the Database:

Run the app.py script. This will also call database.py to set up the database:

python app.py

### Access the Web Application:

Open your web browser and go to http://127.0.0.1:5000/ to access the web application.

Use the form to submit URLs and check their classification status.

### Visualizations:

Optionally, run visual.py to generate and save visualizations of the dataset.

## Conclusion:

This project provides a comprehensive system for detecting phishing URLs, including data preprocessing, model training, database management, and user interaction through a web interface. By following the above steps, users can effectively run and utilize the application for phishing detection.

