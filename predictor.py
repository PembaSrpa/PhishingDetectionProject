import featuresExt as fe  # Importing the feature extraction module
import database as db  # Importing the database module for URL management
import pickle  # For loading the pre-trained model

# Summary:
# This script classifies a URL as either "Phishy" or "Legitimate" using a pre-trained model.
# It first checks a blacklist to see if the URL is known to be malicious. If not found,
# it extracts features from the URL, uses the model to make a prediction, and updates the 
# blacklist if the URL is classified as "Phishy".

model_filename = "model.pkl"  # Filename for the pre-trained model

# Load the pre-trained model from a file
with open(model_filename, 'rb') as f:
    model = pickle.load(f)  # Unpickle the model object

def classifyURL(url):
    """Classify a URL and interact with the database as needed."""
    # Check if the URL is in the blacklist
    if db.search_url(url):
        return "Phishy"  # Return "Phishy" if the URL is blacklisted
    else:
        # Extract features from the URL and classify it using the model
        df = fe.featureExtraction(url)  # Extract features from the URL
        res = int(model.predict([df]))  # Make a prediction based on the features
        if res:
            db.add_url(url)  # Add to blacklist if classified as "Phishy"
            return "Phishy"  # Return "Phishy"
        return "Legitimate"  # Return "Legitimate" if classified as safe
    # Example: Classify a known URL (e.g., youtube.com)