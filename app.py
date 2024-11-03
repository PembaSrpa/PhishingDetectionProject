from flask import Flask, redirect, url_for, request, render_template  # Importing Flask and related functions
import predictor as pr  # Importing the predictor module for URL classification
import database as db  # Importing the database module for URL blacklist management

# Summary:
# This Flask application allows users to check URLs for phishing.
# Users can submit a URL, which is classified as either "Phishy" or "Legitimate".
# The application also maintains a blacklist of URLs that have been classified as phishing.

app = Flask(__name__)  # Creating a Flask application instance

db.init_db(app)  # Initializing the database with the Flask app

@app.route('/')  # Route for the homepage
def index():
    return render_template('index.html')  # Render the index template

@app.route('/catch_phish', methods=['POST', 'GET'])  # Route for the URL submission form
def catch_phish():
    if request.method == 'POST':  # Check if the request method is POST
        url = request.form.get('url')  # Get the URL from the submitted form
        print(url)  # Print the URL to the console (for debugging purposes)
        return redirect(url_for('results', url=url))  # Redirect to the results page with the URL
    return render_template('check.html')  # Render the URL check form template

@app.route('/results', methods=['POST', 'GET'])  # Route for displaying classification results
def results():
    url = request.args.get('url', None)  # Get the URL from the query parameters
    res = pr.classifyURL(url)  # Classify the URL using the predictor module
    return render_template('results.html', url=url, res=res)  # Render the results template with URL and classification result

if __name__ == '__main__':  # Check if the script is being run directly
    app.run(debug=True)  # Start the Flask application in debug mode
