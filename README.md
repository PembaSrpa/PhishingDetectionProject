# Phishing URL Detection Project - Usage Guide

This repository contains a full-stack machine learning solution for detecting phishing websites. This guide focuses specifically on how to install, configure, and operate the application.

---

## 🛠 Installation & Setup

Before running the application, you must set up your local environment and install the necessary dependencies.

### 1. Clone the Repository
Open your terminal or command prompt and run:
```bash
git clone [https://github.com/your-username/PhishingDetectionProject.git](https://github.com/your-username/PhishingDetectionProject.git)
cd PhishingDetectionProject
```

### 2. Create a Virtual Environment (Recommended)
To prevent library conflicts, create a isolated environment:
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install the required Python libraries using the provided requirements file:
```bash
pip install -r requirements.txt
```
*Key libraries being installed: Flask, Scikit-learn, XGBoost, Pandas, and BeautifulSoup4.*

---

## 🚀 How to Use the Application

### Step 1: Initialize the Machine Learning Model
The web application requires a pre-trained model file (`model.pkl`) to make predictions. You must generate this by training the model on the provided dataset.

```bash
python model.py
```
**What happens here:** The script loads `dataset.csv`, extracts features, trains a high-accuracy XGBoost classifier, and saves the result as `model.pkl`. You only need to run this once unless you update the dataset.

### Step 2: Start the Web Server
Launch the Flask application to start the user interface and initialize the SQLite database.

```bash
python app.py
```
**What happens here:**
The script starts a local server. It also calls `database.py` to create a `phishing.db` file (if it doesn't exist) to store your URL blacklist.

### Step 3: Access the Interface
1. Open your web browser.
2. Navigate to: `http://127.0.0.1:5000/`
3. You will see the Phishing Detection homepage.

### Step 4: Scanning a URL
1. Copy a suspicious URL you want to test.
2. Paste it into the input field on the **index.html** page.
3. Click the **"Check"** button.
4. The system will process the URL and redirect you to **results.html**, displaying whether the link is **"Phishy"** or **"Legitimate"**.



---

## 🔧 Advanced Usage & Features

### Managing the Blacklist
The application includes a built-in protection layer. When the machine learning model identifies a URL as phishing:
* The URL is automatically saved to the **SQLite database**.
* The next time any user checks that specific URL, the app will instantly flag it as "Phishy" via the `database.py` logic without needing to re-run the ML model.

### Data Visualization (Optional)
If you wish to see the statistical analysis of the phishing data before using the tool, run:
```bash
python visual.py
```
This will save several graphs into a `diagrams/` folder. These charts help you visualize how the model distinguishes between safe and malicious links based on features like URL length and HTTPS usage.



---

## 📝 Usage Notes & Troubleshooting

* **Database Reset:** If you wish to clear the blacklist, simply delete the `.db` file in the project directory; the app will recreate an empty one upon the next launch.
* **WHOIS Limits:** The `featuresExt.py` script uses the `whois` library. If you check too many URLs in a very short period, your IP might be temporarily throttled by WHOIS servers.
* **Browser Errors:** Ensure no other service is using port 5000. If you see an "Address already in use" error, close other Flask apps or change the port in `app.run()`.

---
