import datetime  # For handling date and time
import ipaddress  # For validating and manipulating IP addresses
import time  # For measuring execution time
from typing import List  # For type hinting with lists
from urllib.parse import urlencode, urlparse  # For URL manipulation
import pandas as pd  # For data manipulation and analysis
import urllib  # For URL handling
import whois  # For retrieving WHOIS information about domains
import re  # For regular expressions
from bs4 import BeautifulSoup  # For parsing HTML and XML documents
import requests  # For making HTTP requests

# Summary: 
# This code defines functions for extracting features from URLs that can help identify phishing attempts. 
# It analyzes various characteristics of the URL, such as its structure, components, and domain information, 
# to produce a feature set that can be used for further classification or machine learning tasks.

# Keywords often found in phishing URLs
HINTS = [
    'wp', 'login', 'includes', 'admin', 'content', 'site', 'images', 'js', 
    'alibaba', 'css', 'myaccount', 'dropbox', 'themes', 'plugins', 'signin', 
    'view', 'verification', 'account', 'update', 'secure', 'webscr', 'paypal', 'bank'
]

def url_length(url):
    # Return the length of the URL
    return len(url)

def having_ip_address(url):
    # Check if the URL contains an IP address
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.' 
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)|'
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}|'
        '[0-9a-fA-F]{7}', url)
    return 1 if match else 0

def count_dots(hostname):
    # Count the number of dots in the hostname
    return hostname.count('.')

def count_hyphens(base_url):
    # Count the number of hyphens in the URL
    return base_url.count('-')

def count_at(base_url):
    # Count the number of '@' symbols in the URL
    return base_url.count('@')

def count_qm(hostname):
    # Count the number of '?' symbols in the URL
    return hostname.count('?')

def count_and(base_url):
    # Count the number of '&' symbols in the URL
    return base_url.count('&')

def count_equal(base_url):
    # Count the number of '=' symbols in the URL
    return base_url.count('=')

def count_percentage(base_url):
    # Count the number of '%' symbols in the URL
    return base_url.count('%')

def count_slash(full_url):
    # Count the number of slashes in the URL
    return full_url.count('/')

def count_colon(url):
    # Count the number of ':' symbols in the URL
    return url.count(':')

def count_semicolumn(url):
    # Count the number of ';' symbols in the URL
    return url.count(';')

def check_www(words_raw):
    # Check if 'www' is in the domain parts of the URL
    return sum(1 for word in words_raw if 'www' in word)

def check_com(url):
    # Check how many '.com' are in the URL
    return url.count('.com')

def count_double_slash(full_url):
    # Check for double slashes after the 6th character
    return 1 if re.search('//', full_url) and full_url.index('//') > 6 else 0

def https_token(scheme):
    # Return 0 if the URL uses HTTPS; otherwise return 1
    return 0 if scheme == 'https' else 1

def prefix_suffix(url):
    # Check if the URL has a hyphen in between parts
    return 1 if re.findall(r"https?://[^\-]+-[^\-]+/", url) else 0

def phish_hints(url_path):
    # Count the number of phishing hints in the URL path
    return sum(url_path.lower().count(hint) for hint in HINTS)

def shortening_service(full_url):
    # Check if the URL is from a known shortening service
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                    'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                    'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                    'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                    'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                    'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                    'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                    'tr\.im|link\.zip\.net',
                    full_url)
    return 1 if match else 0

def whois_registered_domain(url):
    # Check if the domain is registered and matches the input URL
    try:
        domain = urlparse(url).netloc
        domain_name = whois.whois(domain)
        domainResponse = domain_name.domain_name
        if isinstance(domainResponse, list):
            return 1 if all(not re.search(host.lower(), domain) for host in domainResponse) else 0
        else:
            return 1 if not re.search(domainResponse.lower(), domain) else 0
    except:
        return 1

def google_index(url):
    # Check if the URL is indexed by Google
    time.sleep(.6)  # Pause to avoid hitting Google too fast
    user_agent = 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.116 Safari/537.36'
    headers = {'User-Agent': user_agent}  # Use a user-agent to simulate a browser
    query = {'q': 'info:' + url}
    google = "https://www.google.com/search?" + urlencode(query)
    data = requests.get(google, headers=headers)  # Send a request to Google
    soup = BeautifulSoup(str(data.content), "html.parser")  # Parse the response
    try:
        check = soup.find(id="rso").find("div").find("div").find("h3").find("a")
        check['href']  # Attempt to find the link in search results
        return 0  # URL is indexed
    except AttributeError:
        return 1  # URL is not indexed

def domainAge(url):
    # Calculate the age of the domain
    try:
        domain = urlparse(url).netloc
        domain_name = whois.whois(domain)
        creation_date = domain_name.creation_date
        expiration_date = domain_name.expiration_date
        if (expiration_date is None) or (creation_date is None):
            return -1  # If either date is missing, return -1
        creation_date = creation_date[0] if isinstance(creation_date, list) else creation_date
        expiration_date = expiration_date[0] if isinstance(expiration_date, list) else expiration_date
        return abs((expiration_date - creation_date).days)  # Return the age in days
    except:
        return -1  # Return -1 if there's an error

def web_traffic(short_url):
    # Get the web traffic rank of the URL using Alexa
    try:
        rank = BeautifulSoup(urllib.request.urlopen("http://data.alexa.com/data?cli=10&dat=s&url=" + short_url).read(), "xml").find("REACH")['RANK']
        return int(rank)  # Return the traffic rank
    except:
        return 0  # Return 0 if there's an error

def featureExtraction(url):
    # Extract various features from the URL for classification
    parsed = urlparse(url)  # Parse the URL
    scheme = parsed.scheme  # Get the URL scheme (http or https)
    domain = urlparse(url).netloc  # Get the domain name
    # Split the domain into parts for further analysis
    words_raw = re.split("\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", domain.lower())
    features = []  # List to hold the features extracted
    features.append(url_length(url))  # Add URL length
    features.append(having_ip_address(url))  # Check for IP address
    features.append(count_dots(url))  # Count dots in the URL
    features.append(count_hyphens(url))  # Count hyphens in the URL
    features.append(count_at(url))  # Count '@' symbols
    features.append(count_qm(url))  # Count '?' symbols
    features.append(count_and(url))  # Count '&' symbols
    features.append(count_equal(url))  # Count '=' symbols
    features.append(count_percentage(url))  # Count '%' symbols
    features.append(count_slash(url))  # Count slashes
    features.append(count_colon(url))  # Count colons
    features.append(count_semicolumn(url))  # Count semicolons
    features.append(check_www(words_raw))  # Check for 'www'
    features.append(check_com(url))  # Check for '.com'
    features.append(count_double_slash(url))  # Check for double slashes
    features.append(https_token(scheme))  # Check if HTTPS is used
    features.append(prefix_suffix(url))  # Check for hyphens in parts
    features.append(phish_hints(url))  # Count phishing hints
    features.append(shortening_service(url))  # Check for URL shortening service
    features.append(whois_registered_domain(url))  # Check if domain is registered
    features.append(domainAge(url))  # Get domain age
    features.append(web_traffic(url))  # Get web traffic rank
    return features  # Return the list of features
