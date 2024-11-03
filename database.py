from flask_sqlalchemy import SQLAlchemy  # Importing SQLAlchemy for database management
import os  # Importing os for operating system functionalities

# Summary:
# This code defines a SQLAlchemy model for a URL blacklist and functions to initialize the database,
# add new URLs to the blacklist, and check if a URL is already blacklisted. 
# It uses SQLite as the database backend to store the URLs securely.

db = SQLAlchemy()  # Creating an instance of SQLAlchemy

class Blacklist(db.Model):
    """Model for the URL blacklist."""
    __tablename__ = 'blacklist'  # Define the table name
    id = db.Column(db.Integer, primary_key=True)  # Primary key for the blacklist
    url = db.Column(db.String, unique=True, nullable=False)  # Column for storing unique URLs

def init_db(app):
    """Initialize the database with the Flask app context."""
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///url_blacklist.db'  # Database URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable modification tracking
    db.init_app(app)  # Initialize the SQLAlchemy instance with the app

    with app.app_context():  # Use the app context for database operations
        db.create_all()  # Create all tables defined by the models

def add_url(url):
    """Add a new URL to the blacklist."""
    if not search_url(url):  # Check if the URL is already in the blacklist
        new_url = Blacklist(url=url)  # Create a new Blacklist object
        db.session.add(new_url)  # Add the new URL to the session
        db.session.commit()  # Commit the transaction to save the URL

def search_url(url):
    """Check if the URL exists in the blacklist."""
    return db.session.query(Blacklist).filter_by(url=url).first() is not None  # Return True if URL exists

