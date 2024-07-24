import os
import sys
import requests
import PyPDF2
import s2_utils as s2_utils
from dotenv import load_dotenv

# Config module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config_loader import load_config
config = load_config()

# Load Environmental Variables
load_dotenv()

s2_api_key = os.getenv('S2_API_KEY')
graph_url = config['s2']['base_url'][0]

# User Input for Search Parameters
query = input("Enter a search query: ")
start_date = input("How far back do you want to go? (YYYY-MM-DD): ")
sort = "publicationDate:dsc"; # Sort by publication date in descending order

r = requests.post(
    'https://'
)