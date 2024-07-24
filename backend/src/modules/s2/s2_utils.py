import requests
import os
import logging
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from dotenv import load_dotenv

# Adjust the path to the config module
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Refactor?

from config.config_loader import load_config
from fields import (GRAPH_API_PAPER_FIELDS, GRAPH_API_AUTHOR_FIELDS, GRAPH_API_CITATION_FIELDS,
                    GRAPH_API_REFERENCE_FIELDS, RECOMMENDATIONS_API_FIELDS)

# Load environment variables from .env file
load_dotenv()
# Load configuration from config.yaml file
config = load_config()

# Set up Logging
logging.basicConfig(
    level=config['logging']['level'],
    format=config['logging']['format'],
    handlers=[
        logging.FileHandler(config['logging']['files'][0]),
        logging.FileHandler(config['logging']['files'][1])
    ]
)
logger = logging.getLogger(__name__)

# Ensure you have your API key stored securely, e.g., in an environment variable
API_KEY = os.getenv('S2_API_KEY')
BASE_URL = 'https://api.semanticscholar.org'

class Datastore:
    def __init__(self):
        self.data = {}

    def upsert(self, corpus_id, record):
        self.data[corpus_id] = record
        logger.info(f"Upserted record with corpus_id: {corpus_id}")

    def delete(self, corpus_id):
        if corpus_id in self.data:
            del self.data[corpus_id]
            logger.info(f"Deleted record with corpus_id: {corpus_id}")
        else:
            logger.warning(f"Tried to delete non-existent corpus_id: {corpus_id}")

datastore = Datastore()

class S2APIBase:
    def __init__(self):
        self.base_url = BASE_URL
        self.headers = {'x-api-key': API_KEY}
        self.session = self._create_session()

    def _create_session(self):
        session = requests.Session()
        retries = Retry(total=config['retry']['max_attempts'], 
                        backoff_factor=config['retry']['delay'],
                        status_forcelist=[429, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        return session

    def _get(self, endpoint, params=None):
        try:
            response = self.session.get(f"{self.base_url}{endpoint}", headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"GET request failed: {e}")
            return None

    def _post(self, endpoint, json_data=None, params=None):
        try:
            response = self.session.post(f"{self.base_url}{endpoint}", headers=self.headers, json=json_data, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"POST request failed: {e}")
            return None


class GraphAPI(S2APIBase):
    """Class to interact with the Semantic Scholar Graph API."""

    def get_paper_details(self, paper_id, fields=GRAPH_API_PAPER_FIELDS):
        params = {'fields': ','.join(fields)}
        return self._get(f"/graph/v1/paper/{paper_id}", params=params)

    def get_author_details(self, author_id, fields=GRAPH_API_AUTHOR_FIELDS):
        params = {'fields': ','.join(fields)}
        return self._get(f"/graph/v1/author/{author_id}", params=params)

    def search_papers(self, query, offset=0, limit=100, fields=GRAPH_API_PAPER_FIELDS):
        params = {'query': query, 'offset': offset, 'limit': limit, 'fields': ','.join(fields)}
        return self._get("/graph/v1/paper/search", params=params)

    def get_paper_authors(self, paper_id, fields=GRAPH_API_AUTHOR_FIELDS):
        params = {'fields': ','.join(fields)}
        return self._get(f"/graph/v1/paper/{paper_id}/authors", params=params)

    def get_paper_references(self, paper_id, fields=GRAPH_API_REFERENCE_FIELDS):
        params = {'fields': ','.join(fields)}
        return self._get(f"/graph/v1/paper/{paper_id}/references", params=params)

    def get_paper_citations(self, paper_id, fields=GRAPH_API_CITATION_FIELDS):
        params = {'fields': ','.join(fields)}
        return self._get(f"/graph/v1/paper/{paper_id}/citations", params=params)

    def get_author_papers(self, author_id, offset=0, limit=100, fields=GRAPH_API_PAPER_FIELDS):
        params = {'offset': offset, 'limit': limit, 'fields': ','.join(fields)}
        return self._get(f"/graph/v1/author/{author_id}/papers", params=params)

    def batch_get_paper_details(self, paper_ids, fields=GRAPH_API_PAPER_FIELDS):
        params = {'fields': ','.join(fields)}
        payload = {'ids': paper_ids}
        return self._post("/graph/v1/paper/batch", json_data=payload, params=params)

    def batch_get_author_details(self, author_ids, fields=GRAPH_API_AUTHOR_FIELDS):
        params = {'fields': ','.join(fields)}
        payload = {'ids': author_ids}
        return self._post("/graph/v1/author/batch", json_data=payload, params=params)

    def autocomplete_paper_query(self, query):
        params = {'query': query}
        return self._get("/graph/v1/paper/autocomplete", params=params)

    def search_authors(self, query, offset=0, limit=100, fields=GRAPH_API_AUTHOR_FIELDS):
        params = {'query': query, 'offset': offset, 'limit': limit, 'fields': ','.join(fields)}
        return self._get("/graph/v1/author/search", params=params)

    def search_paper_by_title(self, query, fields=GRAPH_API_PAPER_FIELDS):
        params = {'query': query, 'fields': ','.join(fields)}
        return self._get("/graph/v1/paper/search/match", params=params)

    def bulk_search_papers(self, query=None, token=None, fields=GRAPH_API_PAPER_FIELDS):
        params = {'query': query, 'token': token, 'fields': ','.join(fields)}
        return self._get("/graph/v1/paper/search/bulk", params=params)

    def get_paper_embedding(self, paper_id, model_version='specter_v1'):
        params = {'model': model_version}
        return self._get(f"/graph/v1/paper/{paper_id}/embedding", params=params)

    def get_paper_tldr(self, paper_id):
        return self._get(f"/graph/v1/paper/{paper_id}/tldr")


class RecommendationsAPI(S2APIBase):
    """Class to interact with the Semantic Scholar Recommendations API."""

    def get_paper_recommendations(self, paper_id, limit=100, fields=RECOMMENDATIONS_API_FIELDS):
        params = {'limit': limit, 'fields': ','.join(fields)}
        return self._get(f"/recommendations/v1/papers/forpaper/{paper_id}", params=params)

    def get_paper_recommendations_from_list(self, positive_paper_ids, negative_paper_ids=None, limit=100, fields=RECOMMENDATIONS_API_FIELDS):
        payload = {'positivePaperIds': positive_paper_ids, 'negativePaperIds': negative_paper_ids}
        params = {'limit': limit, 'fields': ','.join(fields)}
        return self._post("/recommendations/v1/papers", json_data=payload, params=params)

    def get_author_recommendations(self, author_id, limit=100, fields=RECOMMENDATIONS_API_FIELDS):
        params = {'limit': limit, 'fields': ','.join(fields)}
        return self._get(f"/recommendations/v1/papers/forauthor/{author_id}", params=params)

    def get_author_recommendations_from_list(self, positive_author_ids, negative_author_ids=None, limit=100, fields=RECOMMENDATIONS_API_FIELDS):
        payload = {'positiveAuthorIds': positive_author_ids, 'negativeAuthorIds': negative_author_ids}
        params = {'limit': limit, 'fields': ','.join(fields)}
        return self._post("/recommendations/v1/papers/forauthors", json_data=payload, params=params)

    def get_trending_papers(self, limit=100, fields=RECOMMENDATIONS_API_FIELDS):
        params = {'limit': limit, 'fields': ','.join(fields)}
        return self._get("/recommendations/v1/papers/trending", params=params)


class DatasetsAPI(S2APIBase):
    """Class to interact with the Semantic Scholar Datasets API."""
        
    def get_releases(self):
        return self._get("/datasets/v1/release")

    def get_latest_release(self):
        release = self._get("/datasets/v1/release/latest")
        return release['release_id'] # release

    def get_release(self, release_id):
        return self._get(f"/datasets/v1/release/{release_id}")

    def get_dataset(self, release_id, dataset_name):
        return self._get(f"/datasets/v1/release/{release_id}/dataset/{dataset_name}")

    def get_diff(self, start_release_id, end_release_id, dataset_name):
        return self._get(f"/datasets/v1/diffs/{start_release_id}/to/{end_release_id}/{dataset_name}")
      
    # def update_dataset(self, start_release_id, dataset_name):
    #     s3file = {} # move to global variable?
    #     datastore = s3file
    #     start_release_id = s3file['release_id'] # placeholder for file name in s3 bucket
    #     difflist = self._get(f"/datasets/v1/diffs/{start_release_id}/to/latest/{dataset_name}").json()
    #     for diff in difflist['diffs']:
    #         for url in diff['update_files']:
    #             for json_line in requests.get(url).iter_lines():
    #                 record = json.loads(json_line)
    #                 datastore.upsert(record['corpusid'], record)
    #         for url in diff['delete_files']:
    #             for json_line in requests.get(url).iter_lines():
    #                 record = json.loads(json_line)
    #                 datastore.delete(record['corpusid'])
                    
    # import requests

    def init_dataset(self):
        # Get the latest release
        release_id = self.get_latest_release()
        
        # Get the release details
        release_details = self.get_release(release_id)
        datasets = release_details['datasets']
        
        database_files = {}
        # Get details for each dataset in the release
        for dataset in datasets:
            dataset_name = dataset['name']
            dataset_details = self.get_dataset(release_id, dataset_name)
            database_files[dataset_name] = {
                'name': dataset['name'],
                'description': dataset['description'],
                'README': dataset['README'],
                'files': dataset_details['files']
            }
        
        # Save the compiled details to a JSON file
        with open('database_files.json', 'w') as f:
            json.dump(database_files, f, indent=2)
        
        return database_files

    
# def get_latest_release ():
#     latest_id = requests.get('https://api.semanticscholar.org/datasets/v1/release/latest').json() 
#     return (latest_id['release_id'])
# print(get_latest_release())

if __name__ == "__main__":
    datasets_api = DatasetsAPI()
    database_files = datasets_api.init_dataset()
    print(database_files)