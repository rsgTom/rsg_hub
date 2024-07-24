import os
import sys
import json
import asyncio
import logging
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
import tracemalloc
import re
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from http.cookies import SimpleCookie
import aiohttp
from aiohttp import ClientSession, TCPConnector
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv
from tqdm import tqdm
import uuid

tracemalloc.start()

## CONFIG, LOGGING, ENVIRONMENT SETUP
# Dynamically determine the project root directory and add it to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')) # May need to add more ../ to get to the root
if project_root not in sys.path:
    sys.path.append(project_root)

# Import config loader after adding the project root to the path
from config.config_loader import load_config

# Load environment variables and configuration settings
load_dotenv(os.path.join(project_root, '.env'))
config = load_config(os.path.join(project_root, 'config', 'config.yaml'))

# Ensure logging directories exist
os.makedirs(os.path.dirname(config['logging']['file'][2]), exist_ok=True)
logging.basicConfig(level=config['logging']['level'],
                    format=config['logging']['format'],
                    filename=config['logging']['file'][2],
                    filemode='a')
logger = logging.getLogger(__name__)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(config['logging']['format'])
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Configuration
MAX_RETRIES = config['retry']['max_attempts']
RETRY_DELAY = config['retry']['delay']
TIMEOUT = 10
BATCH_SIZE = int(config['fetch']['batch_size'])
REQUEST_DELAY = float(config['processing']['request_delay'])

## GLOBAL VARIABLES
# URL and password
base_url = config['rsg']['brief_url']
scrape_url = config['rsg']['scrape_url']
password = os.environ.get('RSG_PASSWORD')
json_file_path = config['data_paths']['files']['json'][0]


## FUNCTIONAL CODE
class Scraper:
    def __init__(self, driver: webdriver.Chrome):
        self.driver = driver
        self.processed_urls: set = set()
        self.total_articles_found: int = 0

    @staticmethod
    def clean_html(html_content: str) -> str:
        soup = BeautifulSoup(html_content, 'html.parser')
        for em in soup.find_all('em'):
            em.unwrap()
        return soup.get_text(separator=' ', strip=True)

    def extract_paragraphs(self, html_content: str) -> List[str]:
        soup = BeautifulSoup(html_content, 'html.parser')
        return [p.get_text(separator=' ', strip=True) for p in soup.find_all(['p', 'h2', 'h3', 'h4'])]

    @staticmethod
    def clear_line():
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    @staticmethod
    def generate_unique_id():
        return str(uuid.uuid4())

    def parse_article(self, article: BeautifulSoup) -> Dict[str, Any]:
        url_id = article.find('a', href=True)['href']
        date_str = article.find('time').text if article.find('time') else 'N/A'
        if date_str != 'N/A':
            try:
                date = datetime.strptime(date_str, '%m/%d/%y').strftime('%Y-%m-%d')
            except ValueError:
                date = 'N/A'
        else:
            date = 'N/A'
        
        author_elem = article.find('span', class_='blog-author')
        author = author_elem.get_text(strip=True) if author_elem else 'N/A'
        
        title_elem = article.find('h1', class_='blog-title')
        title = title_elem.get_text(strip=True) if title_elem else 'N/A'
        
        excerpt_elem = article.find('div', class_='blog-excerpt-wrapper')
        excerpt = excerpt_elem.get_text(strip=True) if excerpt_elem else 'N/A'
        
        category_elem = article.find('a', class_='blog-categories')
        category = category_elem.get_text(strip=True) if category_elem else 'N/A'
        
        uuid = self.generate_unique_id()

        return {
            'urlId': url_id,
            'uuid' : uuid,
            'date': date,
            'author': author,
            'title': title,
            'excerpt': excerpt,
            'body': [],  # This will be filled by ContentFetcher
            'images': [],  # This will be filled by ContentFetcher
            'tags': [],  # This will be filled by ContentFetcher
            'references': [], # This will be filled by ContentFetcher
            'categories': category
        }

    def parse_html_data(self, html_content: str) -> List[Dict[str, Any]]:
        soup = BeautifulSoup(html_content, 'html.parser')
        articles = soup.find_all('article', class_='blog-basic-grid--container')
        return [self.parse_article(article) for article in articles]

    async def get_next_page_url(self) -> Optional[str]:
        next_button_xpath = "//div[@class='older']/a[@rel='next']"
        try:
            next_button = await asyncio.to_thread(
                WebDriverWait(self.driver, TIMEOUT).until,
                EC.element_to_be_clickable((By.XPATH, next_button_xpath))
            )
            return next_button.get_attribute('href')
        except TimeoutException:
            logger.info("No more pages to fetch. Reached the last page.")
            return None

    async def fetch_page_data(self, next_page_url: str, existing_titles: set, all_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        await asyncio.to_thread(self.driver.get, next_page_url)

        # Wait for the main content to load instead of the "next" button
        await asyncio.to_thread(
            WebDriverWait(self.driver, TIMEOUT).until,
            EC.presence_of_element_located((By.CLASS_NAME, 'blog-basic-grid--container'))
        )

        html_content = await asyncio.to_thread(lambda: self.driver.page_source)
        new_data = self.parse_html_data(html_content)

        num_new_articles = len(new_data)
        self.total_articles_found += num_new_articles

        self.clear_line()
        print(f"Total articles found: {self.total_articles_found}", end=" ", flush=True)

        if not new_data:
            logger.info("No new articles found on the current page.")
            return []

        existing_urls = {post['urlId'] for post in all_data}
        new_data = [post for post in new_data if post['urlId'] not in existing_urls]

        for post in new_data:
            if post['title'] in existing_titles:
                logger.info(f"Encountered existing post '{post['title']}'. Stopping fetch.")
                return []

        return new_data
    
    async def fetch_all_paginated_data(self, initial_html: str, existing_titles: set) -> List[Dict[str, Any]]:
        all_data = self.parse_html_data(initial_html)
        page_count = 1

        while True:
            self.clear_line()
            print(f"Fetching page {page_count}...", end="", flush=True)

            next_page_url = await self.get_next_page_url()
            if not next_page_url:
                break

            if next_page_url in self.processed_urls:
                logger.info("Encountered a previously processed page URL. Stopping pagination.")
                break

            self.processed_urls.add(next_page_url)
            new_data = await self.fetch_page_data(next_page_url, existing_titles, all_data)
            if not new_data:
                break

            all_data.extend(new_data)
            page_count += 1

        logger.info(f"Total pages fetched: {page_count}")
        logger.info(f"Total articles found: {self.total_articles_found}")
        return all_data

    
    def get_total_articles_found(self) -> int:
        return self.total_articles_found

class ContentFetcher:
    def __init__(self):
        self.session = None
        self.cookies = None

    async def create_session(self, cookies: dict = None):
        if cookies:
            self.cookies = cookies
        
        if self.session is None or self.session.closed:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            if self.cookies:
                cookie = SimpleCookie()
                for name, value in self.cookies.items():
                    cookie[name] = value
                headers['Cookie'] = cookie.output(header='', sep='; ')
            
            connector = TCPConnector(limit=100, force_close=True, enable_cleanup_closed=True)
            self.session = ClientSession(connector=connector, headers=headers)

    async def close_session(self):
        if self.session and not self.session.closed:
            await self.session.close()

    async def fetch_content(self, url_id: str) -> tuple:
        full_url = f"{scrape_url}{url_id}"
        for retry in range(MAX_RETRIES):
            try:
                async with self.session.get(full_url) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        soup = BeautifulSoup(html_content, 'html.parser')

                        body_content = soup.find('div', class_='blog-item-content-wrapper')
                        if body_content:
                            paragraphs = self.extract_paragraphs(body_content)
                            paragraphs, extracted_references = self.extract_and_clean_urls_from_text(paragraphs)
                            cleaned_paragraphs = [Scraper.clean_html(p) for p in paragraphs]  # Clean HTML here
                            images = self.extract_images(body_content)
                            references = self.extract_references(body_content)
                            references.extend(extracted_references)
                        else:
                            logger.warning("Could not find blog-item-content-wrapper")
                            logger.info(f"HTML content: {html_content[:500]}...")  # Log first 500 characters of HTML
                            cleaned_paragraphs = []
                            images = []
                            references = []

                        tags = self.extract_tags(soup)

                        return cleaned_paragraphs, images, tags, references
                    else:
                        logger.error(f"Error fetching content for {full_url}: HTTP {response.status}")
                        logger.debug(f"Response content: {await response.text()}")  # Log response content for debugging
            except aiohttp.ClientError as e:
                logger.error(f"Error fetching content for {full_url}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error fetching content for {full_url}: {e}")
            
            if retry < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)
            else:
                logger.error(f"Max retries reached for {full_url}")
        logger.warning(f"Failed to fetch content for {full_url} after {MAX_RETRIES} attempts")
        return [], [], [], []

    @staticmethod
    def extract_paragraphs(content: BeautifulSoup) -> List[str]:
        paragraphs = []
        for elem in content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'ol', 'ul']):
            text = elem.get_text(strip=True)
            if text:
                paragraphs.append(text)
        return paragraphs

    @staticmethod
    def extract_images(content: BeautifulSoup) -> List[str]:
        images = content.find_all('img')
        image_urls = [img.get('src') for img in images if img.get('src')]
        return image_urls

    @staticmethod
    def extract_tags(soup: BeautifulSoup) -> List[str]:
        tag_wrapper = soup.find('div', class_='blog-meta-item blog-meta-item--tags')
        if tag_wrapper:
            tags = [tag.get_text(strip=True) for tag in tag_wrapper.find_all('a', class_='blog-item-tag')]
            return tags
        return []

    @staticmethod
    def extract_references(content: BeautifulSoup) -> List[str]:
        links = content.find_all('a', href=True)
        references = [link['href'] for link in links if not link['href'].startswith('/') and not link['href'].startswith(scrape_url)]
        return references
    
    @staticmethod
    def extract_and_clean_urls_from_text(paragraphs: List[str]) -> Tuple[List[str], List[str]]:
        url_pattern = re.compile(r'(http[s]?://\S+|www\.\S+|\S+\.(com|org|net|edu|gov|mil|biz|info|io|co|us|uk|ca|de|jp|fr|au|ru|ch|it|nl|se|no|es|br|fi|be|at|dk|pl|cz|me|tv|io))')
        all_urls = []
        cleaned_paragraphs = []
        
        for paragraph in paragraphs:
            urls = url_pattern.findall(paragraph)
            cleaned_text = re.sub(url_pattern, '', paragraph)
            all_urls.extend([url[0] for url in urls])
            cleaned_paragraphs.append(cleaned_text)

        return cleaned_paragraphs, all_urls

    async def fetch_full_content(self, url_ids: List[str]) -> List[tuple]:
        if not self.session:
            await self.create_session()
        
        total_articles = len(url_ids)
        print(f"Scraping {total_articles} articles...")
        results = []

        for url_id in tqdm(url_ids, desc='Fetching articles'):
            try:
                result = await self.fetch_content(url_id)
                results.append(result)
                await asyncio.sleep(REQUEST_DELAY)  # Add delay between requests
            except Exception as e:
                logger.error(f"Error processing article {url_id}: {e}")
                results.append(([], [], [], []))

        return results

class DataStorage:
    def __init__(self):
        self.json_file_path = json_file_path

    def load_existing_data(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.json_file_path):
            with open(self.json_file_path, 'r+', encoding='utf-8') as json_file:
                return json.load(json_file)
        return []

    def update_json_file(self, new_data: List[Dict[str, Any]]):
        existing_data = self.load_existing_data()
        existing_urls = {post['urlId'] for post in existing_data}
        
        new_data = [post for post in new_data if post['urlId'] not in existing_urls]
        combined_data = new_data + existing_data

        with open(self.json_file_path, 'w+', encoding='utf-8') as json_file:
            json.dump(combined_data, json_file, indent=4, ensure_ascii=False)

class Main:
    def __init__(self):
        service = Service(ChromeDriverManager().install())
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        self.driver = webdriver.Chrome(service=service, options=options)
        self.scraper = Scraper(self.driver)
        self.content_fetcher = ContentFetcher()
        self.data_storage = DataStorage()

    async def run(self):
        try:
            logger.info("Starting scraping process...")
            print("Navigating to the URL...")
            await asyncio.to_thread(self.driver.get, base_url)
            await asyncio.sleep(1)
            print("Entering password...", end=" ", flush=True)
            password_field = await asyncio.to_thread(self.driver.find_element, By.NAME, 'password')
            await asyncio.to_thread(password_field.send_keys, password)
            await asyncio.to_thread(password_field.send_keys, Keys.RETURN)

            await asyncio.to_thread(
                WebDriverWait(self.driver, TIMEOUT).until,
                EC.presence_of_element_located((By.CLASS_NAME, 'blog-basic-grid--container'))
            )
            selenium_cookies = self.driver.get_cookies()
            cookie_dict = {cookie['name']: cookie['value'] for cookie in selenium_cookies}
            # Pass the cookies to ContentFetcher
            await self.content_fetcher.create_session(cookie_dict)
            
            print("Fetching initial page content...", end="", flush=True)
            initial_html = await asyncio.to_thread(lambda: self.driver.page_source)

            existing_data = self.data_storage.load_existing_data()
            existing_titles = {post['title'] for post in existing_data}

            print("Fetching all paginated data...", end="", flush=True)
            new_data = await self.scraper.fetch_all_paginated_data(initial_html, existing_titles)
            if not new_data:
                print("No new data found. Exiting.")
                return

            urls = [post['urlId'] for post in new_data]
            
            await self.content_fetcher.create_session(cookie_dict)
            full_content = await self.content_fetcher.fetch_full_content(urls)

            for post, content in zip(new_data, full_content):
                paragraphs, images, tags, references = content
                post['body'] = paragraphs
                post['images'] = images
                post['tags'] = tags
                post['references'] = references

            print(f"Updating JSON file with {len(new_data)} new posts.")
            self.data_storage.update_json_file(new_data)

            print("Scraping completed.")
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            logger.error("Traceback:", exc_info=True)
        finally:
            await asyncio.to_thread(self.driver.quit)
            await self.content_fetcher.close_session()

async def main():
    scraper = Main()
    try:
        await scraper.run()
    except Exception as e:
        logger.error(f"Unhandled exception in main: {str(e)}")
        logger.error("Traceback:", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
