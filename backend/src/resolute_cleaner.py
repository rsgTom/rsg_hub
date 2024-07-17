import os
import sys
import logging
import re
import string
import json
import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from typing import List, Dict, Any, Tuple, Set, Optional
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keybert import KeyBERT
from transformers import logging as hf_logging, AutoModelForTokenClassification, AutoTokenizer, pipeline
import html
import warnings
import nltk
from tqdm import tqdm
import spacy
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np

# CONFIG, LOGGING, ENVIRONMENT SETUP
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.config_loader import load_config

config = load_config(os.path.join(project_root, 'config', 'config.yaml'))
batch_size = config['processing']['batch_size']
keywords_to_extract = config['processing']['keywords_to_extract']
summary_keywords = config['processing']['summary_keywords']

os.makedirs(os.path.dirname(config['logging']['file'][3]), exist_ok=True)
logging.basicConfig(level=config['logging']['level'],
                    format=config['logging']['format'],
                    filename=config['logging']['file'][3],
                    filemode='a')
logger = logging.getLogger(__name__)

raw_json = config['data_paths']['files']['json'][0]
clean_json = config['data_paths']['files']['json'][1]

# Suppress specific warnings
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='spacy')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
hf_logging.set_verbosity_error()

# Download necessary NLTK data quietly
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize spaCy for NER
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    spacy.cli.download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

# Define custom stop words
custom_stopwords = stopwords.words('english') + list(string.punctuation) + ['``', "''", '"']
                                                                            
# Initialize KeyBERT with a smaller model
kb_model = KeyBERT(model='distilbert-base-nli-mean-tokens')

# Initialize transformer model for NER
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model_ner = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
ner_pipeline = pipeline("ner", model=model_ner, tokenizer=tokenizer, aggregation_strategy="simple")


class Polisher:
    """Class for cleaning and polishing data."""

    @staticmethod
    def clean_url(url: str) -> str:
        """Clean and unescape a URL."""
        return html.unescape(url).replace('\\', '')

    @staticmethod
    def handle_encoding(text: Any) -> Any:
        """Handle encoding issues in text data."""
        if isinstance(text, str):
            return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        elif isinstance(text, list):
            return [Polisher.handle_encoding(item) for item in text]
        elif isinstance(text, dict):
            return {key: Polisher.handle_encoding(value) for key, value in text.items()}
        return text

    @staticmethod
    def clean_urls_in_body_column(df: pd.DataFrame) -> None:
        """Clean URLs in the 'body' column of a DataFrame."""
        if 'body' in df.columns:
            df['body'] = df['body'].apply(lambda x: Polisher.clean_url(x) if isinstance(x, str) else [Polisher.clean_url(item) if isinstance(item, str) else item for item in x])


class TextProcessor:
    """Class for processing and analyzing text data."""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.model = kb_model
        self.ner_pipeline = ner_pipeline

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = str(text).replace('\"', '"').replace('\n', ' ').replace('\r', ' ').strip()
        return Polisher.handle_encoding(text)

    def normalize_text(self, text: Any) -> Any:
        """Normalize text or list of texts."""
        if isinstance(text, list):
            return [self._normalize_paragraph(paragraph) for paragraph in text]
        return self._normalize_paragraph(text)

    def _normalize_paragraph(self, text: str) -> str:
        """Normalize a single paragraph of text."""
        text = BeautifulSoup(text, "html.parser").get_text()
        text, preserved = self._preserve_special_patterns(text)
        text = self._normalize_case_and_punctuation(text)
        text = self._restore_preserved_patterns(text, preserved)
        return " ".join(text.split())

    def _preserve_special_patterns(self, text: str) -> Tuple[str, List[str]]:
        """Preserve special patterns in text during normalization."""
        patterns_to_preserve = [
            r'\b[a-zA-Z]{2,}\d+\b',
            r'\b\d+[.,]?\d*\b',
            r'\"[^"]*\"',
        ]
        preserved = []
        for pattern in patterns_to_preserve:
            preserved.extend(re.findall(pattern, text))
        return text, preserved

    def _normalize_case_and_punctuation(self, text: str) -> str:
        """Normalize case and remove punctuation from text."""
        return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())

    def _restore_preserved_patterns(self, text: str, preserved: List[str]) -> str:
        """Restore preserved patterns to normalized text."""
        for item in preserved:
            text = text.replace(' '.join(item.split()), item)
        return text

    def preprocess_text(self, text: Any) -> Any:
        """Preprocess text or list of texts."""
        if isinstance(text, list):
            return [self._preprocess_paragraph(paragraph) for paragraph in text]
        return self._preprocess_paragraph(text)

    def _preprocess_paragraph(self, text: str) -> str:
        """Preprocess a single paragraph of text."""
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        return " ".join(words)

    def extract_keywords(self, text: Any) -> List[str]:
        """Extract keywords from text."""
        text = ' '.join(text) if isinstance(text, list) else text
        if not text.strip():
            return []
        keywords = self.model.extract_keywords(text, stop_words=custom_stopwords, top_n=keywords_to_extract)
        return [kw[0] for kw in keywords]

    def summarize_text(self, text: Any) -> str:
        """Generate a summary of the text."""
        text = ' '.join(text) if isinstance(text, list) else text
        if not text.strip():
            return ''
        keywords = self.model.extract_keywords(text, stop_words=custom_stopwords, top_n=summary_keywords)
        summary = ' '.join([kw[0] for kw in keywords])
        return summary

    def enhanced_ner(self, text: Any) -> List[Tuple[str, str]]:
        """Perform Named Entity Recognition on the text."""
        text = ' '.join(text) if isinstance(text, list) else text
        if not text.strip():
            return []
        entities = self.ner_pipeline(text)
        return [(ent['word'], ent['entity_group']) for ent in entities]


class JsonUpdater:
    """Class for updating JSON data."""

    def __init__(self, raw_json: str, clean_json: str):
        self.raw_json = raw_json
        self.clean_json = clean_json

    def load_existing_data(self) -> List[Dict[str, Any]]:
        """Load existing data from the clean JSON file."""
        if os.path.exists(self.clean_json):
            with open(self.clean_json, 'r+', encoding='utf-8', errors='ignore') as json_file:
                return json.load(json_file)
        return []

    def update_json_file(self, new_data: List[Dict[str, Any]]) -> None:
        """Update the clean JSON file with new data."""
        existing_data = self.load_existing_data()
        existing_ids = {post['unique_id'] for post in existing_data}
        
        new_data = [post for post in new_data if post['uuid'] not in existing_ids]
        combined_data = new_data + existing_data

        with open(self.clean_json, 'w+', encoding='utf-8') as json_file:
            json.dump(combined_data, json_file, indent=4, ensure_ascii=False)


def process_batch(df_batch: pd.DataFrame, text_processor: TextProcessor) -> pd.DataFrame:
    """Process a batch of data."""
    df_batch['summary'] = df_batch['body'].apply(lambda x: text_processor.summarize_text(x))
    df_batch['keywords'] = df_batch.apply(lambda row: (row['tags'] if isinstance(row['tags'], list) else []) + text_processor.extract_keywords(row['body']), axis=1)
    df_batch['keywords'] = df_batch['keywords'].apply(lambda keywords: [kw for kw in keywords if kw.strip() and kw not in string.punctuation])
    df_batch['entities'] = df_batch['body'].apply(lambda x: text_processor.enhanced_ner(x))
    return df_batch


def process_dataframe(df: pd.DataFrame, text_processor: TextProcessor) -> pd.DataFrame:
    """Process the entire DataFrame using multiprocessing."""
    columns_to_process = ['title', 'author', 'excerpt', 'body', 'tags', 'categories']

    logger.info("Starting data processing...")
    print("Applying the magic eraser... 🧹🧙‍♂️")
    
    # Split the dataframe into chunks for multiprocessing
    num_processes = min(cpu_count(), 4)  # Use at most 4 processes
    df_splits = np.array_split(df, num_processes)
    
    # Create a partial function with fixed arguments
    process_func = partial(process_chunk, text_processor=text_processor, columns_to_process=columns_to_process)
    
    # Process the chunks in parallel
    with Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(process_func, df_splits), total=len(df_splits), desc="Processing chunks"))
    
    # Combine the results
    df_processed = pd.concat(results)
    
    logger.info("Data processing completed.")
    return df_processed

def process_chunk(df_chunk: pd.DataFrame, text_processor: TextProcessor, columns_to_process: List[str]) -> pd.DataFrame:
    """Process a chunk of the DataFrame."""
    df_chunk = process_batch_columns(df_chunk, text_processor, columns_to_process)
    df_chunk = process_batch(df_chunk, text_processor)
    return df_chunk

def process_batch_columns(df_batch: pd.DataFrame, text_processor: TextProcessor, columns_to_process: List[str]) -> pd.DataFrame:
    """Process specific columns of a batch."""
    for column in columns_to_process:
        if column in df_batch.columns:
            if column == 'body':
                df_batch[column] = df_batch[column].apply(lambda x: text_processor.clean_text(x))
                df_batch[column] = df_batch[column].apply(lambda x: text_processor.normalize_text(x))
                df_batch[column] = df_batch[column].apply(lambda x: text_processor.preprocess_text(x))
            else:
                df_batch[column] = df_batch[column].apply(lambda x: text_processor.clean_text(x))
                if column != 'author':
                    df_batch[column] = df_batch[column].apply(lambda x: text_processor.normalize_text(x))
                    df_batch[column] = df_batch[column].apply(lambda x: text_processor.preprocess_text(x))
    return df_batch


def load_and_preprocess_data(raw_json: str, existing_ids: Set[str], polisher: Polisher) -> Optional[pd.DataFrame]:
    """Load and preprocess data from the raw JSON file."""
    logger.info("Loading and preprocessing data...")
    try:
        # Load raw JSON data
        with open(raw_json, 'r', encoding='utf-8', errors='ignore') as f:
            raw_data = json.load(f)

        # Apply encoding handler to the entire dataset
        raw_data = polisher.handle_encoding(raw_data)

        # Identify new records
        new_data = [post for post in raw_data if post.get('uuid', post.get('unique_id')) not in existing_ids]
        print(f"Found {len(new_data)} new records to polish. Time to make them shine! ✨")

        if not new_data:
            logger.info("No new records to polish.")
            print("No new records to polish. Jason can rest easy for now. 😴")
            return None

        # Normalize new data to a DataFrame
        df = pd.json_normalize(new_data)

        # Handle missing data
        df.fillna('', inplace=True)

        # Clean URLs in body
        Polisher.clean_urls_in_body_column(df)
        
        # Standardize dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')

        # Remove duplicates within new data
        df.drop_duplicates(subset=['urlId', 'title'], inplace=True)

        logger.info("Data loading and preprocessing completed.")
        return df
    except Exception as e:
        logger.error(f"Error in load_and_preprocess_data: {str(e)}")
        raise


def log_problematic_data(df: pd.DataFrame) -> None:
    """Log problematic data that can't be encoded to JSON."""
    for index, row in df.iterrows():
        try:
            json.dumps(row.to_dict(), ensure_ascii=False)
        except UnicodeEncodeError as e:
            logger.error(f"Encoding error in row {index}: {e}")
            for key, value in row.items():
                try:
                    json.dumps({key: value}, ensure_ascii=False)
                except UnicodeEncodeError:
                    logger.error(f"Problematic field: {key}")


def main():
    """Main function to run the data cleaning and preparation process."""
    json_updater = JsonUpdater(raw_json, clean_json)
    polisher = Polisher()
    text_processor = TextProcessor()

    logger.info("Starting data cleaning and preparation process.")
    print("Browsing new records... 🧙‍♂️")

    # Load existing cleaned data
    existing_data = json_updater.load_existing_data()
    existing_ids = {post.get('uuid', post.get('unique_id')) for post in existing_data}

    # Load and preprocess data
    df = load_and_preprocess_data(raw_json, existing_ids, polisher)
    if df is None:
        return

    # Process the dataframe
    df = process_dataframe(df, text_processor)

    # Convert cleaned DataFrame back to list of dicts
    cleaned_new_data = df.to_dict(orient='records')

    # Log problematic data
    log_problematic_data(df)

    # Update JSON file with cleaned new records
    json_updater.update_json_file(cleaned_new_data)
    print("Uploading Jason's polished records to the Hall of Fame... 🏆📜")

    print(f"\nPolished {len(cleaned_new_data)} records for Jason. 🧹", end=' ', flush=True)

    # Display a sample of the cleaned data
    print(df.head())

    logger.info("Data cleaning and preparation complete. Jason is now sparkling clean! 🧼✨")
    print("Jason's records are now the epitome of cleanliness! 🧼✨")

if __name__ == "__main__":
    main()