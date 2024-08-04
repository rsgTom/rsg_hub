import os
import sys
import json
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from dotenv import load_dotenv
import tqdm
import logging
import yaml


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.config_loader import load_config

load_dotenv()
config = load_config(os.path.join(project_root, 'config', 'config.yaml'))

# OPENAI INITIALIZATION
api_key = os.getenv('OPENAI_API_KEY')
oai_model = os.getenv("SMART_LLM")
temperature = os.getenv("TEMPERATURE")

if not api_key or not oai_model or not temperature:
    raise ValueError("Missing necessary environment variables. Please check your .env file.")

# Logging setup
os.makedirs(os.path.dirname(config['logging']['file'][1]), exist_ok=True)
logging.basicConfig(level=logging.ERROR,
                    format=config['logging']['format'],
                    filename=config['logging']['file'][1],
                    filemode='a')
logger = logging.getLogger(__name__)


class OaiParser:
    def __init__(self):
        self.api_key = api_key
        self.llm_model = oai_model
        self.temperature = temperature
        self.chat = ChatOpenAI(api_key=self.api_key, model=self.llm_model, temperature=float(self.temperature))
        
        
    def setup_parser(self, response_schemas):
        return StructuredOutputParser.from_response_schemas(response_schemas)
    
    
    def setup_prompt(self, template):
        return ChatPromptTemplate.from_template(template=template)
    
    
    def extract_information(self, input_text, parser, prompt):
        format_instructions = parser.get_format_instructions()
        messages = prompt.format_messages(text=input_text, format_instructions=format_instructions)
        response = self.chat.invoke(messages)
        try:
            return parser.parse(response.content)
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Response content: {response.content}")
            return {schema.name: [] for schema in parser.response_schemas}



oai_parser = OaiParser()


response_schema = [
    ResponseSchema(name="summary", description="extract summary from text"),
    ResponseSchema(name="tldr", description="extract tldr from text"),
    ResponseSchema(name="keywords", description="extract keywords from text"),
    ResponseSchema(name="synonyms_acronyms", description="extract synonyms and acronyms from text"),
    ResponseSchema(name="entities", description="extract entities from text"),
    ResponseSchema(name="locations", description="extract locations from text"),
    ResponseSchema(name="diseases", description="extract diseases from text"),
    ResponseSchema(name="variants", description="extract variants from text"),
    ResponseSchema(name="genetic_information", description="extract genetic information from text"),
    ResponseSchema(name="symptoms", description="extract symptoms from text"),
    ResponseSchema(name="medications", description="extract medications from text"),
    ResponseSchema(name="treatments", description="extract treatments from text"),
    ResponseSchema(name="medical_tests", description="extract medical tests from text"),
    ResponseSchema(name="diagnoses", description="extract diagnoses from text"),
    ResponseSchema(name="causes", description="extract causes from text"),
    ResponseSchema(name="key_findings", description="extract key findings from text"),
    ResponseSchema(name="causal_relationships", description="extract causal relationships from text"),
    ResponseSchema(name="scientific_facts", description="extract scientific facts from text"),
    ResponseSchema(name="practical_implications", description="extract practical implications from text"),
    ResponseSchema(name="policy_recommendations", description="extract policy recommendations from text"),
]


prompt_template = """
Ignore all previous instructions. Now for the following text, extract the following information:

summary: extract summary from the text without using jargon or technical terms
output them as a Python list, if not available return []

tldr: create a brief one-sentence tldr from the text without using jargon or technical terms
output them as a Python list, if not available return []

keywords: extract keywords from the text
output them as a Python list, if not available return []

synonyms_acronyms: extract synonyms and acronyms for key terms from the text
output them as a Python list, if not available return []

entities: extract all entities from the text
output them as a Python list, if not available return []

locations: extract locations from the text
output them as a Python list, if not available return []

diseases: extract diseases from the text
output them as a Python list, if not available return []

variants: extract variants from the text
output them as a Python list, if not available return []

genetic_information: extract genetic information from the text
output them as a Python list, if not available return []

symptoms: extract symptoms from the text
output them as a Python list, if not available return []

medications: extract medications from the text
output them as a Python list, if not available return []

treatments: extract treatments from the text
output them as a Python list, if not available return []

medical_tests: extract medical tests from the text
output them as a Python list, if not available return []

diagnoses: extract diagnoses from the text
output them as a Python list, if not available return []

causes: extract causes from the text
output them as a Python list, if not available return []

key_findings: extract key findings from the text
output them as a Python list, if not available return []

causal_relationships: extract causal relationships and dependencies between concepts from the text
output them as a Python list, if not available return []

scientific_facts: extract implicit or explicit scientific facts derived from the text
output them as a Python list, if not available return []

practical_implications: extract implicit and explicit practical implications from the text
output them as a Python list, if not available return []

policy_recommendations: extract implicit and explicit policy recommendations from the text
output them as a Python list, if not available return []

Format the output as a JSON with the following keys:
summary
tldr
keywords
synonyms_acronyms
entities
locations
diseases
variants
genetic_information
symptoms
medications
treatments
medical_tests
diagnoses
causes
key_findings
causal_relationships
scientific_facts
practical_implications
policy_recommendations

text: {text}
{format_instructions}
"""


parser = oai_parser.setup_parser(response_schema)
prompt = oai_parser.setup_prompt(prompt_template)



# Chunking Logic
def chunk_text(text, max_length=128000):
    if isinstance(text, list):
        text = '. '.join(text)
        sentences = text.split('. ')
    chunks, current_chunk = [], []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_length:
            chunks.append('. '.join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append('. '.join(current_chunk))

    return chunks


def map_to_keys(data):
    entities = {
        "Metadata": {
            "uuid": data.get("uuid", ""),
            "urlId": data.get("urlId", "").replace("\\/", "/"),
            "date": data.get("date", ""),
            "author": data.get("author", ""),
            "title": data.get("title", ""),
            "excerpt": data.get("excerpt", ""),
            "summary": data.get("summary", []),
            "tldr": data.get("tldr", []),
            "keywords": data.get("keywords", []),
            "synonyms_acronyms": data.get("synonyms_acronyms", []),
            "entities": data.get("entities", []),
            "locations": data.get("locations", []),
        },
        "Content": {
            "body": data.get("body", ""),
            "images": [img.replace("\\/", "/") for img in data.get("images", [])],
            "tags": data.get("tags", []),
            "references": [ref.replace("\\/", "/") for ref in data.get("references", [])],
            "categories": data.get("categories", []),
        },
        "Medical": {
            "diseases": data.get("diseases", []),
            "variants": data.get("variants", []),
            "genetic_information": data.get("genetic_information", []),
            "symptoms": data.get("symptoms", []),
            "medications": data.get("medications", []),
            "treatments": data.get("treatments", []),
            "medical_tests": data.get("medical_tests", []),
            "diagnoses": data.get("diagnoses", []),
            "causes": data.get("causes", []),
        },
        "Research": {
            "key_findings": data.get("key_findings", []),
            "causal_relationships": data.get("causal_relationships", []),
            "scientific_facts": data.get("scientific_facts", []),
            "practical_implications": data.get("practical_implications", []),
            "policy_recommendations": data.get("policy_recommendations", []),
        }
    }
    return entities
    
    
def extract_entities_and_facts(record):
    uuid = record.get("uuid")
    urlId = record.get("urlId")
    date = record.get("date")
    author = record.get("author")
    title = record.get("title")
    excerpt = record.get("excerpt")
    body = record.get("body")
    images = record.get("images")
    tags = record.get("tags")
    references = record.get("references")
    categories = record.get("categories")
    
    entities = {
        "uuid": uuid,
        "urlId": urlId.replace("\\/", "/"),
        "date": date,
        "author": author,
        "title": title,
        "excerpt": excerpt,
        "body": body,
        "images": [img.replace("\\/", "/") for img in images],
        "tags": tags,
        "references": [ref.replace("\\/", "/") for ref in references],
        "categories": categories,
        "summary": [],
        "tldr": [],
        "keywords": [],
        "synonyms_acronyms": [],
        "entities": [],
        "locations": [],
        "diseases": [],
        "variants": [],
        "genetic_information": [],
        "symptoms": [],
        "medications": [],
        "treatments": [],
        "medical_tests": [],
        "diagnoses": [],
        "causes": [],
        "key_findings": [],
        "causal_relationships": [],
        "scientific_facts": [],
        "practical_implications": [],
        "policy_recommendations": [],
    }
    if body:
        chunks = chunk_text(body)
        for chunk in chunks:
            response = oai_parser.extract_information(chunk, parser, prompt)
            for key, value in response.items():
                entities[key].extend(value)
    return map_to_keys(entities)


def process_json_file(filepath):
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return pd.DataFrame()
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding error: {e}")
        return pd.DataFrame()
        
    records = data
    extracted_data = []

    # Determine the last processed date if the output file already exists
    output_filepath = filepath.replace("./backend/data/raw/blog_posts.json", "./backend/data/raw/blog_posts_extracted.json")
    previous_end = None
    if os.path.exists(output_filepath):
        try:
            with open(output_filepath, 'r') as file:
                existing_data = json.load(file)
                if existing_data:
                    previous_end = max(record.get("Metadata", {}).get("date") for record in existing_data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error reading existing output file: {e}")
    
    for record in tqdm.tqdm(records, desc="Extracting entities and facts"):
        record_date = record.get("date")
        if not previous_end or record_date > previous_end:
            extracted_data.append(extract_entities_and_facts(record))
    
    # Load existing data and append new data if applicable
    if os.path.exists(output_filepath) and previous_end:
        with open(output_filepath, 'r') as file:
            existing_data = json.load(file)
        extracted_data = existing_data + extracted_data

    return pd.DataFrame(extracted_data)


def save_to_json(df, output_filepath):
    # Convert DataFrame to JSON
    json_str = df.to_json(orient='records', indent=4)
    
    # Print a preview of the JSON data
    print("Preview of the JSON data to be saved:")
    print(json_str[:1000])  # Print the first 1000 characters for inspection

    try:
        # Attempt to parse the JSON string to ensure it's valid
        json_data = json.loads(json_str)
        
        # Remove any extra backslashes in URLs (if necessary)
        for record in json_data:
            for key, value in record.items():
                if isinstance(value, str) and "://" in value:
                    record[key] = value.replace("\\/", "/")
        
        # Save to file if valid
        with open(output_filepath, 'w') as file:
            json.dump(json_data, file, indent=4)
        print(f"Data successfully saved to {output_filepath}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding error: {e}")
        print("The JSON string is invalid. Here is a preview:")
        print(json_str[:1000])  # Print the first 1000 characters for inspection



def main():
    filepath = "./backend/data/raw/blog_posts.json"
    output_filepath = "./backend/data/raw/blog_posts_extracted.json"
    df = process_json_file(filepath)
    if not df.empty:
        save_to_json(df, output_filepath)
        print(df.head())
    else:
        print("No data to process.")



if __name__ == "__main__":
    main()
