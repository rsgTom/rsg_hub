import json
import pandas as pd

def correct_urls_and_unicode_in_json(filepath):
    try:
        # Load the JSON file
        with open(filepath, 'r') as file:
            data = json.load(file)
        
        # Correct the URLs and Unicode escapes in the data
        for record in data:
            for key, value in record.items():
                if isinstance(value, str):
                    # Correct URLs
                    if "://" in value:
                        value = value.replace("\\/", "/")
                    # Correct Unicode escapes
                    value = value.encode('utf-8').decode('unicode_escape')
                    record[key] = value
                elif isinstance(value, list):
                    new_list = []
                    for v in value:
                        if isinstance(v, str):
                            # Correct URLs
                            if "://" in v:
                                v = v.replace("\\/", "/")
                            # Correct Unicode escapes
                            v = v.encode('utf-8').decode('unicode_escape')
                        new_list.append(v)
                    record[key] = new_list
                    
                # Correct escape characters in the 'body' field
        for record in data:
            if "Content" in record and "body" in record["Content"]:
                corrected_body = [paragraph.replace('\"', '') for paragraph in record["Content"]["body"]]
                record["Content"]["body"] = corrected_body
        
        # Convert the corrected data to a DataFrame
        df = pd.DataFrame(data)
        
        # Save the corrected data back to the JSON file
        with open(filepath, 'w') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        
        print(f"URLs and Unicode escapes in {filepath} have been corrected and saved.")
        return df

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return pd.DataFrame()
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return pd.DataFrame()

def main():
    """
    A function that calls the correct_urls_and_unicode_in_json function.
    """
    print("Normalizing Jason's Records")
    filepath = "backend/data/raw/blog_posts_extracted.json"
    df = correct_urls_and_unicode_in_json(filepath)
    print(df.head())
    print ("Jason's records are normalized.")

if __name__ == "__main__":
    main()

