import os
import json
import random
import requests
import time

# --- Configuration ---

# Define how many IDs to select for each category, scaled up to a total of 1000.
# The original ratio was 60 (train), 20 (test), 20 (val).
SAMPLE_SIZES = {
    'train': 600,
    'test': 200,
    'val': 200
}

# List of the input JSON files for image IDs.
# The script expects these to be in the same directory where it is run.
INPUT_FILES = [
    'split.dress.test.json',
    'split.dress.train.json',
    'split.dress.val.json',
    'split.shirt.test.json',
    'split.shirt.train.json',
    'split.shirt.val.json',
    'split.toptee.test.json',
    'split.toptee.train.json',
    'split.toptee.val.json'
]

# The name of the main folder where all results will be saved.
OUTPUT_DIR = 'Fashion-IQ'

def load_url_mappings():
    """
    Loads all asin2url.txt files into a single dictionary for easy lookup.
    Returns a dictionary mapping ASIN (image ID) to its image URL.
    """
    url_map = {}
    mapping_files = ['asin2url.dress.txt', 'asin2url.shirt.txt', 'asin2url.toptee.txt']
    print("Loading all image URL mappings...")
    for filename in mapping_files:
        try:
            with open(filename, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        asin, url = parts
                        url_map[asin] = url
        except FileNotFoundError:
            print(f"[Warning] URL mapping file not found: '{filename}'.")
    print(f"Loaded {len(url_map)} URL mappings in total.")
    return url_map

def download_image(url, filepath):
    """
    Downloads an image from a given URL and saves it to the specified path.
    Includes error handling and retries.
    """
    try:
        # Use a timeout to prevent the script from hanging on a bad request.
        response = requests.get(url, stream=True, timeout=10)
        # Check if the request was successful (status code 200).
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
        else:
            # Don't print a failure message for common 404s to reduce noise
            if response.status_code != 404:
                print(f"      [Failed] Status code {response.status_code} for URL: {url}")
            return False
    except requests.exceptions.RequestException:
        # Don't print error for connection issues to reduce noise
        return False

def process_files():
    """
    Samples IDs, finds captions, filters IDs, downloads images, and cleans up
    JSON files based on download success.
    """
    url_mappings = load_url_mappings()
    print("\nStarting the image ID, caption, and image processing...")

    # --- 1. Define and create the new output structure ---
    base_dir = os.path.join(OUTPUT_DIR, 'fashion-iq')
    captions_dir = os.path.join(base_dir, 'captions')
    splits_dir = os.path.join(base_dir, 'image_splits')
    images_dir = os.path.join(base_dir, 'images')

    print(f"Ensuring output directories exist...")
    os.makedirs(captions_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    print(f"Output will be saved in: '{base_dir}'")

    for split_filename in INPUT_FILES:
        try:
            # --- 2. Parse filename ---
            parts = split_filename.replace('.json', '').split('.')
            if len(parts) != 3 or parts[0] != 'split':
                print(f"\n[Warning] Skipping file with unexpected name format: {split_filename}")
                continue
            
            print(f"\nProcessing '{split_filename}'...")

            # --- 3. Load and sample image IDs ---
            with open(split_filename, 'r') as f:
                image_ids = json.load(f)
            
            split_type = parts[2]
            sample_size = SAMPLE_SIZES.get(split_type)
            if not sample_size:
                continue
            
            sampled_ids = random.sample(image_ids, min(len(image_ids), sample_size))
            sampled_ids_set = set(sampled_ids)

            # --- 4. Find corresponding captions ---
            caption_filename = split_filename.replace('split.', 'cap.')
            found_captions = []
            all_captioned_ids = set()

            try:
                with open(caption_filename, 'r') as f:
                    all_captions = json.load(f)
                
                for entry in all_captions:
                    target_id = entry.get('target')
                    candidate_id = entry.get('candidate')
                    
                    if (target_id in sampled_ids_set) or (candidate_id in sampled_ids_set):
                        found_captions.append(entry)
                        if target_id: all_captioned_ids.add(target_id)
                        if candidate_id: all_captioned_ids.add(candidate_id)

                print(f"Found {len(found_captions)} caption entries for the initial sample.")

                # --- 5. Filter the original sampled list ---
                final_split_ids = [id for id in sampled_ids if id in all_captioned_ids]

                # --- 6. Save the initial files into the new structure ---
                output_split_filepath = os.path.join(splits_dir, split_filename)
                with open(output_split_filepath, 'w') as f:
                    json.dump(final_split_ids, f, indent=4)
                print(f"Saved {len(final_split_ids)} initial IDs to '{output_split_filepath}'")

                output_caption_filepath = os.path.join(captions_dir, caption_filename)
                with open(output_caption_filepath, 'w') as f:
                    json.dump(found_captions, f, indent=4)
                print(f"Saved {len(found_captions)} initial captions to '{output_caption_filepath}'")
                
                # --- 7. Download the images ---
                print(f"Starting image download for {len(all_captioned_ids)} unique images...")
                successfully_downloaded_ids = set()
                for image_id in all_captioned_ids:
                    image_filepath = os.path.join(images_dir, f"{image_id}.jpg")
                    if os.path.exists(image_filepath):
                        successfully_downloaded_ids.add(image_id)
                        continue
                    
                    image_url = url_mappings.get(image_id)
                    if image_url:
                        if download_image(image_url, image_filepath):
                            successfully_downloaded_ids.add(image_id)
                            time.sleep(0.05) # Be polite to the server
                    # else: No need to warn here, the check below handles it.

                print(f"Finished: Successfully downloaded or verified {len(successfully_downloaded_ids)} images.")

                # --- 8. Clean JSON files based on download success ---
                failed_downloads = all_captioned_ids - successfully_downloaded_ids
                if failed_downloads:
                    print(f"Detected {len(failed_downloads)} images that could not be downloaded. Cleaning JSON files...")

                    # Clean the captions file
                    cleaned_captions = [
                        entry for entry in found_captions
                        if entry.get('target') in successfully_downloaded_ids and entry.get('candidate') in successfully_downloaded_ids
                    ]
                    with open(output_caption_filepath, 'w') as f:
                        json.dump(cleaned_captions, f, indent=4)
                    print(f"Cleaned captions file: {len(cleaned_captions)} entries remaining.")

                    # Clean the split file
                    cleaned_split_ids = [
                        id for id in final_split_ids if id in successfully_downloaded_ids
                    ]
                    with open(output_split_filepath, 'w') as f:
                        json.dump(cleaned_split_ids, f, indent=4)
                    print(f"Cleaned split file: {len(cleaned_split_ids)} IDs remaining.")
                else:
                    print("All images present. No cleanup needed.")

            except FileNotFoundError:
                print(f"[Warning] Caption file '{caption_filename}' not found. Skipping.")
        
        except FileNotFoundError:
            print(f"\n[Error] The file '{split_filename}' was not found.")
        except Exception as e:
            print(f"\n[Error] An unexpected error occurred with '{split_filename}': {e}")

    print("\n-----------------------------------------")
    print("All files processed. Sampling and downloads complete.")
    print(f"Check the '{OUTPUT_DIR}' folder for the results.")
    print("-----------------------------------------")

if __name__ == '__main__':
    process_files()

