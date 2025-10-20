import json
import random

# --- Configuration ---
# The file created by the previous keyword selection script
INPUT_JSON_PATH = 'laion_template_fashion_all_matches.json'

# The final output file containing the 2000 random samples for training
OUTPUT_JSON_PATH = 'fashion_train_subset_2000.json'

# The number of triplets you want to randomly select
NUM_SAMPLES_TO_SELECT = 2000

# --- Main Logic ---
print(f"Loading data from {INPUT_JSON_PATH}...")
try:
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        all_fashion_triplets = json.load(f)
except FileNotFoundError:
    print(f"Error: Input file not found at {INPUT_JSON_PATH}")
    print("Please make sure you have run the keyword selection script first.")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {INPUT_JSON_PATH}. Check the file format.")
    exit()

num_available = len(all_fashion_triplets)
print(f"Loaded {num_available} fashion-related triplets.")

# Check if we have enough triplets to select from
if num_available == 0:
    print("Error: The input file contains no triplets. Cannot sample.")
    exit()
elif num_available < NUM_SAMPLES_TO_SELECT:
    print(f"Warning: Only {num_available} triplets available, which is less than the requested {NUM_SAMPLES_TO_SELECT}.")
    print("Selecting all available triplets.")
    selected_triplets = all_fashion_triplets
else:
    print(f"Randomly selecting {NUM_SAMPLES_TO_SELECT} triplets...")
    selected_triplets = random.sample(all_fashion_triplets, NUM_SAMPLES_TO_SELECT)

print(f"Selected {len(selected_triplets)} triplets.")

# Save the randomly sampled subset
print(f"Saving sampled subset to {OUTPUT_JSON_PATH}...")
try:
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(selected_triplets, f, indent=4) # Use indent for readability
    print("Successfully created the random subset file.")
except IOError:
    print(f"Error: Could not write output file to {OUTPUT_JSON_PATH}")
