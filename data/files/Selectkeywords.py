import json
import random
from collections import Counter
import re

# --- Configuration ---
INPUT_JSON_PATH = 'laion_template_info.json' # Path to your Laion-Template JSON file
OUTPUT_JSON_PATH = 'laion_template_fashion_all_matches.json' # Output file
# NUM_TRIPLETS_TO_SELECT = 2000 # No longer needed

# --- Fashion-Related Keywords ---
FASHION_KEYWORDS = [
    # Garment Types
    'dress', 'shirt', 't-shirt', 'top', 'blouse', 'sweater', 'sweatshirt', 'hoodie',
    'jacket', 'coat', 'vest', 'blazer',
    'pants', 'trousers', 'jeans', 'shorts', 'leggings',
    'skirt', 'suit', 'gown', 'mini', 'maxi',
    'shoe', 'boot', 'sandal', 'sneaker', 'heel', 'footwear',
    'hat', 'cap', 'beanie',
    'scarf', 'glove', 'mitten',
    'sock', 'stocking',
    'underwear', 'bra', 'briefs',
    'swimsuit', 'bikini',

    # Attributes & Concepts
    'clothing', 'clothes', 'garment', 'apparel', 'outfit', 'wear', 'wearing', 'worn',
    'fashion', 'style', 'stylish', 'vogue', 'vintage', 'elegant', 'ethnic', 'corporate',
    'fabric', 'textile', 'material', 'denim', 'cotton', 'leather', 'silk', 'wool', 'lace', 'sheer',

    # Color/Shade (Expanded)
    'color', 'colored', 'hue', 'shade', 'solid', 'multi-colored', 'lighter', 'darker',
    'red', 'blue', 'green', 'yellow', 'black', 'white', 'grey', 'gray', 'pink', 'purple', 'orange', 'brown',
    'cream', 'tan', 'navy', 'mint', 'gold', 'silver', 'metallic',

    # Pattern/Print (Expanded)
    'pattern', 'print', 'striped', 'striping', 'checked', 'checkered', 'plaid', 'dotted', 'polka dots',
    'floral', 'geometric', 'diamond', 'graphics', 'graphic', 'logo', 'name', 'designs',

    # Features/Details (Expanded)
    'sleeve', 'sleeveless', 'short sleeves', 'long sleeves', 'cap sleeves', 'three quarter length',
    'collar', 'open collar', 'neckline', 'plunging neckline',
    'hem', 'high-low hem', 'asymmetrical hem', 'ruffled hem',
    'waist', 'belt', 'belted', 'bow', 'buttons', 'button down',
    'pocket', 'ruched', 'embellishments', 'sequin', 'overlay',
    'straps', 'spaghetti straps', 'thin straps', 'strapless',

    # Fit/Style (Expanded)
    'long', 'short', 'longer', 'shorter', 'knee-length', 'floor-length',
    'loose', 'looser', 'tight', 'tighter', 'fitted', 'form fitted', 'wrap around', 'flowing', 'revealing',

    # Accessories
    'accessory', 'bag', 'jewelry', 'watch', 'glasses',

    # Actions
    'put on', 'take off', 'try on', 'button', 'zip',
]
# Pre-compile regex for finding whole words (case-insensitive)
KEYWORD_REGEX = re.compile(r'\b(' + '|'.join(FASHION_KEYWORDS) + r')\b', re.IGNORECASE)

# --- Helper Function ---
def contains_fashion_keyword(triplet_data):
    """Checks if any fashion keyword exists in the captions."""
    text_to_check = ""
    # --- CORRECTED KEYS based on your JSON file ---
    if 'reference_caption' in triplet_data and triplet_data['reference_caption']:
        text_to_check += triplet_data['reference_caption'].lower() + " "
    if 'target_caption' in triplet_data and triplet_data['target_caption']:
        text_to_check += triplet_data['target_caption'].lower() + " "
    if 'relative_cap' in triplet_data and triplet_data['relative_cap']:
        text_to_check += triplet_data['relative_cap'].lower() + " "
    # --- ---

    # Check if any keyword matches
    match = KEYWORD_REGEX.search(text_to_check)
    return match is not None # Return True if at least one match found, False otherwise

# --- Main Logic ---
print(f"Loading data from {INPUT_JSON_PATH}...")
try:
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        all_triplets = json.load(f)
except FileNotFoundError:
    print(f"Error: Input file not found at {INPUT_JSON_PATH}")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {INPUT_JSON_PATH}. Check file format.")
    exit()

print(f"Loaded {len(all_triplets)} triplets.")
print("Filtering for triplets containing fashion-related keywords...")

fashion_triplets = []
for i, triplet in enumerate(all_triplets):
    # Make sure the item is a dictionary
    if isinstance(triplet, dict):
        if contains_fashion_keyword(triplet):
            fashion_triplets.append(triplet) # Add the triplet if it contains a keyword
    else:
        print(f"Warning: Item at index {i} is not a dictionary, skipping.")


print(f"Found {len(fashion_triplets)} triplets with fashion-related keywords.")

# Save the subset containing all matches
print(f"Saving {len(fashion_triplets)} selected triplets to {OUTPUT_JSON_PATH}...")
try:
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(fashion_triplets, f, indent=4) # Use indent for readability
    print("Subset saved successfully.")
except IOError:
    print(f"Error: Could not write output file to {OUTPUT_JSON_PATH}")