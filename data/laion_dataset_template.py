from torch.utils.data import Dataset
import json
import PIL
from PIL import Image
from PIL import ImageFile
import os

# These settings help prevent crashes from corrupt or very large image files
Image.MAX_IMAGE_PIXELS = 2300000000
ImageFile.LOAD_TRUNCATED_IMAGES = True


class LaionDataset_Template(Dataset):
    def __init__(self, split: str, preprocess: callable):
        self.preprocess = preprocess
        self.split = split

        if split not in ['train']:
            raise ValueError("This dataset class is only for the 'train' split.")

        # --- PATH CONFIGURATION ---

        # 1. Path to the folder containing your JSON training file
        # This should be the '.../data/files' directory
        json_folder_path = ""

        # 2. Path to the folder containing your PNG image files
        # This is the path you provided for the images
        self.image_path_prefix = ""

        # --- END PATH CONFIGURATION ---


        # Load the JSON file with the list of triplets
        # This will be either 'fashion_train_subset_2000.json' or another training file
        json_path = os.path.join(json_folder_path, 'fashion_train_subset_2000.json')
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.triplets = json.load(f)
        except FileNotFoundError:
            print(f"FATAL ERROR: The training file was not found at {json_path}")
            print("Please ensure the JSON file is in the correct directory.")
            exit() # Exit if the main data file is missing

        print(f"Laion '{split}' dataset initialized with {len(self.triplets)} triplets.")
        print(f"Looking for images in: {self.image_path_prefix}")


    def __getitem__(self, index):
        try:
            # Get the triplet metadata from the JSON file
            triplet_info = self.triplets[index]
            relative_caption = triplet_info['relative_cap']

            # Format the image filenames
            reference_image_name = f"{str(triplet_info['ref_image_id']).zfill(7)}.png"
            target_image_name = f"{str(triplet_info['tgt_image_id']).zfill(7)}.png"

            # Use os.path.join to correctly create the full file path from the image prefix
            reference_image_path = os.path.join(self.image_path_prefix, reference_image_name)
            target_image_path = os.path.join(self.image_path_prefix, target_image_name)

            # Load, convert, and preprocess the reference image
            reference_image = Image.open(reference_image_path).convert('RGB')
            reference_image = self.preprocess(reference_image)

            # Load, convert, and preprocess the target image
            target_image = Image.open(target_image_path).convert('RGB')
            target_image = self.preprocess(target_image)

            return reference_image, target_image, relative_caption

        # If an image is missing or corrupt, this prevents the script from crashing.
        # It returns None, and the `collate_fn` in utils.py will skip this item.
        except (FileNotFoundError, PIL.UnidentifiedImageError) as e:
            print(f"Warning: Skipping item at index {index} due to error: {e}")
            return None

    def __len__(self):
        return len(self.triplets)

