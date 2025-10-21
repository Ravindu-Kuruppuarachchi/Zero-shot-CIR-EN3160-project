import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from flask import Flask, request, jsonify, render_template_string, url_for, send_from_directory

# --- Local Project Imports ---
# Assumes this script is in the root of your ZS-CIR project
from config import Config
from model.model import TransAgg
from utils import get_preprocess, collate_fn
import model.clip as clip # Import clip for loading

# --- 1. CONFIGURATION: PLEASE UPDATE THESE PATHS ---

# Path to your trained CLIP-based fashion model checkpoint
TRAINED_MODEL_PATH = "D:/Documents 2.0/5th semester/computer vision/Vision Project/epoch_10_laion_combined.pth" 

# Path to the folder containing the Fashion-IQ dataset
# --- CORRECTED PATH BASED ON YOUR INPUT ---
FASHION_IQ_BASE_PATH = "D:/Documents 2.0/5th semester/computer vision/Vision Project/fig"

# Which FashionIQ category to use as the product catalog
CATALOG_CATEGORY = 'shirt'

# --- END CONFIGURATION ---

# --- Global Variables ---
app = Flask(__name__)
model = None
preprocess = None
device = None
index_features = None
index_paths = None

# --- HTML & Frontend Template (No changes needed here) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visual Style Search</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: 'Inter', sans-serif; }
        .loader { border: 4px solid #f3f3f3; border-top: 4px solid #4f46e5; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .product-card:hover .product-image { transform: scale(1.05); }
        .product-card:hover .overlay { opacity: 1; }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
</head>
<body class="bg-gray-100">

    <!-- Modal for Search -->
    <div id="search-modal" class="hidden fixed inset-0 bg-gray-900 bg-opacity-75 flex items-center justify-center z-50 p-4">
        <div class="bg-white rounded-2xl shadow-2xl w-full max-w-md p-6 relative">
            <button onclick="closeModal()" class="absolute top-4 right-4 text-gray-400 hover:text-gray-700">&times;</button>
            <h2 class="text-2xl font-bold text-center mb-4">Find a Similar Style</h2>
            <div class="w-full h-48 bg-gray-200 rounded-lg mb-4">
                <img id="modal-image" src="" class="w-full h-full object-contain rounded-lg">
            </div>
            <input type="text" id="text-input" class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500" placeholder="e.g., 'with long sleeves'">
            <button id="search-btn" class="w-full mt-4 bg-indigo-600 text-white font-semibold py-3 rounded-lg hover:bg-indigo-700 transition">
                Search
            </button>
            <div id="modal-loader" class="hidden mx-auto loader mt-4"></div>
            <p id="modal-error" class="hidden text-red-500 text-center mt-2"></p>
        </div>
    </div>

    <!-- Main Content -->
    <div class="container mx-auto p-4 md:p-8">
        <header class="text-center mb-8">
            <h1 class="text-4xl font-bold text-gray-800">Product Listings</h1>
            <p class="text-gray-600 mt-1">Click the 'Find Similar' button on any product to start a visual search.</p>
        </header>

        <!-- Product Grid -->
        <div id="product-gallery" class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6">
            <!-- Products will be loaded here -->
        </div>
        <div id="gallery-loader" class="mx-auto loader mt-8"></div>
    </div>

<script>
    let currentReferenceImage = null;
    const modal = document.getElementById('search-modal');
    const modalImage = document.getElementById('modal-image');
    const textInput = document.getElementById('text-input');
    const searchBtn = document.getElementById('search-btn');
    const modalLoader = document.getElementById('modal-loader');
    const modalError = document.getElementById('modal-error');
    const productGallery = document.getElementById('product-gallery');
    const galleryLoader = document.getElementById('gallery-loader');

    function openModal(imageSrc) {
        currentReferenceImage = imageSrc;
        modalImage.src = imageSrc;
        modal.classList.remove('hidden');
        textInput.value = '';
        modalError.classList.add('hidden');
    }

    function closeModal() {
        modal.classList.add('hidden');
    }

    async function performSearch() {
        const text = textInput.value.trim();
        if (!text) {
            modalError.textContent = 'Please enter a modification.';
            modalError.classList.remove('hidden');
            return;
        }

        modalLoader.classList.remove('hidden');
        searchBtn.disabled = true;
        modalError.classList.add('hidden');

        try {
            const response = await fetch('/visual-search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_path: currentReferenceImage, text: text })
            });
            const data = await response.json();

            if (data.error) {
                modalError.textContent = data.error;
                modalError.classList.remove('hidden');
            } else if (data.results) {
                displayResults(data.results);
                closeModal();
            }
        } catch (err) {
            modalError.textContent = 'A network error occurred.';
            modalError.classList.remove('hidden');
        } finally {
            modalLoader.classList.add('hidden');
            searchBtn.disabled = false;
        }
    }

    function displayResults(imagePaths) {
        productGallery.innerHTML = '';
        imagePaths.forEach(path => {
            productGallery.appendChild(createProductCard(path));
        });
        document.documentElement.scrollTop = 0;
    }

    function createProductCard(imagePath) {
        const card = document.createElement('div');
        card.className = 'bg-white rounded-lg shadow-md overflow-hidden product-card';
        const imageContainer = document.createElement('div');
        imageContainer.className = 'relative aspect-square overflow-hidden';
        const img = document.createElement('img');
        img.src = imagePath;
        img.className = 'w-full h-full object-cover product-image transition-transform duration-300';
        const overlay = document.createElement('div');
        overlay.className = 'absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center opacity-0 transition-opacity duration-300 overlay';
        const button = document.createElement('button');
        button.textContent = 'Find Similar';
        button.className = 'bg-white text-gray-800 font-semibold py-2 px-4 rounded-full text-sm';
        button.onclick = () => openModal(imagePath);
        overlay.appendChild(button);
        imageContainer.appendChild(img);
        imageContainer.appendChild(overlay);
        card.appendChild(imageContainer);
        return card;
    }

    searchBtn.addEventListener('click', performSearch);
    
    window.addEventListener('DOMContentLoaded', async () => {
        try {
            const response = await fetch('/get-initial-products');
            const data = await response.json();
            if(data.products) {
                displayResults(data.products);
            }
        } finally {
            galleryLoader.classList.add('hidden');
        }
    });
</script>
</body>
</html>
"""

# --- Backend Logic ---

# FIX: Move the GalleryDataset class outside of any function to make it global
class GalleryDataset(torch.utils.data.Dataset):
    def __init__(self, paths, preprocess_fn):
        self.paths = paths
        self.preprocess_fn = preprocess_fn
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        try:
            image = Image.open(self.paths[idx]).convert("RGB")
            return self.preprocess_fn(image)
        except Exception: return None

def load_model_and_index():
    """Loads the model and pre-computes the product catalog index."""
    global model, preprocess, device, index_features, index_paths

    print("--- Initializing E-commerce Visual Search ---")
    cfg = Config()
    
    cfg.model_name = "clip-Vit-B/32"
    cfg.encoder = "text"
    device = cfg.device
    print(f"Using device: {device}")

    model = TransAgg(cfg)
    model = model.to(device)

    if not os.path.exists(TRAINED_MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at: {TRAINED_MODEL_PATH}. Please update the path.")
    
    print(f"Loading trained weights from: {TRAINED_MODEL_PATH}")
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=device), strict=False)
    model.eval()
    print("Model loaded successfully.")

    input_dim = model.pretrained_model.visual.input_resolution
    preprocess = get_preprocess(cfg, model, input_dim)

    gallery_path = os.path.join(FASHION_IQ_BASE_PATH, 'Fashion-IQ', 'fashion-iq', 'images')
    print(f"Building product index from: {gallery_path}")
    if not os.path.isdir(gallery_path):
        raise NotADirectoryError(f"Product image directory not found: {gallery_path}")

    split_file = os.path.join(FASHION_IQ_BASE_PATH, 'Fashion-IQ', 'fashion-iq', 'image_splits', f'split.{CATALOG_CATEGORY}.val.json')
    with open(split_file, 'r') as f:
        category_image_names = json.load(f)

    all_image_paths = [os.path.join(gallery_path, name + ".jpg") for name in category_image_names]
    valid_image_paths = [p for p in all_image_paths if os.path.exists(p)]
    
    if not valid_image_paths:
        raise FileNotFoundError(f"No valid images found for category '{CATALOG_CATEGORY}'. Check paths.")

    dataset = GalleryDataset(valid_image_paths, preprocess)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=2, collate_fn=collate_fn)

    features_list = []
    with torch.no_grad():
        for batch_images in tqdm(dataloader, desc="Indexing products"):
            batch_images = batch_images.to(device)
            features = model.pretrained_model.encode_image(batch_images)
            features_list.append(features.cpu())

    index_features = torch.cat(features_list, dim=0).to(device)
    index_paths = valid_image_paths
    print(f"Product index created with {len(index_paths)} items.")
    print("--- Application Ready ---")


@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/get-initial-products')
def get_initial_products():
    num_products = 50
    random_indices = torch.randperm(len(index_paths))[:num_products]
    product_urls = [url_for('serve_product_image', filename=os.path.basename(index_paths[i])) for i in random_indices]
    return jsonify({'products': product_urls})

@app.route('/visual-search', methods=['POST'])
def visual_search():
    data = request.get_json()
    image_url = data.get('image_path')
    mod_text = data.get('text')

    if not image_url or not mod_text:
        return jsonify({'error': 'Missing image or text'}), 400

    try:
        image_filename = os.path.basename(image_url)
        reference_image_path = os.path.join(FASHION_IQ_BASE_PATH, 'Fashion-IQ', 'fashion-iq', 'images', image_filename)
        
        image = Image.open(reference_image_path).convert("RGB")
        preprocessed_image = preprocess(image).unsqueeze(0).to(device)
        
        text_tokens = clip.tokenize([mod_text]).to(device)

        with torch.no_grad():
            query_feature = model.combine_features(preprocessed_image, text_tokens)
            query_feature /= query_feature.norm(dim=-1, keepdim=True)
        
        normalized_index = index_features / index_features.norm(dim=-1, keepdim=True)
        similarities = (query_feature @ normalized_index.T).squeeze(0)
        top_k_indices = torch.topk(similarities, k=20).indices

        result_urls = [url_for('serve_product_image', filename=os.path.basename(index_paths[i])) for i in top_k_indices]
        
        return jsonify({'results': result_urls})

    except Exception as e:
        print(f"Error during search: {e}")
        return jsonify({'error': 'Failed to process search request.'}), 500

@app.route('/products/<path:filename>')
def serve_product_image(filename):
    image_dir = os.path.join(FASHION_IQ_BASE_PATH, 'Fashion-IQ', 'fashion-iq', 'images')
    return send_from_directory(image_dir, filename)


if __name__ == '__main__':
    load_model_and_index()
    app.run(host='0.0.0.0', port=5000, debug=False)

