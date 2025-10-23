import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from flask import Flask, request, jsonify, render_template_string, url_for, send_from_directory
import random

# --- Local Project Imports ---
from config import Config
from model.model import TransAgg
from utils import get_preprocess, collate_fn
import model.clip as clip

# --- 1. CONFIGURATION ---
TRAINED_MODEL_PATH = "D:/Documents 2.0/5th semester/computer vision/Vision Project/epoch_10_laion_combined.pth"
FASHION_IQ_BASE_PATH = "D:/Documents 2.0/5th semester/computer vision/Vision Project/fig"
CATALOG_CATEGORIES = ['shirt', 'dress', 'household', 'toys']

# --- Global Variables & App State ---
app = Flask(__name__)
model = None
preprocess = None
device = None
index_features = {}
index_paths = {}

cart_items = {} # Will store { 'product_name': {'details': product_info, 'quantity': count} }
wishlist_items = set() # Will store product names

# --- HTML & Frontend Templates ---
# Main Home Page Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>StyleNStay - Visual Search</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: 'Inter', sans-serif; }
        .loader { border: 4px solid #f3f3f3; border-top: 4px solid #4f46e5; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .product-card:hover .product-image { transform: scale(1.05); }
        .product-card:hover .overlay { opacity: 1; }
        input[type="radio"]:checked + label {
            background-color: #4f46e5;
            color: white;
            border-color: #4f46e5;
        }
        .wishlist-btn.active svg { fill: #ef4444; color: #ef4444; }
        .animate-fade-in { animation: fadeIn 0.3s ease-out; }
        @keyframes fadeIn { from { opacity: 0; transform: scale(0.95); } to { opacity: 1; transform: scale(1); } }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://unpkg.com/lucide@latest"></script>
</head>
<body class="bg-gray-50">

    <!-- Modal for Search -->
    <div id="search-modal" class="hidden fixed inset-0 bg-gray-900 bg-opacity-75 flex items-center justify-center z-50 p-4">
        <div class="bg-white rounded-2xl shadow-2xl w-full max-w-md p-6 relative animate-fade-in">
            <button onclick="closeModal()" class="absolute top-4 right-4 text-gray-400 hover:text-gray-700 focus:outline-none">&times;</button>
            <h2 class="text-2xl font-bold text-center mb-4">Find a Similar Style</h2>
            <div class="w-full h-48 bg-gray-200 rounded-lg mb-4"><img id="modal-image" src="" class="w-full h-full object-contain rounded-lg"></div>
            <input type="text" id="text-input" class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500" placeholder="e.g., 'with long sleeves'">
            <button id="search-btn" class="w-full mt-4 bg-indigo-600 text-white font-semibold py-3 rounded-lg hover:bg-indigo-700 transition">Search</button>
            <div id="modal-loader" class="hidden mx-auto loader mt-4"></div>
            <p id="modal-error" class="hidden text-red-500 text-center mt-2"></p>
        </div>
    </div>

    <!-- Main Content -->
    <div class="container mx-auto">
        <!-- Header -->
        <header class="bg-white shadow-sm sticky top-0 z-40">
            <nav class="mx-auto flex max-w-7xl items-center justify-between p-4 lg:px-8">
                <div class="flex items-center gap-x-4">
                    <a href="/" class="-m-1.5 p-1.5 flex items-center">
                        <svg class="h-8 w-auto text-indigo-600" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
                        <span class="ml-2 text-xl font-bold text-gray-800">StyleNStay</span>
                    </a>
                </div>
                <div class="hidden lg:flex lg:gap-x-12 items-center">
                     <div id="category-selector" class="flex justify-center space-x-2"></div>
                </div>
                <div class="flex items-center gap-x-6">
                    <a href="/wishlist" class="text-gray-600 hover:text-indigo-600 relative">
                        <i data-lucide="heart"></i>
                        <span id="wishlist-counter" class="absolute -top-2 -right-2 bg-red-500 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center">0</span>
                    </a>
                    <a href="/cart" class="text-gray-600 hover:text-indigo-600 relative">
                        <i data-lucide="shopping-cart"></i>
                         <span id="cart-counter" class="absolute -top-2 -right-2 bg-red-500 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center">0</span>
                    </a>
                </div>
            </nav>
        </header>

        <!-- Hero Banner -->
        <section class="relative bg-gradient-to-r from-indigo-500 to-purple-500 text-white py-20 px-4 text-center">
            <h1 class="text-4xl md:text-5xl font-extrabold tracking-tight">Find Exactly What You're Looking For</h1>
            <p class="mt-4 max-w-2xl mx-auto text-lg text-indigo-100">Visually search for products and modify them with your own words. Your next find is just a search away.</p>
        </section>

        <main class="max-w-7xl mx-auto p-4 md:p-8">
            <div class="flex flex-col md:flex-row justify-between items-center mb-6 bg-white p-4 rounded-lg shadow-sm">
                <div class="flex items-center space-x-2 mb-4 md:mb-0">
                    <span class="font-semibold">Filter by:</span>
                    <button class="filter-btn px-3 py-1 text-sm border rounded-full hover:bg-gray-200" data-filter="trending">Trending</button>
                    <button class="filter-btn px-3 py-1 text-sm border rounded-full hover:bg-gray-200" data-filter="top-rated">Top Rated</button>
                    <button class="filter-btn px-3 py-1 text-sm border rounded-full hover:bg-gray-200" data-filter="sale">On Sale</button>
                </div>
                <div>
                    <select id="sort-select" class="border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500">
                        <option value="latest">Sort by: Latest</option>
                        <option value="price-asc">Price: Low to High</option>
                        <option value="price-desc">Price: High to Low</option>
                        <option value="rating">Highest Rated</option>
                    </select>
                </div>
            </div>

            <div id="product-gallery" class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6"></div>
            <div id="gallery-loader" class="mx-auto loader mt-8"></div>
        </main>
    </div>

<script>
    let currentReferenceImage = null;
    let allProducts = [];
    const modal=document.getElementById('search-modal'),modalImage=document.getElementById('modal-image'),textInput=document.getElementById('text-input'),searchBtn=document.getElementById('search-btn'),modalLoader=document.getElementById('modal-loader'),modalError=document.getElementById('modal-error'),productGallery=document.getElementById('product-gallery'),galleryLoader=document.getElementById('gallery-loader'),categorySelector=document.getElementById('category-selector'),wishlistCounter=document.getElementById('wishlist-counter'),cartCounter=document.getElementById('cart-counter'),sortSelect=document.getElementById('sort-select');

    function getSelectedCategory(){return document.querySelector('input[name="category"]:checked')?.value||null}
    function openModal(a){currentReferenceImage=a;modalImage.src=a;modal.classList.remove('hidden');textInput.value='';modalError.classList.add('hidden')}
    function closeModal(){modal.classList.add('hidden')}

    

    async function performSearch(){
        const a=textInput.value.trim(),b=getSelectedCategory();
        if(!a){modalError.textContent='Please enter a modification.';modalError.classList.remove('hidden');return}
        if(!b){modalError.textContent='Please select a category.';modalError.classList.remove('hidden');return}
        modalLoader.classList.remove('hidden');searchBtn.disabled=!0;modalError.classList.add('hidden');
        try{const c=await fetch('/visual-search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image_path:currentReferenceImage,text:a,category:b})}),d=await c.json();d.error?(modalError.textContent=d.error,modalError.classList.remove('hidden')):d.results&&(allProducts=d.results,displayResults(d.results),closeModal())}catch(c){modalError.textContent='A network error occurred.';modalError.classList.remove('hidden')}finally{modalLoader.classList.add('hidden');searchBtn.disabled=!1}
    }




    function displayResults(a){productGallery.innerHTML='';a.forEach(b=>productGallery.appendChild(createProductCard(b)));document.documentElement.scrollTop=0; lucide.createIcons();}
    
    function createProductCard(product) {
        const card = document.createElement('div');
        card.className = 'bg-white rounded-lg shadow-md overflow-hidden product-card flex flex-col';
        card.dataset.product = JSON.stringify(product);
        
        let badgeHTML = '';
        if (product.badge.text) {
            const color = product.badge.type === 'sale' ? 'bg-red-500' : product.badge.type === 'new' ? 'bg-blue-500' : 'bg-yellow-500';
            badgeHTML = `<div class="absolute top-2 left-2 text-xs font-bold text-white px-2 py-1 rounded-full ${color}">${product.badge.text}</div>`;
        }

        const starHTML = Array(5).fill(0).map((_, i) => i < Math.floor(product.rating) ? '<i data-lucide="star" class="w-4 h-4 text-yellow-400 fill-current"></i>' : '<i data-lucide="star" class="w-4 h-4 text-gray-300"></i>').join('');

        card.innerHTML = `
            <div class="relative aspect-square overflow-hidden">
                <img src="${product.path}" class="w-full h-full object-cover product-image transition-transform duration-300">
                ${badgeHTML}
                <button class="absolute top-2 right-2 text-gray-300 hover:text-red-500 wishlist-btn ${product.wished ? 'active' : ''}" onclick="toggleWishlist(this, event)">
                    <i data-lucide="heart" class="w-6 h-6"></i>
                </button>
                <div class="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center opacity-0 transition-opacity duration-300 overlay">
                    <button class="bg-white text-gray-800 font-semibold py-2 px-4 rounded-full text-sm" onclick="openModal('${product.path}')">Find Similar</button>
                </div>
            </div>
            <div class="p-4 flex flex-col flex-grow">
                <h3 class="font-semibold text-sm text-gray-700 truncate">${product.name}</h3>
                <div class="flex items-center mt-2">
                    <div class="flex">${starHTML}</div>
                    <span class="text-xs text-gray-500 ml-2">(${product.reviews})</span>
                </div>
                <div class="mt-2 flex items-baseline gap-x-2">
                    <span class="text-lg font-bold text-gray-800">$${product.price.toFixed(2)}</span>
                    ${product.originalPrice > product.price ? `<span class="text-sm text-gray-500 line-through">$${product.originalPrice.toFixed(2)}</span>` : ''}
                </div>
                <p class="text-xs text-gray-500 mt-1">${product.sold} sold</p>
                <button class="w-full mt-4 bg-indigo-100 text-indigo-700 font-semibold py-2 rounded-lg hover:bg-indigo-200 transition text-sm" onclick="addToCart(this, event)">Add to Cart</button>
            </div>
        `;
        return card;
    }

    async function toggleWishlist(btn, event) {
        event.stopPropagation();
        const card = btn.closest('.product-card');
        const product = JSON.parse(card.dataset.product);
        const isWished = !btn.classList.contains('active');
        
        const response = await fetch('/api/wishlist', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ product_name: product.name, wished: isWished })
        });
        const data = await response.json();
        wishlistCounter.textContent = data.count;
        btn.classList.toggle('active', isWished);
    }
    
    async function addToCart(btn, event) {
        event.stopPropagation();
        const card = btn.closest('.product-card');
        const product = JSON.parse(card.dataset.product);
        
        const response = await fetch('/api/add-to-cart', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ product: product })
        });
        const data = await response.json();
        cartCounter.textContent = data.count;
        
        btn.textContent = 'Added!';
        setTimeout(() => { btn.textContent = 'Add to Cart'; }, 1000);
    }

    async function loadInitialProducts(category) {
        galleryLoader.classList.remove('hidden');
        productGallery.innerHTML = '';
        try {
            const response = await fetch(`/get-initial-products?category=${category}`);
            const data = await response.json();
            if (data.products) { allProducts = data.products; displayResults(allProducts); }
        } finally {
            galleryLoader.classList.add('hidden');
        }
    }
    
    async function updateCounters() {
        const response = await fetch('/api/counters');
        const data = await response.json();
        wishlistCounter.textContent = data.wishlist_count;
        cartCounter.textContent = data.cart_count;
    }

    function sortProducts(sortBy) {
        let sorted = [...allProducts];
        if (sortBy === 'price-asc') sorted.sort((a,b) => a.price - b.price);
        else if (sortBy === 'price-desc') sorted.sort((a,b) => b.price - a.price);
        else if (sortBy === 'rating') sorted.sort((a,b) => b.rating - a.rating);
        displayResults(sorted);
    }
    
    sortSelect.addEventListener('change', (e) => sortProducts(e.target.value));
    searchBtn.addEventListener('click', performSearch);
    
    window.addEventListener('DOMContentLoaded', async () => {
        lucide.createIcons();
        updateCounters();
        const response = await fetch('/get-categories');
        const data = await response.json();
        const categories = data.categories || [];
        
        let firstAvailableCategory = null;

        categories.forEach((catInfo, index) => {
            const radioId = `cat-${catInfo.name}`;
            const radio = document.createElement('input');
            radio.type = 'radio'; radio.id = radioId; radio.name = 'category'; radio.value = catInfo.name; radio.className = 'hidden';
            
            if (catInfo.available && !firstAvailableCategory) {
                firstAvailableCategory = catInfo.name;
                radio.checked = true;
            }

            radio.onchange = () => {
                if (catInfo.available) {
                    loadInitialProducts(catInfo.name);
                } else {
                    galleryLoader.classList.add('hidden');
                    productGallery.innerHTML = `<div class='col-span-full text-center p-10 bg-gray-100 rounded-lg'>
                        <h3 class='text-xl font-semibold text-gray-700'>Coming Soon!</h3>
                        <p class='text-gray-500 mt-2'>The '${catInfo.name}' category is under construction.</p>
                    </div>`;
                }
            };
            
            const label = document.createElement('label');
            label.htmlFor = radioId; label.textContent = catInfo.name.charAt(0).toUpperCase() + catInfo.name.slice(1);
            if(catInfo.available) {
                label.className = 'px-4 py-2 border rounded-full cursor-pointer transition-colors';
            } else {
                label.className = 'px-4 py-2 border border-gray-300 bg-gray-100 text-gray-400 rounded-full cursor-not-allowed';
            }

            categorySelector.appendChild(radio);
            categorySelector.appendChild(label);
        });

        if (firstAvailableCategory) {
            loadInitialProducts(firstAvailableCategory);
        } else {
            galleryLoader.classList.add('hidden');
            productGallery.innerHTML = '<p class="text-center text-gray-500 col-span-full">No product categories loaded.</p>';
        }
    });
</script>
</body>
</html>
"""

# Cart Page Template
CART_PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Your Shopping Cart - StyleNStay</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style> body { font-family: 'Inter', sans-serif; } </style>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://unpkg.com/lucide@latest"></script>
</head>
<body class="bg-gray-100">
    <header class="bg-white shadow-sm">
        <nav class="mx-auto flex max-w-7xl items-center justify-between p-4 lg:px-8">
            <a href="/" class="-m-1.5 p-1.5 flex items-center">
                <svg class="h-8 w-auto text-indigo-600" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
                <span class="ml-2 text-xl font-bold text-gray-800">StyleNStay</span>
            </a>
            <div><a href="/" class="text-indigo-600 hover:underline">Continue Shopping</a></div>
        </nav>
    </header>
    <main class="max-w-4xl mx-auto p-4 md:p-8">
        <h1 class="text-3xl font-bold text-gray-800 mb-6">Your Shopping Cart</h1>
        <div id="cart-container" class="bg-white rounded-lg shadow-md">
            <!-- Cart items will be loaded here -->
        </div>
    </main>
<script>
    async function loadCartItems() {
        const response = await fetch('/api/get-cart');
        const data = await response.json();
        const container = document.getElementById('cart-container');
        
        if (Object.keys(data.cart).length === 0) {
            container.innerHTML = `<p class="p-8 text-center text-gray-500">Your cart is empty.</p>`;
            return;
        }

        let subtotal = 0;
        let cartHTML = '<ul class="divide-y divide-gray-200">';

        for (const [name, item] of Object.entries(data.cart)) {
            const itemTotal = item.details.price * item.quantity;
            subtotal += itemTotal;
            cartHTML += `
                <li class="p-4 sm:p-6 flex space-x-4">
                    <img src="${item.details.path}" class="w-24 h-24 rounded-md object-cover">
                    <div class="flex-1 flex flex-col justify-between">
                        <div>
                            <h3 class="font-semibold text-gray-800">${item.details.name}</h3>
                            <p class="text-sm text-gray-500">Quantity: ${item.quantity}</p>
                        </div>
                        <p class="font-semibold text-gray-900">$${itemTotal.toFixed(2)}</p>
                    </div>
                </li>
            `;
        }
        cartHTML += '</ul>';

        cartHTML += `
            <div class="p-6 border-t border-gray-200">
                <div class="flex justify-between items-center mb-4">
                    <span class="text-lg font-medium text-gray-600">Subtotal</span>
                    <span class="text-xl font-bold text-gray-900">$${subtotal.toFixed(2)}</span>
                </div>
                <button class="w-full bg-indigo-600 text-white font-semibold py-3 rounded-lg hover:bg-indigo-700 transition">Proceed to Checkout</button>
            </div>
        `;
        container.innerHTML = cartHTML;
    }
    
    window.addEventListener('DOMContentLoaded', loadCartItems);
</script>
</body>
</html>
"""

# Wishlist Page Template (Placeholder)
WISHLIST_PAGE_TEMPLATE = """
<!DOCTYPE html><html lang="en"><head><title>Wishlist</title></head><body><h1>Wishlist Page</h1><p>Under construction.</p><a href="/">Go Back</a></body></html>
"""


# --- Backend Logic ---

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

def generate_product_details(image_path, wished_set):
    """Generates random mock data for a product."""
    product_name = os.path.basename(image_path).replace('.jpg', '').replace('_', ' ').title()
    original_price = random.uniform(25.0, 150.0)
    discount = random.uniform(0.1, 0.4) if random.random() > 0.5 else 0
    price = original_price * (1 - discount)
    badge_type = 'none'; badge_text = ''
    if discount > 0: badge_type = 'sale'; badge_text = f"{int(discount * 100)}% OFF"
    elif random.random() > 0.8: badge_type = 'new'; badge_text = 'New'
    elif random.random() > 0.7: badge_type = 'trending'; badge_text = 'Trending'
    return {'path': url_for('serve_product_image', filename=os.path.basename(image_path)),'name': product_name,'price': price,'originalPrice': original_price,'rating': round(random.uniform(3.5, 5.0) * 2) / 2,'reviews': random.randint(10, 500),'sold': random.randint(100, 10000),'badge': {'type': badge_type, 'text': badge_text}, 'wished': product_name in wished_set }

def load_model_and_index():
    """Loads model and pre-computes indexes for multiple categories."""
    global model, preprocess, device, index_features, index_paths
    print("--- Initializing E-commerce Visual Search ---")
    cfg = Config(); cfg.model_name = "clip-Vit-B/32"; cfg.encoder = "text"; device = cfg.device
    print(f"Using device: {device}")
    model = TransAgg(cfg); model = model.to(device)
    if not os.path.exists(TRAINED_MODEL_PATH): raise FileNotFoundError(f"Trained model not found at: {TRAINED_MODEL_PATH}")
    print(f"Loading trained weights from: {TRAINED_MODEL_PATH}")
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=device), strict=False)
    model.eval(); print("Model loaded successfully.")
    input_dim = model.pretrained_model.visual.input_resolution
    preprocess = get_preprocess(cfg, model, input_dim)
    
    for category in CATALOG_CATEGORIES:
        print(f"\n--- Building index for category: '{category}' ---")
        gallery_path = os.path.join(FASHION_IQ_BASE_PATH, 'Fashion-IQ', 'fashion-iq', 'images')
        split_file = os.path.join(FASHION_IQ_BASE_PATH, 'Fashion-IQ', 'fashion-iq', 'image_splits', f'split.{category}.val.json')
        if not os.path.exists(split_file): print(f"Warning: Split file for '{category}' not found. Skipping."); continue
        with open(split_file, 'r') as f: category_image_names = json.load(f)
        all_image_paths = [os.path.join(gallery_path, name + ".jpg") for name in category_image_names]
        valid_image_paths = [p for p in all_image_paths if os.path.exists(p)]
        if not valid_image_paths: print(f"Warning: No valid images found for '{category}'."); continue
        dataset = GalleryDataset(valid_image_paths, preprocess)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=0, collate_fn=collate_fn)
        features_list = []
        with torch.no_grad():
            for batch_images in tqdm(dataloader, desc=f"Indexing {category}"):
                batch_images = batch_images.to(device)
                encoded_output = model.pretrained_model.encode_image(batch_images)
                features = encoded_output[0] if isinstance(encoded_output, tuple) else encoded_output
                features_list.append(features.cpu())
        if not features_list: continue
        index_features[category] = torch.cat(features_list, dim=0).to(device)
        index_paths[category] = valid_image_paths
        print(f"'{category}' index created with {len(valid_image_paths)} items.")
    print("\n--- Application Ready ---")

# --- Page Routes ---
@app.route('/')
def home(): return render_template_string(HTML_TEMPLATE)

@app.route('/cart')
def cart_page(): return render_template_string(CART_PAGE_TEMPLATE)

@app.route('/wishlist')
def wishlist_page(): return render_template_string(WISHLIST_PAGE_TEMPLATE)

# --- API Routes ---
@app.route('/get-categories')
def get_categories():
    category_status = [{'name': cat, 'available': cat in index_features} for cat in CATALOG_CATEGORIES]
    return jsonify({'categories': category_status})

@app.route('/get-initial-products')
def get_initial_products():
    category = request.args.get('category', default=CATALOG_CATEGORIES[0], type=str)
    if category not in index_paths: return jsonify({'error': 'Invalid category'}), 400
    num_products = 50; paths_for_category = index_paths[category]
    if not paths_for_category: return jsonify({'products': []})
    random_indices = torch.randperm(len(paths_for_category))[:num_products]
    products = [generate_product_details(paths_for_category[i], wishlist_items) for i in random_indices]
    return jsonify({'products': products})

@app.route('/visual-search', methods=['POST'])
def visual_search():
    data = request.get_json()
    image_url, mod_text, category = data.get('image_path'), data.get('text'), data.get('category')
    if not all([image_url, mod_text, category]): return jsonify({'error': 'Missing data'}), 400
    if category not in index_features: return jsonify({'error': 'Category not available'}), 400
    try:
        image_filename = os.path.basename(image_url)
        reference_image_path = os.path.join(FASHION_IQ_BASE_PATH, 'Fashion-IQ', 'fashion-iq', 'images', image_filename)
        image = Image.open(reference_image_path).convert("RGB")
        preprocessed_image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            query_feature = model.combine_features(preprocessed_image, [mod_text])
            query_feature /= query_feature.norm(dim=-1, keepdim=True)
        current_index_features = index_features[category]; current_index_paths = index_paths[category]
        normalized_index = current_index_features / current_index_features.norm(dim=-1, keepdim=True)
        similarities = (query_feature.to(torch.float32) @ normalized_index.to(torch.float32).T).squeeze(0)
        top_k_indices = torch.topk(similarities, k=20).indices
        results = [generate_product_details(current_index_paths[i], wishlist_items) for i in top_k_indices]
        return jsonify({'results': results})
    except Exception as e:
        print(f"Error during search: {e}")
        return jsonify({'error': 'Failed to process request.'}), 500


@app.route('/api/add-to-cart', methods=['POST'])
def add_to_cart_api():
    data = request.get_json()
    product = data.get('product')
    if not product: return jsonify({'error': 'No product data'}), 400
    
    product_name = product['name']
    if product_name in cart_items:
        cart_items[product_name]['quantity'] += 1
    else:
        cart_items[product_name] = {'details': product, 'quantity': 1}
    
    total_items = sum(item['quantity'] for item in cart_items.values())
    return jsonify({'success': True, 'count': total_items})

@app.route('/api/wishlist', methods=['POST'])
def wishlist_api():
    data = request.get_json()
    product_name = data.get('product_name')
    wished = data.get('wished')
    if wished:
        wishlist_items.add(product_name)
    else:
        wishlist_items.discard(product_name)
    return jsonify({'success': True, 'count': len(wishlist_items)})
    
@app.route('/api/counters')
def get_counters():
    total_cart_items = sum(item['quantity'] for item in cart_items.values())
    return jsonify({'cart_count': total_cart_items, 'wishlist_count': len(wishlist_items)})

@app.route('/api/get-cart')
def get_cart_api():
    return jsonify({'cart': cart_items})

# --- Static File Serving ---
@app.route('/products/<path:filename>')
def serve_product_image(filename):
    image_dir = os.path.join(FASHION_IQ_BASE_PATH, 'Fashion-IQ', 'fashion-iq', 'images')
    return send_from_directory(image_dir, filename)

if __name__ == '__main__':
    if os.name == 'nt': torch.multiprocessing.freeze_support()
    load_model_and_index()
    app.run(host='0.0.0.0', port=5000, debug=False)

