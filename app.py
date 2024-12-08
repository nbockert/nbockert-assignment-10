from flask import Flask, render_template, request
import os
from sklearn.metrics.pairwise import euclidean_distances

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from clip import clip
import torch
from PIL import Image

# from open_clip import create_model_and_transforms, tokenizer
import torch.nn.functional as F
app = Flask(__name__)

image_folder = 'static/images/coco_images_resized'

def load_image_embeddings():
    df = pd.read_pickle('image_embeddings.pickle')  # Load embeddings from the pickle file
    print("made it here 1")
    # Convert the embeddings column to a numpy array for easier comparison
    image_embeddings = np.vstack(df['embedding'].to_numpy())
    image_names = df['file_name'].to_numpy()
    return image_embeddings, image_names

    

# Route for the home page
@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        return search()

    return render_template('index.html')


def search():
    query_type = request.form.get('query_type')
    print("querytype:",query_type)
    text_query = request.form.get('text_query')
    weight = float(request.form.get('weight', 0.5))
    k = int(request.form.get('num_principle_components'))
    train_pca(image_folder,k)
    uploaded_image = request.files.get('image_query')
    image_path = None
    precomputed_image_embeddings, image_names = load_image_embeddings()
 
    if uploaded_image and uploaded_image.filename != '':
        print("made it here")
        image_path = os.path.join(image_folder, uploaded_image.filename)
        uploaded_image.save(image_path)
     
    # Handle each query type
    if query_type == 'text':
        results = text_query_handler(text_query, precomputed_image_embeddings)
    elif query_type == 'image':
        results = image_query_handler(image_path, precomputed_image_embeddings)
    elif query_type == 'hybrid':
        results = hybrid_query_handler(text_query, image_path, weight, precomputed_image_embeddings)
    else:
        results = []

    # Retrieve top results
    print("number of results",len(results))
    final_results = get_top_results(results, image_names)
    return render_template('results.html', results=final_results)

text_model = SentenceTransformer('all-MiniLM-L6-v2')
clip_model, preprocess = clip.load("ViT-B/32")
def text_query_handler(text_query, image_embeddings):
    """
    Generate embeddings for text query and calculate similarity with image metadata.

    Args:
        text_query (str): User-provided text query.
        image_metadata_embeddings (np.array): Precomputed metadata embeddings.

    Returns:
        list: Sorted list of similarity scores with corresponding indices.
    """
    # Generate text embedding
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
   
    clip_model.eval()
    text = clip.tokenize([text_query]).to(device)
    with torch.no_grad():
 
        text_embedding = clip_model.encode_text(text)
        text_embedding = text_embedding / text_embedding.norm(p=2, dim=-1, keepdim=True)
  
    similarity_scores = cosine_similarity(text_embedding, image_embeddings)
    print(similarity_scores)
    # Return indices of top 5 results
    sorted_indices = np.argsort(-similarity_scores[0])[:5]
    return [(index, similarity_scores[0][index]) for index in sorted_indices]


def image_query_handler(uploaded_image, precomputed_image_embeddings):
    """
    Generate embeddings for the uploaded image and calculate similarity.

    Args:
        uploaded_image (str): Path to the user-uploaded image.
        precomputed_image_embeddings (np.array): Database of precomputed image embeddings.

    Returns:
        list: Sorted list of similarity scores with corresponding indices.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    image = preprocess(Image.open(uploaded_image)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        query_embedding = clip_model.encode_image(image)
        image_embedding = F.normalize(query_embedding, p=2, dim=-1)
    return nearest_neighbors(image_embedding,precomputed_image_embeddings)


def hybrid_query_handler(text_query, uploaded_image, weight, precomputed_image_embeddings):
    """
    Compute a weighted combination of text and image similarity scores.

    Args:
        text_query (str): Text input by the user.
        uploaded_image (str): Path to the uploaded image.
        weight (float): Weight for text similarity (0.0 to 1.0).
        image_metadata_embeddings (np.array): Metadata embeddings.
        precomputed_image_embeddings (np.array): Precomputed image embeddings.

    Returns:
        list: Top 5 results based on combined scores.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    image = preprocess(Image.open(uploaded_image)).unsqueeze(0).to(device)
    # clip_model.eval()
    # text = clip.tokenize([text_query]).to(device)
    with torch.no_grad():
        text_embedding = clip_model.encode_text(clip.tokenize([text_query]).to(device))
        text_embedding = F.normalize(text_embedding, p=2, dim=-1)
        image_embedding = clip_model.encode_image(image)
        image_embedding = F.normalize(image_embedding, p=2, dim=-1)
        # text_embedding = clip_model.encode_text(text)

    query = F.normalize(weight * text_embedding  + (1.0 - weight) * image_embedding, p=2, dim=-1)
    similarity_scores = cosine_similarity(query, precomputed_image_embeddings)
    sorted_indices = np.argsort(-similarity_scores[0])[:5]
    return [(index, similarity_scores[0][index]) for index in sorted_indices]

def get_top_results(results, image_database):
    """
    Retrieve the top results and their scores.

    Args:
        results (list): List of (index, score) tuples.
        image_database (list): List of image file paths.

    Returns:
        list: Top results with paths and scores.
    """
    return [{"image_url": image_database[index], "score": round(score, 4)} for index, score in results]

pca = None
pca_embeddings = None

def load_images(image_dir, max_images=2000, target_size=(224, 224)):
    # Mock implementation to load images
    # Replace with actual image loading and resizing logic
    images = []
    image_names = []
    for i, filename in enumerate(os.listdir(image_dir)):
        if i >= max_images:
            break
        image_path = os.path.join(image_dir, filename)
        image = np.random.rand(*target_size, 3)  # Mock image array
        images.append(image.flatten())
        image_names.append(filename)
    return np.array(images), image_names

def train_pca(image_dir, n_components):
    global pca, pca_embeddings
    # Load images and train PCA
    train_images, train_image_names = load_images(image_dir, max_images=2000, target_size=(224, 224))
    print(f"Loaded {len(train_images)} images for PCA training.")
    pca = PCA(n_components=n_components)
    pca.fit(train_images)
    print(f"Trained PCA on {len(train_images)} samples.")
    pca_embeddings = pca.transform(train_images)  # Precompute PCA embeddings

def nearest_neighbors(query_embedding, embeddings, top_k=5):
    # Reshape query_embedding to (1, n_components) if it's 1D
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # Calculate euclidean distances between query and all embeddings
    distances = euclidean_distances(query_embedding, embeddings).flatten()
    
    # Get indices of top_k smallest distances
    nearest_indices = np.argsort(distances)[:top_k]
    desc_indicies = nearest_indices[::-1]
    return [(index, distances[index]) for index in desc_indicies]

if __name__ == '__main__':
    app.run(debug=True)
