import numpy as np
import json
import torch
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
EMBEDDINGS_PATH = 'hafez_embeddings.npy'
POEMS_PATH = 'hafez_poems.json'

def load_data():
    try:
        embeddings = np.load(EMBEDDINGS_PATH)
        with open(POEMS_PATH, 'r', encoding='utf-8') as f:
            poems = json.load(f)
        print("✅ Data and embeddings loaded successfully.")
        return embeddings, poems
    except FileNotFoundError:
        print(f"Error: Could not find data files ('{EMBEDDINGS_PATH}' or '{POEMS_PATH}').")
        print("Please run the data processing script first to generate these files.")
        exit()

def find_similar_poems(query, model, embeddings, poems, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    
    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        results.append({
            "poem": poems[idx],
            "score": f"{score:.4f}"
        })
    return results

if __name__ == "__main__":
    print("Loading the semantic search model...")
    model = SentenceTransformer(MODEL_NAME)
    
    poem_embeddings, poems = load_data()
    
    print("\n--- Hafez Semantic Search Engine ---")
    print("Enter your query to find similar verses. Type 'خروج' to exit.")
    
    while True:
        user_query = input("\n> Your query: ")
        
        if user_query.lower() in ['خروج', 'exit']:
            print("Goodbye!")
            break
            
        search_results = find_similar_poems(user_query, model, poem_embeddings, poems)
        
        print("-" * 20)
        for result in search_results:
            print(f"Score: {result['score']} | Verse: {result['poem']}")
        print("-" * 20)
