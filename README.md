# Semantic Search in Hafez’s Poetry

This project is an intelligent search engine designed to find **semantically similar verses** in Hafez’s *Divan*.  
Instead of performing word-by-word matching, the system understands the *meaning* of your query and retrieves conceptually related verses.

## ✨ Features

- **Semantic Search:** Understands the meaning of the query instead of relying on keyword matching.  
- **Transformer-based Models:** Uses the `paraphrase-multilingual-MiniLM-L12-v2` model to generate semantic embeddings.  
- **Fast and Optimized:** Caches embeddings for repeated queries to avoid redundant computation.

## 🚀 Installation

To run this project locally, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone github.com/opaip/SearchInHafez.git
    cd SearchInHafez
    ```

2. **Install dependencies:**
    It’s recommended to create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r req.txt
    ```

## 💻 Usage

Run the main script to start the program:
```bash
python search.py
