#!/usr/bin/env python3
"""
Model Download Script for Adobe India Hackathon 2025 - Challenge 1B
Downloads all required ML models for offline use in the competition container
"""

import os
from pathlib import Path

def download_models():
    """Download and cache all required models for offline use"""
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("🚀 Downloading models for offline competition environment...")
    
    # 1. Download Sentence Transformers models
    try:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.cross_encoder import CrossEncoder
        
        print("📥 Downloading all-MiniLM-L6-v2 (semantic search)...")
        s_model = SentenceTransformer('all-MiniLM-L6-v2')
        s_model.save(str(models_dir / 'all-MiniLM-L6-v2'))
        print("✅ Semantic model downloaded")
        
        print("📥 Downloading ms-marco-MiniLM-L-6-v2 (cross-encoder)...")
        ce_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        ce_model.save(str(models_dir / 'ms-marco-MiniLM-L-6-v2'))
        print("✅ Cross-encoder model downloaded")
        
    except ImportError:
        print("⚠️ Sentence Transformers not available - models will use fallbacks")
    except Exception as e:
        print(f"⚠️ Error downloading Sentence Transformers models: {e}")
    
    # 2. Download Transformers QA model
    try:
        from transformers import pipeline, AutoTokenizer, AutoModel
        
        print("📥 Downloading distilbert-base-cased-distilled-squad (QA)...")
        qa_pipe = pipeline(
            'question-answering', 
            model='distilbert-base-cased-distilled-squad',
            tokenizer='distilbert-base-cased-distilled-squad'
        )
        qa_pipe.save_pretrained(str(models_dir / 'distilbert-base-cased-distilled-squad'))
        print("✅ QA model downloaded")
        
    except ImportError:
        print("⚠️ Transformers not available - QA will use fallbacks")
    except Exception as e:
        print(f"⚠️ Error downloading Transformers models: {e}")
    
    # 3. Download additional tokenizer if needed
    try:
        from transformers import AutoTokenizer
        print("📥 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')
        tokenizer.save_pretrained(str(models_dir / 'distilbert-base-cased-distilled-squad'))
        print("✅ Tokenizer downloaded")
        
    except Exception as e:
        print(f"⚠️ Error downloading tokenizer: {e}")
    
    print("🏆 All models downloaded and cached for offline use!")
    print(f"📁 Models saved in: {models_dir.absolute()}")
    
    # List downloaded models
    if models_dir.exists():
        print("\n📋 Downloaded models:")
        for model_path in models_dir.iterdir():
            if model_path.is_dir():
                print(f"  ✓ {model_path.name}")

if __name__ == "__main__":
    download_models()
