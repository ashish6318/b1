#!/usr/bin/env python3
"""
Challenge 1B: Persona-Driven Document Intelligence System

This system analyzes document collections to extract and prioritize relevant sections
based on a specific persona and their job-to-be-done. The implementation uses a
five-stage processing pipeline that combines semantic understanding with keyword
matching to deliver precise, contextually relevant results.

Key components:
- Advanced query understanding and expansion
- Hybrid search combining semantic and keyword approaches
- Multi-feature scoring and re-ranking
- Intelligent subsection extraction
- Robust fallback mechanisms for different environments
"""

import json
import os
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Set
import re
from collections import defaultdict, Counter
import math
import numpy as np

# Import from local process_pdfs module
from process_pdfs import AdvancedPDFProcessor

# Configure logging for system monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check availability of optional NLP libraries and set fallback flags
HAS_SENTENCE_TRANSFORMERS = False
HAS_TRANSFORMERS = False
HAS_YAKE = False
HAS_BM25 = False
HAS_SUMY = False

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.cross_encoder import CrossEncoder
    HAS_SENTENCE_TRANSFORMERS = True
    logger.info("Sentence Transformers available - Using advanced semantic processing")
except ImportError:
    logger.warning("Sentence Transformers not available - Using fallback embeddings")

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
    logger.info("Transformers available - Question-answering capabilities enabled")
except ImportError:
    logger.warning("Transformers not available - Using extractive fallback")

try:
    import yake
    HAS_YAKE = True
    logger.info("YAKE available - Advanced keyword extraction enabled")
except ImportError:
    logger.warning("YAKE not available - Using simple keyword extraction")

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
    logger.info("BM25 available - Hybrid search enabled")
except ImportError:
    logger.warning("BM25 not available - Using TF-IDF fallback")

try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer
    HAS_SUMY = True
    logger.info("Sumy available - Advanced summarization enabled")
except ImportError:
    logger.warning("Sumy not available - Using extractive fallback")


class AdvancedQueryProcessor:
    """
    Processes user queries to understand intent and expand search terms.
    
    This class handles the first stage of the pipeline by analyzing the persona
    and job-to-be-done to create a comprehensive query representation that
    captures both explicit and implicit information needs.
    """
    
    def __init__(self):
        self.keyword_extractor = None
        self.setup_keyword_extraction()
        
        # Domain-agnostic synonym database for query expansion
        # These mappings help find relevant content even when different terminology is used
        self.synonym_db = {
            # Analysis and evaluation terms
            'analyze': ['examine', 'evaluate', 'assess', 'study', 'investigate', 'review'],
            'study': ['research', 'investigation', 'analysis', 'examination', 'exploration'],
            'review': ['examine', 'assess', 'evaluate', 'analyze', 'survey'],
            
            # Business and financial terminology
            'revenue': ['income', 'earnings', 'sales', 'turnover', 'profit', 'financial'],
            'performance': ['results', 'outcomes', 'effectiveness', 'efficiency', 'metrics'],
            'trends': ['patterns', 'changes', 'developments', 'movements', 'directions'],
            
            # Academic and research terminology
            'methodology': ['method', 'approach', 'technique', 'procedure', 'framework'],
            'literature': ['publications', 'papers', 'studies', 'articles', 'documents'],
            'benchmarks': ['standards', 'metrics', 'baselines', 'references', 'measures'],
            'findings': ['results', 'conclusions', 'discoveries', 'outcomes', 'insights'],
            
            # Common task-oriented terms
            'prepare': ['create', 'develop', 'generate', 'produce', 'build'],
            'identify': ['find', 'locate', 'discover', 'determine', 'recognize'],
            'summarize': ['overview', 'summary', 'abstract', 'synopsis', 'recap'],
            'compare': ['contrast', 'evaluate', 'analyze', 'assess', 'examine'],
            
            # Content classification terms
            'concepts': ['ideas', 'principles', 'theories', 'notions', 'topics'],
            'information': ['data', 'details', 'facts', 'content', 'material'],
            'examples': ['instances', 'cases', 'samples', 'illustrations', 'demonstrations']
        }
    
    def setup_keyword_extraction(self):
        """Initialize the keyword extraction system based on available libraries"""
        if HAS_YAKE:
            self.keyword_extractor = yake.KeywordExtractor(
                lan="en",
                n=3,  # Extract up to 3-word phrases
                dedupLim=0.7,
                top=20
            )
        else:
            logger.info("Using simple keyword extraction fallback")
    
    def process_query(self, persona: Dict[str, Any], job_to_be_done: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform persona and job-to-be-done into a comprehensive query representation.
        
        This method creates a rich query structure that captures both explicit requirements
        and implicit needs based on the user's role and task.
        """
        
        # Extract basic information from inputs
        persona_text = persona.get('role', '')
        jtbd_text = job_to_be_done.get('task', '')
        base_query = f"As a {persona_text}, I need to {jtbd_text}"
        
        # Extract important keywords and expand them with synonyms
        keywords = self.extract_keywords(jtbd_text)
        expanded_keywords = self.expand_keywords(keywords)
        
        # Generate a hypothetical answer to improve search quality
        hypothetical_answer = self.generate_hypothetical_answer(base_query, expanded_keywords)
        
        # Create a comprehensive query structure for the pipeline
        query_representation = {
            'base_query': base_query,
            'persona': persona_text,
            'task': jtbd_text,
            'keywords': keywords,
            'expanded_keywords': expanded_keywords,
            'hypothetical_answer': hypothetical_answer,
            'search_query': hypothetical_answer if hypothetical_answer else base_query,
            'faceted_questions': self.generate_faceted_questions(jtbd_text)
        }
        
        logger.info(f"Query processed: {len(keywords)} keywords, {len(expanded_keywords)} expanded terms")
        return query_representation
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from the given text using available methods"""
        if HAS_YAKE and self.keyword_extractor:
            keywords = self.keyword_extractor.extract_keywords(text)
            # YAKE returns (score, keyword) tuples - extract just the keywords
            return [str(kw[1]) for kw in keywords[:10]]  # Take top 10 keywords
        else:
            # Simple fallback: extract meaningful words
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            # Remove common stop words that don't add search value
            stop_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'would', 'there', 'could', 'other'}
            return [w for w in words if w not in stop_words][:10]
    
    def expand_keywords(self, keywords: List[str]) -> List[str]:
        """Expand the keyword list with synonyms and related terms to improve search coverage"""
        expanded = set(keywords)
        
        for keyword in keywords:
            # Handle different keyword formats (tuples, strings, numpy types)
            if isinstance(keyword, (tuple, list)):
                keyword = str(keyword[0])  # Extract keyword from tuple
            else:
                keyword = str(keyword)  # Convert to string for consistency
            
            # Add synonyms from our predefined database
            if keyword.lower() in self.synonym_db:
                expanded.update(self.synonym_db[keyword.lower()])
            
            # Add stemmed/related forms
            if keyword.endswith('ing'):
                expanded.add(keyword[:-3])  # remove 'ing'
            if keyword.endswith('ed'):
                expanded.add(keyword[:-2])  # remove 'ed'
            if keyword.endswith('s'):
                expanded.add(keyword[:-1])  # remove 's'
        
        return list(expanded)
    
    def generate_hypothetical_answer(self, query: str, keywords: List[str]) -> str:
        """Generate hypothetical answer using HyDE technique - Universal templates"""
        # Create domain-agnostic templates that work for any content type
        
        # Universal task patterns that work across all domains
        task_patterns = {
            'literature review': "This comprehensive review examines {keywords}. The analysis covers methodologies, findings, and key insights from available sources.",
            'review': "The review of {keywords} provides detailed analysis. Key aspects include methodology, results, and comparative insights.",
            'analyze': "Analysis of {keywords} reveals important patterns and relationships. The examination shows trends, metrics, and outcomes.",
            'compare': "Comparison of {keywords} highlights similarities and differences. Key factors include performance, characteristics, and effectiveness.",
            'prepare': "Preparation involves understanding {keywords}. Essential elements include planning, methodology, and implementation strategies.",
            'study': "The study of {keywords} provides comprehensive insights. Research covers methodology, analysis, and conclusions.",
            'plan': "Planning for {keywords} requires detailed consideration. Key components include objectives, resources, and implementation strategies.",
            'identify': "Identification of {keywords} involves systematic analysis. Important factors include characteristics, criteria, and selection methods.",
            'summarize': "Summary of {keywords} covers main points and key insights. Essential information includes highlights, conclusions, and recommendations.",
            'examine': "Examination of {keywords} provides detailed investigation. Analysis includes methodology, findings, and implications.",
            'evaluate': "Evaluation of {keywords} assesses effectiveness and quality. Criteria include performance, outcomes, and comparative analysis.",
            'research': "Research on {keywords} involves systematic investigation. Methods include data collection, analysis, and interpretation of results."
        }
        
        # Determine task type and generate appropriate hypothesis
        query_lower = query.lower()
        for pattern, template in task_patterns.items():
            if pattern in query_lower:
                # Fill template with actual keywords (domain-agnostic)
                keyword_str = ', '.join(keywords[:3]) if keywords else 'the subject matter'
                hypothesis = template.format(keywords=keyword_str)
                return hypothesis
        
        # Default hypothesis if no pattern matches
        if keywords:
            keyword_str = ', '.join(keywords[:3])
            return f"This document provides comprehensive information about {keyword_str}. The content covers key aspects, methodology, and detailed analysis relevant to the topic."
        else:
            return "This document contains relevant information and analysis that addresses the specified requirements and objectives."
        
        # Default hypothesis if no pattern matches
        if keywords:
            keyword_str = ', '.join(keywords[:3])
            return f"This document provides comprehensive information about {keyword_str}. The content covers key aspects, methodology, and detailed analysis relevant to the topic."
        else:
            return "This document contains relevant information and analysis that addresses the specified requirements and objectives."
    
    def generate_faceted_questions(self, jtbd_text: str) -> List[str]:
        """Generate multiple specific questions from JTBD for iterative QA"""
        questions = []
        
        # Universal question patterns that work across all domains
        if any(word in jtbd_text.lower() for word in ['review', 'literature', 'survey']):
            questions.extend([
                "What are the main topics covered?",
                "What methodologies are discussed?",
                "What are the key findings or results?",
                "What conclusions are presented?"
            ])
        elif any(word in jtbd_text.lower() for word in ['analyze', 'analysis', 'examine']):
            questions.extend([
                "What are the main trends?",
                "What are the key metrics?",
                "What factors influence performance?",
                "What are the conclusions?"
            ])
        elif any(word in jtbd_text.lower() for word in ['compare', 'comparison', 'contrast']):
            questions.extend([
                "What are the main differences?",
                "What are the similarities?", 
                "What are the advantages and disadvantages?",
                "What are the key characteristics?"
            ])
        elif any(word in jtbd_text.lower() for word in ['prepare', 'plan', 'create', 'develop']):
            questions.extend([
                "What are the essential components?",
                "What steps are required?",
                "What resources are needed?",
                "What are the best practices?"
            ])
        elif any(word in jtbd_text.lower() for word in ['study', 'learn', 'understand']):
            questions.extend([
                "What are the key concepts?",
                "What are the main principles?",
                "What are the important details?",
                "What should be remembered?"
            ])
        
        # Always add generic questions if no specific ones match
        if not questions:
            questions = [
                "What are the main topics?",
                "What are the key points?",
                "What information is provided?",
                "What are the important details?"
            ]
        
        return questions[:4]  # Limit to top 4 questions


class EnhancedDocumentParser:
    """Stage 2: Advanced PDF parsing with table/figure extraction"""
    
    def __init__(self):
        self.pdf_processor = AdvancedPDFProcessor()
    
    def parse_document(self, pdf_path: Path, filename: str) -> List[Dict]:
        """Parse PDF with enhanced structural analysis"""
        try:
            # Use advanced PDF processor with path injection
            pdf_data = self.pdf_processor.process_pdf(str(pdf_path))
            
            # Inject PDF path for complete text extraction between headings
            pdf_data['pdf_path'] = str(pdf_path)
            
            # Extract sections with enhanced metadata
            sections = self.build_enhanced_sections(pdf_data, filename)
            
            # Post-process for quality improvements
            sections = self.enhance_sections(sections)
            
            logger.info(f"Enhanced parsing: {len(sections)} sections from {filename}")
            return sections
            
        except Exception as e:
            logger.error(f"Error parsing {pdf_path}: {e}")
            return []
    
    def build_enhanced_sections(self, pdf_data: Dict, filename: str) -> List[Dict]:
        """Builds sections with precise coordinate-based text extraction."""
        import fitz
        outline = pdf_data.get('outline', [])
        pdf_path = pdf_data.get('pdf_path')
        if not outline or not pdf_path: 
            return self.build_enhanced_sections_fallback(pdf_data, filename)

        doc = fitz.open(pdf_path)
        sections = []
        
        # Add end marker for final section processing
        end_marker = {'page': len(doc) + 1, 'bbox': (0, 0, 0, 0)}
        outline_with_end = outline + [end_marker]

        for i, heading in enumerate(outline):
            start_page = heading.get('page', 1)
            start_bbox = heading.get('bbox')
            if not start_bbox: 
                continue  # Skip headings without bbox data

            next_heading = outline_with_end[i+1]
            end_page = next_heading.get('page', len(doc))
            end_bbox = next_heading.get('bbox')
            
            content_text = ""
            for page_num in range(start_page, end_page + 1):
                if page_num > len(doc): 
                    continue
                page = doc.load_page(page_num - 1)
                
                # Calculate precise text extraction coordinates
                start_y = start_bbox[3] if page_num == start_page else 0
                end_y = end_bbox[1] if page_num == end_page and end_bbox else page.rect.height
                
                # Extract text within the coordinate bounds
                clip_rect = fitz.Rect(0, start_y, page.rect.width, end_y)
                if clip_rect.is_valid and clip_rect.height > 5:
                    content_text += page.get_text(clip=clip_rect).strip() + "\n"

            # Combine heading and content
            full_content = f"{heading['text']}\n\n{content_text.strip()}"
            if len(content_text.strip()) > 20:  # Ensure substantial content
                sections.append({
                    'document': filename, 
                    'section_title': heading['text'],
                    'content': full_content, 
                    'page_number': start_page,
                    'content_type': self.classify_content_type(heading['text']),
                    'quality_score': 0.9  # High quality due to precise extraction
                })

        doc.close()
        return sections
    
    def build_enhanced_sections_fallback(self, pdf_data: Dict, filename: str) -> List[Dict]:
        """Fallback method when PyMuPDF extraction fails"""
        sections = []
        outline = pdf_data.get('outline', [])
        
        for i, heading in enumerate(outline):
            heading_text = heading.get('text', '').strip()
            page_num = heading.get('page', 1)
            
            # Use heading text plus some context from outline
            content_lines = [heading_text]
            for j in range(i + 1, min(i + 3, len(outline))):
                next_heading = outline[j]
                if next_heading.get('level') not in ['H1', 'H2']:
                    content_lines.append(next_heading.get('text', ''))
            
            content = '\n'.join(content_lines)
            
            if len(content) > 30:
                section = {
                    'document': filename,
                    'section_title': heading_text,
                    'content': content,
                    'page_number': page_num,
                    'content_type': self.classify_content_type(heading_text),
                    'quality_score': 0.5
                }
                sections.append(section)
        
        return sections
    
    def classify_content_type(self, text: str) -> str:
        """Classify content type for enhanced processing"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['table', 'figure', 'chart', 'graph']):
            return 'visual'
        elif any(word in text_lower for word in ['method', 'approach', 'procedure', 'algorithm']):
            return 'methodological'
        elif any(word in text_lower for word in ['result', 'finding', 'outcome', 'conclusion']):
            return 'results'
        elif any(word in text_lower for word in ['introduction', 'overview', 'background']):
            return 'background'
        elif any(word in text_lower for word in ['discussion', 'analysis', 'interpretation']):
            return 'analysis'
        else:
            return 'general'
    
    def calculate_quality_score(self, section: Dict) -> float:
        """Calculate content quality score"""
        content = section['content']
        
        score = 0.5  # Base score
        
        # Length appropriateness
        length = len(content.split())
        if 50 <= length <= 300:
            score += 0.2
        elif length > 300:
            score += 0.1
        elif length < 20:
            score -= 0.2
        
        # Content type bonus
        content_type = section.get('content_type', 'general')
        type_bonuses = {
            'methodological': 0.2,
            'results': 0.2,
            'analysis': 0.15,
            'visual': 0.1,
            'background': 0.05,
            'general': 0.0
        }
        score += type_bonuses.get(content_type, 0.0)
        
        # Structure indicators
        if any(indicator in content.lower() for indicator in ['first', 'second', 'third', 'steps']):
            score += 0.1
        
        return min(max(score, 0.0), 1.0)
    
    def enhance_sections(self, sections: List[Dict]) -> List[Dict]:
        """Post-process sections for quality enhancement"""
        # Filter out very short sections
        enhanced = [s for s in sections if len(s['content']) > 30]
        
        # Merge very similar adjacent sections
        if len(enhanced) > 1:
            merged = [enhanced[0]]
            for section in enhanced[1:]:
                last_section = merged[-1]
                
                # Check similarity of titles
                title_similarity = self.simple_similarity(
                    last_section['section_title'], 
                    section['section_title']
                )
                
                if (title_similarity > 0.7 and 
                    last_section['page_number'] == section['page_number']):
                    # Merge with previous section
                    last_section['content'] += f"\n\n{section['content']}"
                    last_section['quality_score'] = max(
                        last_section['quality_score'], 
                        section['quality_score']
                    )
                else:
                    merged.append(section)
            
            enhanced = merged
        
        return enhanced
    
    def simple_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity based on word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0


class HybridSearchEngine:
    """Stage 3: Hybrid semantic + keyword search with ranking fusion"""
    
    def __init__(self):
        self.semantic_model = None
        self.bm25_index = None
        self.setup_models()
    
    def setup_models(self):
        """Initialize semantic and keyword search models"""
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                # Try loading from local cache first (for competition environment)
                model_path = Path(__file__).parent / "models" / "all-MiniLM-L6-v2"
                if model_path.exists():
                    self.semantic_model = SentenceTransformer(str(model_path))
                    logger.info("✓ Semantic search model loaded from cache")
                else:
                    # Fallback to online download (for development)
                    self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                    logger.info("✓ Semantic search model loaded from online")
            except Exception as e:
                logger.warning(f"Failed to load semantic model: {e}")
                self.semantic_model = None
        
        logger.info("Hybrid search engine initialized")
    
    def build_search_index(self, sections: List[Dict]):
        """Build search indices for all sections"""
        # Prepare corpus for BM25
        corpus = []
        for section in sections:
            # Combine title and content for search
            text = f"{section['section_title']} {section['content']}"
            corpus.append(text.lower().split())
        
        # Build BM25 index
        if HAS_BM25 and corpus:
            self.bm25_index = BM25Okapi(corpus)
            logger.info(f"✓ BM25 index built with {len(corpus)} documents")
        else:
            logger.info("Using TF-IDF fallback for keyword search")
    
    def search(self, query_representation: Dict, sections: List[Dict], top_k: int = 25) -> List[Tuple[Dict, float]]:
        """Perform hybrid search and return ranked candidates"""
        if not sections:
            return []
        
        # Ensure top_k doesn't exceed available sections
        top_k = min(top_k, len(sections))
        
        # Always rebuild the index for each new collection
        self.build_search_index(sections)
        
        # Get search query
        search_query = query_representation['search_query']
        keywords = query_representation['expanded_keywords']
        
        # Stage 3a: Semantic search
        semantic_scores = self.semantic_search(search_query, sections)
        
        # Stage 3b: Keyword search
        keyword_scores = self.keyword_search(keywords, sections)
        
        # Stage 3c: Reciprocal Rank Fusion
        fused_scores = self.reciprocal_rank_fusion(semantic_scores, keyword_scores)
        
        # Return top candidates with bounds checking
        candidates = []
        for i, score in fused_scores[:top_k]:
            if 0 <= i < len(sections):
                candidates.append((sections[i], score))
            else:
                logger.warning(f"Index {i} out of bounds for sections list (length: {len(sections)})")
        
        logger.info(f"Hybrid search: {len(candidates)} candidates from {len(sections)} sections")
        return candidates
    
    def semantic_search(self, query: str, sections: List[Dict]) -> List[Tuple[int, float]]:
        """Perform semantic search using sentence transformers"""
        if not self.semantic_model:
            # Fallback: simple word overlap
            return self.simple_semantic_search(query, sections)
        
        try:
            # Encode query
            query_embedding = self.semantic_model.encode([query])
            
            # Encode all sections
            section_texts = []
            for section in sections:
                text = f"{section['section_title']} {section['content']}"
                section_texts.append(text)
            
            section_embeddings = self.semantic_model.encode(section_texts)
            
            # Calculate similarities using optimized cos_sim
            from sentence_transformers.util import cos_sim
            similarities = cos_sim(query_embedding, section_embeddings)[0]
            
            # Return ranked indices with scores
            ranked = [(i, float(score)) for i, score in enumerate(similarities)]
            ranked.sort(key=lambda x: x[1], reverse=True)
            
            return ranked
            
        except Exception as e:
            logger.warning(f"Semantic search error: {e}, using fallback")
            return self.simple_semantic_search(query, sections)
    
    def simple_semantic_search(self, query: str, sections: List[Dict]) -> List[Tuple[int, float]]:
        """Fallback semantic search using word overlap"""
        query_words = set(query.lower().split())
        
        scores = []
        for i, section in enumerate(sections):
            text = f"{section['section_title']} {section['content']}"
            section_words = set(text.lower().split())
            
            if not section_words:
                score = 0.0
            else:
                # Jaccard similarity
                intersection = query_words.intersection(section_words)
                union = query_words.union(section_words)
                score = len(intersection) / len(union) if union else 0.0
            
            scores.append((i, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def keyword_search(self, keywords: List[str], sections: List[Dict]) -> List[Tuple[int, float]]:
        """Perform keyword search using BM25 or TF-IDF fallback"""
        if HAS_BM25 and self.bm25_index:
            # Use BM25
            query_tokens = []
            for keyword in keywords:
                query_tokens.extend(keyword.lower().split())
            
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Validate that scores match sections count
            if len(scores) != len(sections):
                logger.warning(f"BM25 scores length ({len(scores)}) != sections length ({len(sections)}), using fallback")
                return self.tfidf_search(keywords, sections)
            
            ranked = [(i, float(score)) for i, score in enumerate(scores)]
            ranked.sort(key=lambda x: x[1], reverse=True)
            
            return ranked
        else:
            # TF-IDF fallback
            return self.tfidf_search(keywords, sections)
    
    def tfidf_search(self, keywords: List[str], sections: List[Dict]) -> List[Tuple[int, float]]:
        """Simple TF-IDF based keyword search"""
        keyword_set = set(kw.lower() for kw in keywords)
        
        scores = []
        for i, section in enumerate(sections):
            text = f"{section['section_title']} {section['content']}"
            words = text.lower().split()
            
            if not words:
                score = 0.0
            else:
                # Simple keyword density
                matches = sum(1 for word in words if word in keyword_set)
                score = matches / len(words)
            
            scores.append((i, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def reciprocal_rank_fusion(self, semantic_scores: List[Tuple[int, float]], 
                             keyword_scores: List[Tuple[int, float]], k: int = 60) -> List[Tuple[int, float]]:
        """Combine rankings using Reciprocal Rank Fusion"""
        
        # Create rank mappings
        semantic_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(semantic_scores)}
        keyword_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(keyword_scores)}
        
        # Calculate RRF scores
        all_indices = set(semantic_ranks.keys()).union(set(keyword_ranks.keys()))
        
        rrf_scores = []
        for idx in all_indices:
            semantic_rank = semantic_ranks.get(idx, len(semantic_scores) + 1)
            keyword_rank = keyword_ranks.get(idx, len(keyword_scores) + 1)
            
            # RRF formula
            rrf_score = (1 / (k + semantic_rank)) + (1 / (k + keyword_rank))
            rrf_scores.append((idx, rrf_score))
        
        # Sort by RRF score
        rrf_scores.sort(key=lambda x: x[1], reverse=True)
        
        return rrf_scores


class AdvancedReRanker:
    """Stage 4: Precision re-ranking with cross-encoder and multi-feature scoring"""
    
    def __init__(self):
        self.cross_encoder = None
        self.setup_models()
    
    def setup_models(self):
        """Initialize cross-encoder model"""
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                # Try loading from local cache first (for competition environment)
                model_path = Path(__file__).parent / "models" / "ms-marco-MiniLM-L-6-v2"
                if model_path.exists():
                    self.cross_encoder = CrossEncoder(str(model_path))
                    logger.info("✓ Cross-encoder model loaded from cache")
                else:
                    # Fallback to online download (for development)
                    self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                    logger.info("✓ Cross-encoder model loaded from online")
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder: {e}")
                self.cross_encoder = None
    
    def rerank_candidates(self, query_representation: Dict, candidates: List[Tuple[Dict, float]]) -> List[Dict]:
        """Perform precision re-ranking with multi-feature scoring"""
        
        if not candidates:
            return []
        
        # Extract query information
        search_query = query_representation['search_query']
        persona = query_representation['persona']
        keywords = query_representation['expanded_keywords']
        
        # Calculate multiple features for each candidate
        scored_candidates = []
        
        for section, initial_score in candidates:
            features = self.extract_features(section, query_representation, initial_score)
            final_score = self.calculate_final_score(features)
            
            section_with_score = section.copy()
            section_with_score['relevance_score'] = final_score
            section_with_score['feature_breakdown'] = features
            
            scored_candidates.append(section_with_score)
        
        # Sort by final score
        scored_candidates.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # REFINEMENT 4: Apply section diversity to prevent duplicate/generic content
        diversified_candidates = self.apply_section_diversity(scored_candidates)
        
        logger.info(f"Re-ranking complete: top score = {diversified_candidates[0]['relevance_score']:.3f}")
        return diversified_candidates
    
    def extract_features(self, section: Dict, query_representation: Dict, initial_score: float) -> Dict:
        """Extract multiple features for scoring"""
        
        content = section['content']
        title = section['section_title']
        
        features = {
            'initial_score': initial_score,
            'cross_encoder_score': 0.0,
            'keyword_density': 0.0,
            'content_quality': section.get('quality_score', 0.5),
            'title_relevance': 0.0,
            'length_score': 0.0,
            'content_type_bonus': 0.0,
            'persona_relevance': 0.0,  # REFINEMENT 2: Persona-driven scoring
            'specificity_score': 0.0,  # REFINEMENT 2: Specificity assessment
            'generic_penalty': 0.0     # REFINEMENT 2: Generic content penalty
        }
        
        # Cross-encoder score (most important)
        if self.cross_encoder:
            try:
                query_text = query_representation['search_query']
                section_text = f"{title} {content}"
                # Fix: Cross-encoder expects list of pairs, even for single prediction
                cross_score = float(self.cross_encoder.predict([(query_text, section_text)])[0])
                features['cross_encoder_score'] = cross_score
            except Exception as e:
                logger.warning(f"Cross-encoder error: {e}")
                features['cross_encoder_score'] = initial_score
        else:
            features['cross_encoder_score'] = initial_score
        
        # Keyword density
        keywords = query_representation['expanded_keywords']
        if keywords:
            text_words = content.lower().split()
            keyword_matches = sum(1 for word in text_words if word in [kw.lower() for kw in keywords])
            features['keyword_density'] = keyword_matches / len(text_words) if text_words else 0.0
        
        # Title relevance
        query_words = set(query_representation['search_query'].lower().split())
        title_words = set(title.lower().split())
        if title_words:
            title_overlap = len(query_words.intersection(title_words)) / len(title_words)
            features['title_relevance'] = title_overlap
        
        # Length appropriateness
        word_count = len(content.split())
        if 100 <= word_count <= 400:
            features['length_score'] = 1.0
        elif 50 <= word_count <= 600:
            features['length_score'] = 0.8
        elif word_count >= 30:
            features['length_score'] = 0.6
        else:
            features['length_score'] = 0.3
        
        # Content type bonus
        content_type = section.get('content_type', 'general')
        type_bonuses = {
            'methodological': 0.2,
            'results': 0.2,
            'analysis': 0.15,
            'visual': 0.1,
            'background': 0.05,
            'general': 0.0
        }
        features['content_type_bonus'] = type_bonuses.get(content_type, 0.0)
        
        # REFINEMENT 2: Dynamic persona-driven relevance scoring
        persona = query_representation.get('persona', '')
        task = query_representation.get('task', '')
        combined_text = f"{title} {content}".lower()

        # Use the original, unexpanded keywords from the task for specificity
        task_keywords = query_representation.get('keywords', [])
        
        if task_keywords:
            persona_matches = sum(1 for keyword in task_keywords if keyword.lower() in combined_text)
            features['persona_relevance'] = min(1.0, persona_matches / len(task_keywords) * 2.0) # Boost score
        else:
            features['persona_relevance'] = 0.0
        
        # Specificity assessment (favor specific over generic content)
        specific_indicators = ['specific', 'detailed', 'step-by-step', 'practical', 'actionable', 'concrete', 'examples', 'case study']
        specificity_matches = sum(1 for indicator in specific_indicators if indicator in combined_text)
        features['specificity_score'] = min(specificity_matches / len(specific_indicators), 1.0)
        
        # Generic content penalty (penalize overly generic sections)
        generic_indicators = ['general', 'overview', 'introduction', 'conclusion', 'summary', 'basic information']
        generic_matches = sum(1 for indicator in generic_indicators if indicator in combined_text)
        features['generic_penalty'] = -min(generic_matches / len(generic_indicators), 0.5) # Negative penalty
        
        return features
    
    def calculate_final_score(self, features: Dict) -> float:
        """Calculate weighted final score from multiple features"""
        
        # REFINEMENT 2: Updated weights for persona-driven relevance
        weights = {
            'cross_encoder_score': 0.35,    # Slightly reduced for persona balance
            'initial_score': 0.18,          # Hybrid search foundation
            'persona_relevance': 0.15,      # REFINEMENT 2: Persona-specific relevance
            'keyword_density': 0.12,        # Keyword relevance
            'specificity_score': 0.08,      # REFINEMENT 2: Favor specific content
            'title_relevance': 0.07,        # Title matching
            'content_quality': 0.03,        # Content quality
            'length_score': 0.02,           # Length appropriateness
            'content_type_bonus': 0.02,     # Content type bonus
            'generic_penalty': -0.02        # REFINEMENT 2: Penalize generic content
        }
        
        final_score = 0.0
        for feature, weight in weights.items():
            feature_value = features.get(feature, 0.0)
            final_score += weight * feature_value
        
        # Ensure score is in [0, 1] range
        return min(max(final_score, 0.0), 1.0)
    
    def apply_section_diversity(self, scored_candidates: List[Dict]) -> List[Dict]:
        """REFINEMENT 4: Apply diversity filtering to prevent duplicate/generic sections"""
        if len(scored_candidates) <= 10:
            return scored_candidates  # No need for diversity if few candidates
        
        diverse_sections = []
        seen_titles = set()
        title_similarity_threshold = 0.7
        
        for candidate in scored_candidates:
            title = candidate['section_title'].lower().strip()
            
            # Skip exact duplicate titles
            if title in seen_titles:
                continue
            
            # Check for very similar titles (prevent multiple "Introduction", "Conclusion" etc.)
            is_too_similar = False
            for seen_title in seen_titles:
                similarity = self.calculate_title_similarity(title, seen_title)
                if similarity > title_similarity_threshold:
                    is_too_similar = True
                    break
            
            if is_too_similar:
                continue
            
            # Skip overly generic sections if we already have enough content
            if len(diverse_sections) >= 5:  # After we have 5 good sections
                generic_indicators = ['introduction', 'conclusion', 'overview', 'summary', 'general']
                if any(indicator in title for indicator in generic_indicators):
                    continue
            
            # Add section to diverse set
            diverse_sections.append(candidate)
            seen_titles.add(title)
            
            # Stop when we have enough diverse sections
            if len(diverse_sections) >= 15:  # Limit to 15 diverse sections
                break
        
        return diverse_sections
    
    def calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles (simple word overlap method)"""
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0


class AdvancedSubsectionAnalyzer:
    """Final Stage 5: QA-first subsection analysis with a robust keyword-based fallback."""
    
    def __init__(self):
        self.qa_pipeline = None
        self.setup_models()
    
    def setup_models(self):
        """Initialize models from local cache."""
        if HAS_TRANSFORMERS:
            try:
                model_path = Path(__file__).parent / "models" / "distilbert-base-cased-distilled-squad"
                if model_path.exists():
                    self.qa_pipeline = pipeline('question-answering', model=str(model_path), tokenizer=str(model_path))
                else:
                     logger.error("❌ QA Model not found. Please run download_models.py")
            except Exception as e:
                logger.error(f"Failed to load QA pipeline: {e}")
    
    def clean_text_output(self, text: str) -> str:
        """Clean text by removing unwanted newlines and formatting issues"""
        if not text:
            return text
        
        # Replace multiple newlines with single space
        cleaned = re.sub(r'\n+', ' ', text)
        
        # Replace multiple spaces with single space
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Strip leading/trailing whitespace
        cleaned = cleaned.strip()
        
        return cleaned
    
    def analyze_subsections(self, top_sections: List[Dict], query_representation: Dict) -> List[Dict]:
        """Prioritizes QA for concise snippets, with a robust keyword fallback."""
        subsections = []
        
        # 1. Primary Strategy: Use QA on top-ranked sections
        if self.qa_pipeline:
            faceted_questions = query_representation['faceted_questions']
            for section in top_sections[:5]:
                qa_subsections = self.iterative_qa_extraction(section, faceted_questions)
                subsections.extend(qa_subsections)

        # 2. Robust Fallback: If QA fails or finds nothing, use keyword scoring
        if not subsections:
            logger.info("QA returned no results, falling back to keyword sentence extraction.")
            for section in top_sections[:5]:
                keyword_subsections = self.keyword_sentence_extraction(section, query_representation)
                subsections.extend(keyword_subsections)
                
        subsections.sort(key=lambda x: x.get('relevance_score', 0.0), reverse=True)
        return subsections[:15] # Return the top 15 snippets overall
    
    def semantic_sentence_extraction(self, section: Dict, query: str) -> List[Dict]:
        """
        Extracts the most relevant sentences using semantic similarity.
        This is the robust primary method for subsection analysis.
        """
        if not self.semantic_model:
            logger.warning("Semantic model not available for subsection extraction")
            return []
        
        content = section['content']
        document = section['document']
        page_number = section.get('page_number', 1)
        
        # Split content into sentences using robust regex
        sentences = re.split(r'(?<=[.!?])\s+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]  # Filter short sentences
        
        if not sentences:
            logger.warning(f"No valid sentences found in {document}")
            return []
        
        try:
            # Generate embeddings for query and all sentences
            query_embedding = self.semantic_model.encode(query)
            sentence_embeddings = self.semantic_model.encode(sentences)
            
            # Validate embeddings were generated
            if sentence_embeddings is None or len(sentence_embeddings) == 0:
                logger.warning(f"No sentence embeddings generated for {document}")
                return []
            
            # Calculate cosine similarity
            from sentence_transformers.util import cos_sim
            similarities = cos_sim(query_embedding, sentence_embeddings)[0]
            
            # Create subsections from most similar sentences
            subsections = []
            
            # Validate we have similarities before processing
            if len(similarities) == 0:
                logger.warning(f"No similarities calculated for {document}")
                return []
            
            # Get indices of top 3 sentences (or fewer if we have less than 3)
            num_sentences = min(3, len(similarities))
            if num_sentences == 0:
                logger.warning(f"No sentences to process for {document}")
                return []
                
            top_indices = np.argsort(similarities)[-num_sentences:][::-1]
            
            for rank, i in enumerate(top_indices):
                score = float(similarities[i])
                if score > 0.3:  # Confidence threshold
                    refined_text = sentences[i]
                    
                    # Expand context if sentence is too short
                    if len(refined_text) < 100 and i > 0:
                        # Add previous sentence for context
                        refined_text = f"{sentences[i-1]} {refined_text}"
                    
                    # Clean the text output
                    refined_text = self.clean_text_output(refined_text)
                    
                    subsection = {
                        'document': document,
                        'refined_text': refined_text,
                        'page_number': page_number,
                        'extraction_method': 'semantic_sentence_search',
                        'relevance_score': score,
                        'rank': rank + 1
                    }
                    subsections.append(subsection)
            
            logger.info(f"🎯 Semantic extraction: {len(subsections)} subsections from {document}")
            return subsections
            
        except Exception as e:
            logger.warning(f"Semantic sentence extraction failed: {e}")
            return []
    
    def extract_subsections_fallback(self, section: Dict, query_representation: Dict) -> List[Dict]:
        """Fallback method: Robust subsection extraction using keyword matching and content analysis"""
        try:
            content = section['content']
            if len(content) < 100:  # Too short for meaningful subsections
                return []
            
            # Extract key terms from query
            query_keywords = set()
            if 'keywords' in query_representation:
                query_keywords.update(query_representation['keywords'])
            if 'expanded_terms' in query_representation:
                query_keywords.update(query_representation['expanded_terms'])
            
            # REFINEMENT 3: Optimize for short, actionable snippets
            # Split content into sentences first for better control
            sentences = re.split(r'(?<=[.!?])\s+', content)
            
            # Create short chunks (1-2 sentences, max 150 words)
            chunks = []
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if len(sentence) < 30:  # Too short
                    continue
                    
                # Single sentence if long enough (50-150 words)
                word_count = len(sentence.split())
                if 50 <= word_count <= 150:
                    chunks.append(sentence)
                elif word_count < 50 and i + 1 < len(sentences):
                    # Combine with next sentence if both are short
                    next_sentence = sentences[i + 1].strip()
                    combined = f"{sentence} {next_sentence}"
                    if len(combined.split()) <= 150:
                        chunks.append(combined)
                    else:
                        chunks.append(sentence)  # Use just first sentence
                elif word_count > 150:
                    # Truncate long sentences to key portion
                    words = sentence.split()
                    # Find the most relevant part with keywords
                    best_start = 0
                    best_score = 0
                    
                    for start in range(0, len(words) - 50, 25):  # Check every 25 words
                        snippet_words = words[start:start + 100]  # 100-word window
                        snippet = ' '.join(snippet_words)
                        snippet_lower = snippet.lower()
                        
                        score = sum(1 for kw in query_keywords if kw.lower() in snippet_lower)
                        if score > best_score:
                            best_score = score
                            best_start = start
                    
                    # Extract best 100-word snippet
                    best_snippet = ' '.join(words[best_start:best_start + 100])
                    chunks.append(best_snippet + "..." if best_start + 100 < len(words) else best_snippet)
            
            # Score chunks based on relevance and actionability
            scored_chunks = []
            for chunk in chunks[:15]:  # Limit processing
                score = 0
                chunk_lower = chunk.lower()
                
                # Keyword matching (higher weight)
                for keyword in query_keywords:
                    if keyword.lower() in chunk_lower:
                        score += 2.0
                
                # REFINEMENT 3: Dynamic actionable content scoring
                # Extract action words from job description for any domain
                job_description = query_representation.get('job_to_be_done', '').lower()
                if job_description:
                    # Common actionable verbs across all domains
                    action_verbs = ['prepare', 'analyze', 'create', 'develop', 'identify', 'evaluate', 'assess', 'compare', 'review', 'examine', 'study', 'research', 'plan', 'organize', 'implement', 'design', 'build', 'solve']
                    
                    # Look for action-oriented content
                    for verb in action_verbs:
                        if verb in chunk_lower:
                            score += 1.0
                    
                    # Look for job-specific keywords in content
                    job_keywords = [word for word in job_description.split() if len(word) > 3][:5]
                    for keyword in job_keywords:
                        if keyword in chunk_lower:
                            score += 1.5
                
                # Boost for specific patterns
                if re.search(r'\d+\.|\•|\-\s|step\s\d+', chunk):
                    score += 1.0
                
                # Boost for short, focused content
                word_count = len(chunk.split())
                if 30 <= word_count <= 100:
                    score += 0.5
                elif word_count <= 30:
                    score += 0.2
                
                if score > 0:
                    scored_chunks.append((chunk, score))
            
            # Sort by score and select top subsections
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            
            subsections = []
            for i, (chunk, score) in enumerate(scored_chunks[:3]):  # REFINEMENT 3: Reduced to top 3 for focus
                if score >= 1.0:  # Higher relevance threshold
                    # REFINEMENT 3: Ensure snippet length is optimal (50-150 words)
                    words = chunk.split()
                    if len(words) > 150:
                        # Truncate to 150 words, ending at sentence boundary
                        truncated = ' '.join(words[:150])
                        last_period = truncated.rfind('.')
                        if last_period > len(truncated) * 0.8:  # If period is near end
                            chunk = truncated[:last_period + 1]
                        else:
                            chunk = truncated + "..."
                    
                    subsection = {
                        "document": section['document'],
                        "page_number": section.get('page_number', 1),
                        "subsection_text": chunk,
                        "relevance_score": min(score / 5.0, 1.0),  # Normalize score
                        "rank": i + 1,
                        "extraction_method": "optimized_snippet"
                    }
                    subsections.append(subsection)
            
            logger.info(f"🎯 Fallback extraction: {len(subsections)} subsections from {section['document']}")
            return subsections
            
        except Exception as e:
            logger.warning(f"Error in fallback subsection extraction: {e}")
            return []
    
    def extract_section_subsections(self, section: Dict, questions: List[str], 
                                  query_representation: Dict) -> List[Dict]:
        """Extract subsections from a single section using iterative QA"""
        
        content = section['content']
        document = section['document']
        page_number = section.get('page_number', 1)
        
        subsections = []
        
        # Method 1: Iterative QA approach
        if self.qa_pipeline and questions:
            qa_subsections = self.iterative_qa_extraction(
                content, questions, document, page_number
            )
            subsections.extend(qa_subsections)
        
        # Method 2: Extractive summarization
        summary_subsections = self.extractive_summarization(
            content, document, page_number
        )
        subsections.extend(summary_subsections)
        
        # Method 3: Structural splitting (fallback)
        if not subsections:
            structural_subsections = self.structural_splitting(
                content, document, page_number
            )
            subsections.extend(structural_subsections)
        
        return subsections
    
    def iterative_qa_extraction(self, section: Dict, questions: List[str]) -> List[Dict]:
        """Extracts short, direct answers using the QA model."""
        subsections = []
        for question in questions:
            try:
                result = self.qa_pipeline(question=question, context=section['content'])
                if result['score'] > 0.05: # Lowered threshold to be more inclusive
                    subsections.append({
                        'document': section['document'], 'page_number': section.get('page_number', 1),
                        'refined_text': result['answer'].strip(), 'relevance_score': result['score']
                    })
            except Exception:
                continue
        return subsections

    def keyword_sentence_extraction(self, section: Dict, query_representation: Dict) -> List[Dict]:
        """A robust fallback that scores sentences based on keyword density."""
        content = section['content']
        keywords = set(query_representation['expanded_keywords'])
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', content) if len(s.strip().split()) > 5]
        if not sentences: return []
        
        scored_sentences = []
        for sentence in sentences:
            words = set(sentence.lower().split())
            score = len(words.intersection(keywords))
            if score > 0:
                scored_sentences.append({'text': sentence, 'score': score})
        
        scored_sentences.sort(key=lambda x: x['score'], reverse=True)
        
        subsections = []
        for item in scored_sentences[:3]: # Take top 3 sentences
            subsections.append({
                'document': section['document'], 'page_number': section.get('page_number', 1),
                'refined_text': item['text'], 'relevance_score': item['score'] / len(keywords) if keywords else 0
            })
        return subsections
    
    def refine_qa_answer(self, qa_result: Dict, full_content: str) -> str:
        """Return the direct, short answer from the QA model."""
        answer = qa_result['answer'].strip()
        # Clean up any non-alphanumeric characters at the start/end
        answer = re.sub(r'^[^\w\s]+', '', answer)
        answer = re.sub(r'[^\w\s]+$', '', answer)
        return answer
    
    def extractive_summarization(self, content: str, document: str, page_number: int) -> List[Dict]:
        """Extract key sentences using extractive summarization"""
        
        subsections = []
        
        if HAS_SUMY and self.summarizer:
            try:
                # Use Sumy for extractive summarization
                parser = PlaintextParser.from_string(content, Tokenizer("english"))
                summary_sentences = self.summarizer(parser.document, 3)  # Top 3 sentences
                
                for i, sentence in enumerate(summary_sentences):
                    refined_text = str(sentence)
                    
                    if len(refined_text) > 50:  # Minimum length
                        # Clean the text output
                        refined_text = self.clean_text_output(refined_text)
                        
                        subsection = {
                            'document': document,
                            'subsection_id': f"summary_{i+1}",
                            'refined_text': refined_text,
                            'page_number': page_number,
                            'extraction_method': 'extractive_summary',
                            'relevance_score': 0.7 - (i * 0.1)  # Decreasing relevance
                        }
                        subsections.append(subsection)
                        
            except Exception as e:
                logger.warning(f"Extractive summarization error: {e}")
        
        # Fallback: simple sentence extraction
        if not subsections:
            sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 50]
            for i, sentence in enumerate(sentences[:3]):
                refined_text = sentence + '.'
                # Clean the text output
                refined_text = self.clean_text_output(refined_text)
                
                subsection = {
                    'document': document,
                    'subsection_id': f"sentence_{i+1}",
                    'refined_text': refined_text,
                    'page_number': page_number,
                    'extraction_method': 'sentence_extraction',
                    'relevance_score': 0.6 - (i * 0.1)
                }
                subsections.append(subsection)
        
        return subsections
    
    def structural_splitting(self, content: str, document: str, page_number: int) -> List[Dict]:
        """Split content structurally as fallback method"""
        
        # Split by paragraphs or logical breaks
        paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 50]
        
        if not paragraphs:
            # Split by sentences
            sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 50]
            if len(sentences) > 0:
                paragraphs = ['. '.join(sentences[i:i+2]) + '.' for i in range(0, len(sentences), 2) if i < len(sentences)]
        
        subsections = []
        for i, paragraph in enumerate(paragraphs[:3]):
            if paragraph:
                # Clean the text output
                refined_text = self.clean_text_output(paragraph)
                
                subsection = {
                    'document': document,
                    'subsection_id': f"structural_{i+1}",
                    'refined_text': refined_text,
                    'page_number': page_number,
                    'extraction_method': 'structural',
                    'relevance_score': 0.5 - (i * 0.05)
                }
                subsections.append(subsection)
        
        return subsections


class WinningDocumentIntelligenceSystem:
    """Main system orchestrating the five-stage pipeline"""
    
    def __init__(self):
        self.query_processor = AdvancedQueryProcessor()
        self.document_parser = EnhancedDocumentParser()
        self.search_engine = HybridSearchEngine()
        self.reranker = AdvancedReRanker()
        self.subsection_analyzer = AdvancedSubsectionAnalyzer()
        
        logger.info("Document Intelligence System initialized")
    
    def process_collection(self, input_file: str, output_file: str):
        """Process document collection using the five-stage pipeline"""
        
        start_time = time.time()
        logger.info(f"Processing collection: {input_file}")
        
        # Load input
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        # Stage 1: Advanced Query Understanding
        logger.info("Stage 1: Advanced Query Understanding")
        query_representation = self.query_processor.process_query(
            input_data['persona'], 
            input_data['job_to_be_done']
        )
        
        # Stage 2: Enhanced Document Parsing
        logger.info("Stage 2: Enhanced Document Parsing")
        pdf_dir = Path(input_file).parent / "PDFs"
        all_sections = []
        
        for doc in input_data['documents']:
            pdf_path = pdf_dir / doc['filename']
            if pdf_path.exists():
                sections = self.document_parser.parse_document(pdf_path, doc['filename'])
                all_sections.extend(sections)
        
        logger.info(f"Parsed {len(all_sections)} sections from {len(input_data['documents'])} documents")
        
        # Stage 3: Hybrid Search
        logger.info("Stage 3: Hybrid Search (Semantic + Keyword)")
        candidates = self.search_engine.search(query_representation, all_sections, top_k=25)
        
        # Stage 4: Precision Re-ranking
        logger.info("Stage 4: Precision Re-ranking with Multi-feature Scoring")
        ranked_sections = self.reranker.rerank_candidates(query_representation, candidates)
        
        # Take top sections for output - optimized for competition scoring
        # Use adaptive selection based on collection size
        num_sections = min(len(ranked_sections), max(10, len(all_sections) // 10))
        top_sections = ranked_sections[:num_sections]
        
        # Stage 5: Advanced Subsection Analysis
        logger.info("Stage 5: Advanced Subsection Analysis")
        subsections = self.subsection_analyzer.analyze_subsections(top_sections, query_representation)
        
        # Create output in required format
        output_data = {
            "metadata": {
                "input_documents": [doc['filename'] for doc in input_data['documents']],
                "persona": input_data['persona']['role'],
                "job_to_be_done": input_data['job_to_be_done']['task'],
                "processing_timestamp": datetime.now().isoformat(),
                "processing_time_seconds": round(time.time() - start_time, 2),
                "system_version": "winning_v1.0",
                "pipeline_stages": 5
            },
            "extracted_sections": [],
            "subsection_analysis": []  # Required field name for output format
        }
        
        # Add top sections
        for i, section in enumerate(top_sections, 1):
            output_section = {
                "document": section['document'],
                "page_number": section.get('page_number', 1),
                "section_title": section['section_title'],
                "importance_rank": i
            }
            output_data["extracted_sections"].append(output_section)
        
        # Add subsections in correct format
        for subsection in subsections:
            # Handle different field names from different extraction methods
            text_content = subsection.get('refined_text') or subsection.get('subsection_text') or subsection.get('text', '')
            # Clean the text content
            text_content = self.subsection_analyzer.clean_text_output(text_content)
            
            output_subsection = {
                "document": subsection['document'],
                "refined_text": text_content,
                "page_number": subsection['page_number']
            }
            output_data["subsection_analysis"].append(output_subsection)
        
        # Save output
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        processing_time = time.time() - start_time
        top_scores = [f"{s['relevance_score']:.3f}" for s in top_sections]
        
        logger.info(f"Processing complete in {processing_time:.2f}s")
        logger.info(f"Top relevance scores: {top_scores}")
        logger.info(f"Subsections extracted: {len(subsections)}")
        
        return output_data


def main():
    """Main execution function for the system"""
    
    logger.info("Starting Document Intelligence System")
    logger.info(f"Advanced NLP capabilities: ST={HAS_SENTENCE_TRANSFORMERS}, "
                f"T={HAS_TRANSFORMERS}, Y={HAS_YAKE}, BM25={HAS_BM25}, S={HAS_SUMY}")
    
    system = WinningDocumentIntelligenceSystem()
    base_dir = Path(__file__).parent
    
    # Create our_outputs directory if it doesn't exist
    our_outputs_dir = base_dir / "our_outputs"
    our_outputs_dir.mkdir(exist_ok=True)
    
    # Automatically discover all collections (100% generic approach)
    collections_found = []
    for item in base_dir.iterdir():
        if item.is_dir() and (item / "challenge1b_input.json").exists():
            collections_found.append(item)
    
    # Sort collections for consistent processing order
    collections_found.sort(key=lambda x: x.name)
    
    if not collections_found:
        logger.error("❌ No valid collections found. Looking for directories with 'challenge1b_input.json'")
        return
    
    logger.info(f"🔍 Found {len(collections_found)} collections to process")
    
    # Process all discovered collections (works with any naming scheme)
    for idx, collection_dir in enumerate(collections_found, 1):
        input_file = collection_dir / "challenge1b_input.json"
        # Generate safe output filename from collection directory name
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', collection_dir.name.lower())
        output_file = our_outputs_dir / f"{safe_name}_output.json"
        
        logger.info(f"Processing collection: {collection_dir.name}")
        
        try:
            system.process_collection(str(input_file), str(output_file))
            logger.info(f"Collection '{collection_dir.name}' completed successfully")
        except Exception as e:
            logger.error(f"Collection '{collection_dir.name}' failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
