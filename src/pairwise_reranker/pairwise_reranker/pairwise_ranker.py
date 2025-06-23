import numpy as np
from typing import List, Dict, Tuple, Optional
from itertools import combinations
from groq import Groq
from .prompts.pairwise_prompt import PairwisePromptGenerator

class PairwiseReranker:
    def __init__(self, model_name: str = "llama3-70b-8192", api_key: str = None):
        """
        Initialize the pairwise reranker with Groq's LLM.
        """
        if not api_key:
            raise ValueError("Groq API key is required")
            
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.comparison_cache = {}  # Cache for comparison results
        self.token_count = 0  # Track token usage
        self.prompt_generator = PairwisePromptGenerator()
    
    def generate_pairwise_prompt(self, query: str, doc1: str, doc2: str) -> List[Dict]:
        """
        Generate the prompt for pairwise comparison between two documents.
        """
        return self.prompt_generator.generate(query, doc1, doc2)
    
    def compare_pair(self, query: str, doc1: str, doc2: str) -> Optional[str]:
        """
        Compare two documents for a given query using Groq API.
        Returns:
            'A' if doc1 is better, 'B' if doc2 is better, 'T' for tie, or None on error
        """
        # Create a cache key (order-independent)
        h1, h2 = hash(doc1), hash(doc2)
        cache_key = (min(h1, h2), max(h1, h2), hash(query))
        
        if cache_key in self.comparison_cache:
            return self.comparison_cache[cache_key]
            
        messages = self.generate_pairwise_prompt(query, doc1, doc2)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=1
            )
            
            # Track token usage
            self.token_count += response.usage.total_tokens
            
            choice = response.choices[0].message.content.strip().upper()
            
            if choice in ['A', 'B', 'T']:
                self.comparison_cache[cache_key] = choice
                return choice
            return 'T'  # Treat invalid responses as ties
            
        except Exception as e:
            print(f"Error in comparison: {e}")
            return 'T'  # Treat errors as ties
    
    def get_token_count(self) -> int:
        """Get total token usage for the session"""
        return self.token_count
    
    def _heapsort(self, query: str, documents: List[str], indices: List[int]) -> List[int]:
        """
        Heapsort implementation using Groq pairwise comparisons.
        Returns list of sorted indices (best first).
        """
        n = len(indices)
        if n <= 1:
            return indices
            
        # Build max-heap: best document at root
        def llm_comparator(i: int, j: int) -> bool:
            """Return True if document i is better than document j"""
            result = self.compare_pair(query, documents[i], documents[j])
            if result == 'A':  # i is better
                return True
            elif result == 'B':  # j is better
                return False
            else:  # Tie or error
                return False  # Maintain current order

        # Heapify from the bottom up
        for i in range(n//2 - 1, -1, -1):
            self._siftdown(indices, i, n, llm_comparator)
        
        # Extract elements from heap
        for i in range(n-1, 0, -1):
            indices[i], indices[0] = indices[0], indices[i]  # swap
            self._siftdown(indices, 0, i, llm_comparator)
        
        return indices
    
    def _siftdown(self, arr: List[int], i: int, n: int, comparator) -> None:
        """Heap siftdown operation"""
        while True:
            left = 2 * i + 1
            right = 2 * i + 2
            largest = i
            
            if left < n and comparator(arr[left], arr[largest]):
                largest = left
            if right < n and comparator(arr[right], arr[largest]):
                largest = right
                
            if largest == i:
                break
                
            arr[i], arr[largest] = arr[largest], arr[i]
            i = largest
    
    def rank_documents_prp_sorting(self, query: str, documents: List[str]) -> List[Tuple[int, int]]:
        """
        Rank documents using PRP-Sorting approach with Heapsort.
        Returns list of (index, rank) pairs where rank is 1-based (1 = best).
        """
        if not documents:
            return []
            
        indices = list(range(len(documents)))
        sorted_indices = self._heapsort(query, documents, indices)
        
        # Convert to (index, rank) pairs (1-based rank)
        return [(idx, rank+1) for rank, idx in enumerate(sorted_indices)]
    
    def rank_documents_prp_allpair(self, query: str, documents: List[str]) -> List[Tuple[int, float]]:
        """
        Rank documents using PRP-Allpair approach with global aggregation.
        """
        n = len(documents)
        if n == 0:
            return []
            
        wins = np.zeros(n)
        comparisons = np.zeros(n)
        
        for i, j in combinations(range(n), 2):
            result = self.compare_pair(query, documents[i], documents[j])
            
            if result == 'A':
                wins[i] += 1
            elif result == 'B':
                wins[j] += 1
            elif result == 'T':  # Tie
                wins[i] += 0.5
                wins[j] += 0.5
                
            comparisons[i] += 1
            comparisons[j] += 1
        
        # Normalize scores (win percentage)
        scores = np.divide(wins, comparisons, where=comparisons > 0, out=np.zeros(n))
        return list(enumerate(scores))
    
    def rerank(self, 
              query: str, 
              documents: List[str], 
              method: str = "allpair",
              initial_ranking: Optional[List[int]] = None,
              top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Rerank documents using specified method.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            method: "sorting" (PRP-Sorting) or "allpair" (PRP-Allpair)
            initial_ranking: Optional initial ranking for tie-breaking
            top_k: Number of top documents to return
            
        Returns:
            List of (index, score) pairs sorted by relevance
        """
        if len(documents) > 20 and method == "allpair":
            print("Warning: Allpair method may be expensive for >20 documents")
        
        if method == "sorting":
            ranked = self.rank_documents_prp_sorting(query, documents)
            # Convert ranks to scores (higher rank = better)
            max_rank = len(documents)
            scored_results = [(idx, max_rank - rank + 1) for idx, rank in ranked]
        else:
            scored_results = self.rank_documents_prp_allpair(query, documents)
        
        # Sort by score descending
        scored_results.sort(key=lambda x: -x[1])
        
        # Apply initial ranking for ties if provided
        if initial_ranking is not None:
            rank_dict = {idx: i for i, idx in enumerate(initial_ranking)}
            scored_results.sort(key=lambda x: (-x[1], rank_dict.get(x[0], float('inf'))))
        
        return scored_results[:top_k]

# Example usage
if __name__ == "__main__":
    # Initialize with your Groq API key
    reranker = PairwiseReranker(
        model_name="llama3-70b-8192",
        api_key="your_api_key_here"  # Replace with your actual key
    )

    query = "What are the benefits of regular exercise?"
    documents = [
        "Regular exercise improves cardiovascular health and reduces stress.",
        "Exercise helps with weight management and boosts mood.",
        "Physical activity increases energy levels and improves sleep.",
        "Working out strengthens muscles and bones.",
        "Studies show exercise reduces risk of chronic diseases."
    ]
    
    # Get initial ranking (e.g., from BM25)
    initial_ranking = [0, 1, 2, 3, 4]

    # PRP-Sorting approach
    print("PRP-Sorting Results:")
    results = reranker.rerank(query, documents, method="sorting", initial_ranking=initial_ranking)
    for idx, score in results:
        print(f"Rank {score:.1f}: {documents[idx][:50]}...")
    
    # PRP-Allpair approach
    print("\nPRP-Allpair Results:")
    results = reranker.rerank(query, documents, method="allpair", initial_ranking=initial_ranking)
    for idx, score in results:
        print(f"Score {score:.3f}: {documents[idx][:50]}...")
    
    print(f"\nTotal tokens used: {reranker.get_token_count()}")