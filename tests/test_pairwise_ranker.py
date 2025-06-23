import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pairwise_reranker.pairwise_reranker import PairwiseReranker

@pytest.fixture
def mock_reranker():
    """Fixture for PairwiseReranker with mocked Groq client"""
    with patch('pairwise_reranker.pairwise_ranker.Groq') as mock_groq, \
         patch('pairwise_reranker.pairwise_ranker.PairwisePromptGenerator') as mock_prompt:
        # Mock prompt generation
        mock_prompt.return_value.generate.return_value = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User prompt with {query}, {doc1}, {doc2}"}
        ]
        
        reranker = PairwiseReranker(api_key="test_key")
        reranker.client = MagicMock()
        reranker.comparison_cache = {}
        return reranker

def test_generate_pairwise_prompt(mock_reranker):
    """Test prompt generation structure"""
    query = "test query"
    doc1 = "Document A content"
    doc2 = "Document B content"
    
    prompt = mock_reranker.generate_pairwise_prompt(query, doc1, doc2)
    
    assert len(prompt) == 2
    assert prompt[0]["role"] == "system"
    assert prompt[1]["role"] == "user"

def test_compare_pair(mock_reranker):
    """Test comparison logic with various responses"""
    # Mock API response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = " A "  # Test whitespace handling
    
    # Test valid responses
    mock_reranker.client.chat.completions.create.return_value = mock_response
    assert mock_reranker.compare_pair("q", "doc1", "doc2") == "A"
    
    mock_response.choices[0].message.content = "B"
    assert mock_reranker.compare_pair("q", "doc1", "doc2") == "B"
    
    # Test tie
    mock_response.choices[0].message.content = "T"
    assert mock_reranker.compare_pair("q", "doc1", "doc2") == "T"
    
    # Test invalid response
    mock_response.choices[0].message.content = "INVALID"
    assert mock_reranker.compare_pair("q", "doc1", "doc2") == "T"
    
    # Test caching
    mock_reranker.comparison_cache = {(hash("doc1"), hash("doc2"), hash("q")): "B"}
    assert mock_reranker.compare_pair("q", "doc1", "doc2") == "B"

def test_heapsort(mock_reranker):
    """Test heapsort implementation"""
    # Setup mock comparisons
    mock_reranker.compare_pair = MagicMock(side_effect=[
        'A',  # 0 vs 1 -> 0 better
        'B',  # 0 vs 2 -> 2 better
        'B',  # 1 vs 2 -> 2 better
    ])
    
    documents = ["doc0", "doc1", "doc2"]
    indices = [0, 1, 2]
    
    sorted_indices = mock_reranker._heapsort("query", documents, indices)
    
    # Should be [2, 0, 1] since:
    # - 2 beats both 0 and 1
    # - 0 beats 1
    assert sorted_indices == [2, 0, 1]

def test_rank_prp_sorting(mock_reranker):
    """Test PRP-sorting ranking method"""
    mock_reranker._heapsort = MagicMock(return_value=[2, 0, 1])
    documents = ["doc0", "doc1", "doc2"]
    
    ranked = mock_reranker.rank_documents_prp_sorting("query", documents)
    
    # Expected: index 2 (rank1), index 0 (rank2), index 1 (rank3)
    assert ranked == [(2, 1), (0, 2), (1, 3)]

def test_rank_prp_allpair(mock_reranker):
    """Test PRP-allpair ranking method"""
    mock_reranker.compare_pair = MagicMock(side_effect=[
        'A',  # 0 vs 1 -> 0 wins
        'B',  # 0 vs 2 -> 2 wins
        'T',  # 1 vs 2 -> tie
    ])
    
    documents = ["doc0", "doc1", "doc2"]
    ranked = mock_reranker.rank_documents_prp_allpair("query", documents)
    
    # Calculate expected scores:
    # doc0: 1 win (vs doc1) + 0.5? -> actually 1 win, 1 loss -> 1/2 = 0.5
    # doc1: 1 loss (vs doc0) + 0.5 tie (vs doc2) -> 0.5/2 = 0.25
    # doc2: 1 win (vs doc0) + 0.5 tie (vs doc1) -> 1.5/2 = 0.75
    scores = {idx: score for idx, score in ranked}
    assert scores[0] == 0.5
    assert scores[1] == 0.25
    assert scores[2] == 0.75

def test_rerank_methods(mock_reranker):
    """Test both reranking methods"""
    documents = ["doc0", "doc1", "doc2"]
    initial_ranking = [0, 1, 2]
    
    # Mock underlying ranking methods
    mock_reranker.rank_documents_prp_sorting = MagicMock(
        return_value=[(2, 1), (0, 2), (1, 3)]
    )
    mock_reranker.rank_documents_prp_allpair = MagicMock(
        return_value=[(0, 0.5), (1, 0.25), (2, 0.75)]
    )
    
    # Test sorting method
    sorting_results = mock_reranker.rerank("q", documents, method="sorting")
    # Should convert ranks to scores: 3-1=2, 3-2=1, 3-3=0
    assert sorting_results == [(2, 2.0), (0, 1.0), (1, 0.0)]
    
    # Test allpair method
    allpair_results = mock_reranker.rerank("q", documents, method="allpair")
    assert allpair_results == [(2, 0.75), (0, 0.5), (1, 0.25)]

def test_rerank_tie_breaking(mock_reranker):
    """Test initial ranking tie-breaking"""
    mock_reranker.rank_documents_prp_allpair = MagicMock(
        return_value=[(0, 1.0), (1, 1.0), (2, 1.0)]  # All same score
    )
    
    documents = ["doc0", "doc1", "doc2"]
    initial_ranking = [2, 0, 1]  # Preferred order: 2 > 0 > 1
    
    results = mock_reranker.rerank(
        "q", documents, method="allpair", initial_ranking=initial_ranking
    )
    
    # Should maintain initial ranking order for ties
    assert [idx for idx, _ in results] == [2, 0, 1]

def test_edge_cases(mock_reranker):
    """Test empty documents and small lists"""
    # Empty documents
    assert mock_reranker.rerank("q", []) == []
    
    # Single document
    results = mock_reranker.rerank("q", ["single doc"])
    assert results == [(0, 1.0)]
    
    # Two documents
    mock_reranker.compare_pair = MagicMock(return_value="A")
    results = mock_reranker.rerank("q", ["doc1", "doc2"], method="allpair")
    assert results == [(0, 1.0), (1, 0.0)] or results == [(0, 1.0), (1, 0.5)]