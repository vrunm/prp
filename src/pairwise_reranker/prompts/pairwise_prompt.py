from . import templates
from typing import List, Dict

class PairwisePromptGenerator:
    def generate(self, query: str, doc1: str, doc2: str) -> List[Dict[str, str]]:
        """Generate the prompt for pairwise comparison"""
        # Truncate documents for efficiency
        doc1_trunc = doc1[:1000]
        doc2_trunc = doc2[:1000]
        
        user_content = templates.USER_PROMPT.format(
            query=query,
            doc1=doc1_trunc,
            doc2=doc2_trunc
        )
        
        return [
            {"role": "system", "content": templates.SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]