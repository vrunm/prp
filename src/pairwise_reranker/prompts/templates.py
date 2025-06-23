# System prompt template
SYSTEM_PROMPT = (
    "You are a ranking assistant that compares two passages in response to a query. "
    "Respond ONLY with 'A' if Passage A is better, 'B' if Passage B is better, or 'T' if they are equal."
)

# User prompt template
USER_PROMPT = (
    "Query: {query}\n\n"
    "Passage A: {doc1}\n\n"
    "Passage B: {doc2}\n\n"
    "Which passage is more relevant? Respond ONLY with A, B, or T:"
)