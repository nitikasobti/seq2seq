import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx

# Ensure you have the necessary NLTK resources
nltk.download('punkt_tab')

def textrank_summarizer(text, percentage=20, format='paragraph'):
    # Step 1: Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)
    total_sentences = len(sentences)
    
    # Step 2: Create a sentence similarity matrix using CountVectorizer
    vectorizer = CountVectorizer().fit_transform(sentences)
    similarity_matrix = cosine_similarity(vectorizer)
    
    # Step 3: Build a graph using the similarity matrix
    nx_graph = nx.from_numpy_array(similarity_matrix)
    
    # Step 4: Rank the sentences using the PageRank algorithm
    scores = nx.pagerank(nx_graph)
    
    # Step 5: Sort sentences by their scores
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    # Step 6: Determine the number of sentences based on user percentage
    num_sentences = max(1, int(total_sentences * (percentage / 100)))
    
    # Step 7: Select the top N sentences based on ranking
    selected_sentences = [sentence for _, sentence in ranked_sentences[:num_sentences]]
    
    # Step 8: Format the summary based on user preference
    if format == 'points':
        summary = "\n".join(f"- {sentence}" for sentence in selected_sentences)
    else:
        summary = " ".join(selected_sentences)
    
    return summary
