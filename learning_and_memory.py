import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

class LearningMemoryModule:
    def __init__(self):
        self.long_term_memory = pd.DataFrame(columns=['experience', 'knowledge'])
        self.nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, alpha=0.0001, solver='adam', random_state=42)
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def update(self, experience, knowledge):
        logging.debug(f"Updating long-term memory with experience: {experience} and knowledge: {knowledge}")
        new_data = pd.DataFrame({'experience': [experience], 'knowledge': [knowledge]})
        self.long_term_memory = pd.concat([self.long_term_memory, new_data], ignore_index=True)

    def extract_knowledge(self, experience):
        logging.debug(f"Extracting knowledge from experience: {experience}")
        # Use TF-IDF to extract keywords as a simple form of knowledge
        tfidf_matrix = self.vectorizer.fit_transform([experience])
        feature_names = self.vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        top_keywords = [feature_names[i] for i in scores.argsort()[-5:]]
        knowledge = ', '.join(top_keywords)
        logging.debug(f"Extracted knowledge: {knowledge}")
        return knowledge

    def train_models(self):
        X = self.vectorizer.fit_transform(self.long_term_memory['experience'])
        y = self.long_term_memory['knowledge'].tolist()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.nn_model.fit(X_train, y_train)
        accuracy = self.nn_model.score(X_test, y_test)
        logging.debug(f"Neural Network Accuracy: {accuracy}")

    def predict(self, experience):
        logging.debug(f"Predicting knowledge for experience: {experience}")
        experience_tfidf = self.vectorizer.transform([experience])
        knowledge = self.nn_model.predict(experience_tfidf)[0]
        logging.debug(f"Predicted knowledge: {knowledge}")
        return knowledge

    def retrieve_memory(self, query):
        logging.debug(f"Retrieving memory for query: {query}")
        if self.long_term_memory.empty:
            logging.debug("Long-term memory is empty")
            return None
        
        query_tfidf = self.vectorizer.transform([query])
        experiences_tfidf = self.vectorizer.transform(self.long_term_memory['experience'])
        similarities = cosine_similarity(query_tfidf, experiences_tfidf).flatten()
        most_similar_index = similarities.argmax()
        most_similar_experience = self.long_term_memory.iloc[most_similar_index]
        logging.debug(f"Retrieved memory: {most_similar_experience}")
        
        return most_similar_experience