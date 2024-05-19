from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import ollama
import networkx as nx
import logging

class KnowledgeBase:
    def __init__(self):
        self.graph = nx.Graph()
        self.messages = []

    def add_node(self, node_id, data):
        #logging.debug(f"Adding node: {node_id}, Data: {data}")
        self.graph.add_node(node_id, **data)

    def update(self, observation):
        node_id = observation['id']
        node_data = observation['data']
        self.add_node(node_id, node_data)
        self.messages.append(observation)

    def get_node(self, node_id):
        return self.graph.nodes[node_id]

    def search_by_vector(self, query_vector, top_k=5):
        similarities = []
        for node_id in self.graph.nodes:
            node_data = self.get_node(node_id)
            if 'vector' in node_data:
                node_vector = node_data['vector']
                similarity = cosine_similarity([query_vector], [node_vector])[0][0]
                similarities.append({'id': node_id, 'similarity': similarity})

        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_nodes = similarities[:top_k]

        relevant_memories = []
        for node in top_nodes:
            node_data = self.get_node(node['id'])
            relevant_memories.append({
                'id': node['id'],
                'text': node_data['text'],
                'relevance': node['similarity']
            })

        return relevant_memories

    def vectorize(self, text):
        vector = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return normalize([vector['embedding']])[0]

    def get_context(self, n=5):
        return self.messages[-n:]

    def integrate_new_knowledge(self, new_knowledge):
        """
        Integrates new knowledge into the knowledge base, ensuring relevance and non-redundancy.
        """
        # Check that new_knowledge is a dictionary and has an 'id'
        if not isinstance(new_knowledge, dict) or 'id' not in new_knowledge:
            logging.error("Invalid new_knowledge format: Expected a dictionary with an 'id' key.")
            return  # Or raise an exception based on your error handling strategy

        logging.debug(f"Attempting to integrate new knowledge: {new_knowledge}")
        existing_knowledge = self.retrieve(new_knowledge['id'])
        if existing_knowledge:
            # Merge or update the knowledge based on some criteria
            self.merge_knowledge(existing_knowledge, new_knowledge)
        else:
            # Directly add new knowledge if not redundant
            self.graph.add_node(new_knowledge['id'], **new_knowledge['data'])
        logging.info(f"New knowledge integrated successfully: {new_knowledge['id']}")

    def retrieve(self, node_id):
        """
        Retrieves a node from the knowledge base.
        """
        if self.graph.has_node(node_id):
            return self.graph.nodes[node_id]
        else:
            return None

    def merge_knowledge(self, existing_knowledge, new_knowledge):
        """
        Merges new knowledge into existing knowledge.
        """
        logging.debug(f"Merging knowledge: {existing_knowledge}, {new_knowledge}")
        # Merge or update the knowledge based on some criteria
        self.graph.nodes[existing_knowledge['id']].update(new_knowledge['data'])
        logging.info(f"Knowledge merged successfully: {existing_knowledge['id']}")


