from langchain_community.llms import ollama
from knowledge_base import KnowledgeBase
from perception_and_action import PerceptionActionInterface
from learning_and_memory import LearningMemoryModule
import logging

class AGI:
    def __init__(self):
        self.llm = ollama.Ollama(model="mistral")
        self.knowledge_base = KnowledgeBase()
        self.learning_memory = LearningMemoryModule()
        self.goals = []
        self.current_plan = []

    def perceive(self, text_input):
        logging.debug(f"Received input: {text_input}")
        vector = self.knowledge_base.vectorize(text_input)
        
        # Generate a unique ID for each user input
        input_id = f"user_input_{len(self.knowledge_base.graph.nodes)}"
        
        observation = {'id': input_id, 'data': {'text': text_input, 'vector': vector}}
        self.knowledge_base.update(observation)

    def reason(self):
        latest_input_id = f"user_input_{len(self.knowledge_base.graph.nodes) - 1}"
        user_input = self.knowledge_base.get_node(latest_input_id)['text']
        user_input_vector = self.knowledge_base.get_node(latest_input_id)['vector']

        relevant_memories = self.knowledge_base.search_by_vector(user_input_vector)

        prompt = f"User Input: {user_input}\n\n"
        prompt += "Relevant Memories:\n"
        for memory in relevant_memories:
            prompt += f"- Node ID: {memory['id']}, Text: \"{memory['text']}\", Relevance: {memory['relevance']:.2f}\n"
        
        prompt += "\nGoals:\n"
        for goal in self.goals:
            prompt += f"- {goal}\n"

        prompt += "\nBased on the relevant memories and the user's input, " \
                "provide a specific and concise response to the question or prompt. " \
                "If the answer can be directly inferred from the relevant memories, " \
                "state it clearly. If the relevant memories seem to be irrelevant, you can ignore them. " \
                " If ultimately you can't decide, provide a friendly response indicating that " \
                "you don't have enough information to answer confidently.\n\nResponse:"

        logging.debug(f"Generated prompt for reasoning: {prompt}")
        response = self.llm(prompt)
        logging.debug(f"Generated response: {response}")
        self.current_plan.append(response)

    def learn(self):
        context = self.knowledge_base.get_context()
        experiences = [msg['data']['text'] for msg in context]
        experience = " ".join(experiences)
        
        knowledge = self.learning_memory.extract_knowledge(experience)
        self.learning_memory.update(experience, knowledge)

        knowledge_dict = {
            'id': f'knowledge_{len(self.knowledge_base.graph.nodes)}',
            'data': {
                'category': 'food_preferences',
                'food_item': knowledge.split(',')[3].strip(),  # Assumes the 4th item is the food
                'sentiment': knowledge.split(',')[0].strip()   # Assumes the 1st item is the sentiment
            }
        }
        self.knowledge_base.integrate_new_knowledge(knowledge_dict)

    def run(self, text_input):
        self.perceive(text_input)
        self.reason()
        self.learn()
        return self.current_plan[-1]

    def set_goals(self, goals):
        self.goals = goals
        logging.info(f"Goals set: {goals}")