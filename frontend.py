import gradio as gr
from agi_core import AGI
import logging

agi = AGI()

def setup_logging():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def run_chatbot(user_input, state):
    logging.info(f"User input: {user_input}")
    response = agi.run(user_input)
    logging.info(f"AGI response: {response}")
    return response, state

def launch():
    setup_logging()

    initial_goals = [
        "Engage in conversation with the user",
        "Answer user queries based on knowledge",
        "Learn from the conversation"
    ]

    agi = AGI()
    agi.set_goals(initial_goals)

    logging.info("AGI system initialized")

    iface = gr.Interface(
        fn=run_chatbot,
        inputs=[gr.Textbox(label="User Input"), gr.State()],
        outputs=[gr.Textbox(label="AGI Response"), gr.State()],
        allow_flagging="never",
        title="AGI Chatbot",
        description="Engage in a conversation with the AGI chatbot.",
        examples=[
            ["Hello! How are you?"],
            ["What can you tell me about AI?"],
            ["Can you recommend a good book?"]
        ],
        cache_examples=False,
    )

    iface.launch(share=True)

if __name__ == "__main__":
    launch()
