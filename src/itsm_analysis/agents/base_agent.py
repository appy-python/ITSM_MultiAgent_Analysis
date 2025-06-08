import os
from abc import ABC, abstractmethod
from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv

load_dotenv()

model = os.getenv("MODEL")
class BaseAgent(ABC):
    def __init__(self, name):
        self.name = name
        self.llm = ChatOllama(model=model)

    @abstractmethod
    def run(self, state: dict) -> dict:
        """Process input state and return an updated state."""
        pass