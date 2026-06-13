from dotenv import load_dotenv
load_dotenv()
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os

class BaseAgent(ABC):
    """Base class for all specialized agents"""
    
    def __init__(self, name: str, role: str, model_provider: str = "groq"):
        self.name = name
        self.role = role
        self.llm = self._initialize_llm(model_provider)
        self.memory: List[Dict] = []
        
    def _initialize_llm(self, provider: str):
        """Initialize the language model"""
        provider = provider.lower().strip() # 防止大小写导致匹配失败
        
        if provider == "groq":
            return ChatGroq(
                groq_api_key=os.getenv('GROQ_API_KEY'),
                model="llama-3.1-8b-instant", # 升级为更稳定的 Llama 3.1 现代模型
                temperature=0.1,
                max_tokens=1500,
            )
        elif provider == "openai":
            return ChatOpenAI(
                model="gpt-4o-mini",          # 1. 修正参数名为 model  2. 修正模型名称
                api_key=os.getenv('OPENAI_API_KEY'),
                temperature=0.1
            )
        else:
            # 安全防线：抛出清晰的错误，而不是让它返回 None 导致后续莫名崩掉
            raise ValueError(f"Unsupported model provider: '{provider}'. Choose 'groq' or 'openai'.")
        
    @abstractmethod
    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task specific to this agent"""
        pass
        
    def add_to_memory(self, interaction: Dict):
        """Add interaction to agent memory"""
        self.memory.append(interaction)