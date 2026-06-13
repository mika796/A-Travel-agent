from .base_agent import BaseAgent
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
# 注释掉境外搜索工具，不再导入
# from utils.place_info_search import GooglePlaceSearchTool, TavilyPlaceSearchTool
import os
import asyncio
import time

class ResearchAgent(BaseAgent):
    """Agent specialized in destination research and attractions"""
    
    def __init__(self, model_provider: str = "groq"):
        super().__init__(
            name="Research Agent",
            role="Destination research and attraction discovery specialist",
            model_provider=model_provider
        )
        
        # 完全禁用 Google / Tavily 搜索工具
        # gplaces_key = os.getenv("GPLACES_API_KEY", "")
        # self.google_places_search = GooglePlaceSearchTool(gplaces_key)
        # self.tavily_search = TavilyPlaceSearchTool()
        
    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process research tasks"""
        task_type = task.get("type", "destination_research")
        
        if task_type == "destination_research":
            return await self._research_destination(task)
        elif task_type == "attractions":
            return await self._find_attractions(task)
        else:
            return await self._general_research(task)
            
    async def _research_destination(self, task: Dict) -> Dict:
        """Research destination details"""
        destination = task.get("destination")
        duration = task.get("duration", "5 days")
        
        # 精简提示词，大幅减少Token，解决413超限
        system_prompt = f"Help plan a {duration} trip to {destination}, keep reply concise."
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Introduce {destination} for a {duration} trip.")
        ]
        
        # 增加重试机制，应对限流
        try:
            response = await self.llm.ainvoke(messages)
        except Exception as e:
            print("请求超限，等待60秒重试...")
            time.sleep(60)
            response = await self.llm.ainvoke(messages)
        
        attractions = f"【AI 推荐{destination}热门景点、美食与游玩项目】：请查看下方行程规划内容"

        self.add_to_memory({
            "task": "destination_research",
            "destination": destination,
            "response": response.content
        })
        
        return {
            "agent": self.name,
            "task_type": "destination_research",
            "destination": destination,
            "research_data": response.content,
            "attractions": attractions,
            "status": "completed"
        }
        
    async def _find_attractions(self, task: Dict) -> Dict:
        """Find attractions"""
        destination = task.get("destination")
        
        attractions = f"{destination} 热门游玩景点（AI 推荐）"
        restaurants = f"{destination} 特色美食与餐厅（AI 推荐）"
        activities = f"{destination} 休闲体验活动（AI 推荐）"
            
        return {
            "agent": self.name,
            "task_type": "attractions",
            "destination": destination,
            "attractions": attractions,
            "restaurants": restaurants,
            "activities": activities,
            "status": "completed"
        }
        
    async def _general_research(self, task: Dict) -> Dict:
        """General research fallback"""
        query = task.get("query", "")
        
        messages = [
            SystemMessage(content="Answer briefly."),
            HumanMessage(content=query)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
        except Exception as e:
            print("请求超限，等待60秒重试...")
            time.sleep(60)
            response = await self.llm.ainvoke(messages)
        
        return {
            "agent": self.name,
            "task_type": "general_research",
            "response": response.content,
            "status": "completed"
        }