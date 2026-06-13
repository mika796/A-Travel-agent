from .base_agent import BaseAgent
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from utils.weather_info import WeatherForecastTool
import os
import asyncio

class WeatherAgent(BaseAgent):
    """Agent specialized in weather forecasting and analysis"""
    
    def __init__(self, model_provider: str = "groq"):
        super().__init__(
            name="Weather Agent",
            role="Weather forecasting and travel weather advisory specialist",
            model_provider=model_provider
        )
        
        # 🛠️ 终极硬编码清洗：如果不是真正的 Weather Key 格式，直接扔 None 强制绕过 Pydantic
        weather_key = os.getenv('OPENWEATHERMAP_API_KEY', "")
        if not weather_key or "WEATHER" in weather_key.upper() or "你的" in weather_key:
            weather_key = None
            
        self.weather_service = WeatherForecastTool(weather_key)
        
    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process weather-related tasks"""
        task_type = task.get("type", "forecast")
        
        if task_type == "forecast":
            return await self._get_weather_forecast(task)
        else:
            return await self._get_current_weather(task)
            
    async def _get_weather_forecast(self, task: Dict) -> Dict:
        """Get weather forecast for destination"""
        destination = task.get("destination")
        
        try:
            current_weather = await asyncio.to_thread(self.weather_service.get_current_weather, destination)
            forecast_weather = await asyncio.to_thread(self.weather_service.get_forecast_weather, destination)
            
            system_prompt = f"""You are a weather analysis specialist. Analyze this weather data for {destination} and provide:
            
            1. Current weather summary
            2. 5-day forecast overview
            3. Best days for outdoor activities
            4. What to pack recommendations
            5. Weather-based activity suggestions
            6. Any weather warnings or considerations
            
            Be practical and helpful for travelers."""
            
            weather_data = {
                "current": current_weather,
                "forecast": forecast_weather
            }
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Weather data for {destination}: {weather_data}")
            ]
            
            analysis = await self.llm.ainvoke(messages)
            
            return {
                "agent": self.name,
                "task_type": "weather_forecast",
                "destination": destination,
                "weather_analysis": analysis.content,
                "raw_weather_data": weather_data,
                "status": "completed"
            }
            
        except Exception as e:
            return {
                "agent": self.name,
                "task_type": "weather_forecast",
                "error": str(e),
                "status": "failed"
            }
            
    async def _get_current_weather(self, task: Dict) -> Dict:
        """Get current weather"""
        destination = task.get("destination")
        
        try:
            weather_data = await asyncio.to_thread(self.weather_service.get_current_weather, destination)
            
            if weather_data:
                temp = weather_data.get('main', {}).get('temp', 'N/A')
                desc = weather_data.get('weather', [{}])[0].get('description', 'N/A')
                analysis = f"Current weather in {destination}: {temp}°C, {desc}"
            else:
                analysis = f"Could not fetch weather for {destination}"
                
            return {
                "agent": self.name,
                "task_type": "current_weather",
                "destination": destination,
                "weather_analysis": analysis,
                "status": "completed"
            }
            
        except Exception as e:
            return {
                "agent": self.name,
                "task_type": "current_weather",
                "error": str(e),
                "status": "failed"
            }