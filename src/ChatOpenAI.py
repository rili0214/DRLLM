import os
import openai  #type: ignore
from typing import List, Dict, Any
from src.util import log_title

class ToolCall:
    def __init__(self, id: str, function_name: str, function_arguments: str):
        self.id = id
        self.function = {
            "name": function_name,
            "arguments": function_arguments
        }

class ChatOpenAI:
    def __init__(self, model: str, system_prompt: str = '', tools: List[Dict[str, Any]] = [], context: str = ''):
        self.client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        self.model = model
        self.tools = tools
        self.messages: List[Dict[str, Any]] = []

        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
        if context:
            self.messages.append({"role": "user", "content": context})
            
    async def chat(self, prompt: str = None) -> Dict[str, Any]:
        log_title('CHAT')
        if prompt:
            self.messages.append({"role": "user", "content": prompt})

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            stream=True,
            tools=self._get_tools_definition(),
        )

        content = ""
        tool_calls_builder: Dict[int, Dict[str, Any]] = {}

        log_title('RESPONSE')
        async for chunk in stream:
            delta = chunk.choices[0].delta

            if delta.content:
                content_chunk = delta.content or ""
                content += content_chunk
                print(content_chunk, end='', flush=True)

            if delta.tool_calls:
                for tool_call_chunk in delta.tool_calls:
                    index = tool_call_chunk.index
                    if index not in tool_calls_builder:
                        tool_calls_builder[index] = {"id": "", "function": {"name": "", "arguments": ""}}
                    
                    current_call = tool_calls_builder[index]
                    if tool_call_chunk.id:
                        current_call["id"] += tool_call_chunk.id
                    if tool_call_chunk.function.name:
                        current_call["function"]["name"] += tool_call_chunk.function.name
                    if tool_call_chunk.function.arguments:
                        current_call["function"]["arguments"] += tool_call_chunk.function.arguments

        tool_calls = list(tool_calls_builder.values())
        self.messages.append({
            "role": "assistant", 
            "content": content, 
            "tool_calls": [{ "id": call["id"], "type": "function", "function": call["function"] } for call in tool_calls]
        })

        return {
            "content": content,
            "tool_calls": tool_calls
        }

    def append_tool_result(self, tool_call_id: str, tool_output: str):
        self.messages.append({
            "role": "tool",
            "content": tool_output,
            "tool_call_id": tool_call_id
        })

    def _get_tools_definition(self) -> List[Dict[str, Any]]:
        if not self.tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["inputSchema"],
                },
            }
            for tool in self.tools
        ]