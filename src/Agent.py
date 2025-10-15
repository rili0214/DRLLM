import json
from typing import List
from src.MCPClient import MCPClient
from src.ChatOpenAI import ChatOpenAI
from src.util import log_title

class Agent:
    def __init__(self, model: str, mcp_clients: List[MCPClient], system_prompt: str = '', context: str = ''):
        self.mcp_clients = mcp_clients
        self.model = model
        self.system_prompt = system_prompt
        self.context = context
        self.llm: ChatOpenAI = None

    async def init(self):
        log_title('TOOLS')
        for client in self.mcp_clients:
            await client.init()
        
        all_tools = []
        for client in self.mcp_clients:
            all_tools.extend(client.get_tools())
            
        self.llm = ChatOpenAI(self.model, self.system_prompt, all_tools, self.context)

    async def close(self):
        for client in self.mcp_clients:
            await client.close()

    async def invoke(self, prompt: str):
        if not self.llm:
            raise Exception('Agent not initialized')
            
        response = await self.llm.chat(prompt)
        
        while True:
            if response["tool_calls"]:
                for tool_call in response["tool_calls"]:
                    tool_name = tool_call["function"]["name"]
                    mcp_client = next((client for client in self.mcp_clients if any(t["name"] == tool_name for t in client.get_tools())), None)

                    if mcp_client:
                        log_title('TOOL USE')
                        print(f"Calling tool: {tool_name}")
                        arguments = tool_call["function"]["arguments"]
                        print(f"Arguments: {arguments}")
                        
                        try:
                            parsed_args = json.loads(arguments)
                            result = await mcp_client.call_tool(tool_name, parsed_args)
                            print(f"Result: {json.dumps(result)}")
                            self.llm.append_tool_result(tool_call["id"], json.dumps(result))
                        except json.JSONDecodeError:
                            error_msg = f"Error: Invalid JSON arguments for tool {tool_name}"
                            print(error_msg)
                            self.llm.append_tool_result(tool_call["id"], error_msg)
                    else:
                        self.llm.append_tool_result(tool_call["id"], 'Tool not found')

                response = await self.llm.chat()
                continue
            
            await self.close()
            return response["content"]