import asyncio
from typing import List, Dict, Any
from mcp.client import Client
from mcp.client.stdio import StdioClientTransport
from mcp.protocol import Tool

class MCPClient:
    def __init__(self, name: str, command: str, args: List[str], version: str = "0.0.1"):
        self.mcp = Client(name=name, version=version)
        self.command = command
        self.args = args
        self.transport: StdioClientTransport = None
        self.tools: List[Tool] = []

    async def init(self):
        """
        连接到 MCP 服务器并获取可用工具列表
        """
        try:
            self.transport = StdioClientTransport(
                command=self.command,
                args=self.args,
            )
            await self.mcp.connect(self.transport)

            tools_result = await self.mcp.list_tools()
            self.tools = tools_result.tools
            
            print(f"Connected to MCP server '{self.mcp.name}' with tools: {[tool.name for tool in self.tools]}")

        except Exception as e:
            print(f"Failed to connect to MCP server '{self.mcp.name}': {e}")
            raise e

    async def close(self):
        """
        关闭与 MCP 服务器的连接
        """
        if self.mcp:
            await self.mcp.close()

    def get_tools(self) -> List[Dict[str, Any]]:
        """
        以 OpenAI 工具格式返回工具定义
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
            }
            for tool in self.tools
        ]

    async def call_tool(self, name: str, params: Dict[str, Any]) -> Any:
        """
        调用指定的工具
        """
        return await self.mcp.call_tool(name=name, arguments=params)