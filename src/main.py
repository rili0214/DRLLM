import asyncio
import os
from dotenv import load_dotenv
from src.MCPClient import MCPClient
from src.Agent import Agent
from src.RAG import AdvancedRAG # <--- 引入新模块
from src.util import log_title

# 加载 .env 文件
load_dotenv()

OUT_PATH = os.path.join(os.getcwd(), 'output')
KNOWLEDGE_DIR = os.path.join(os.getcwd(), 'knowledge')
TASK = f"""
告诉我Antonette的信息,先从我给你的context中找到相关信息,总结后创作一个关于她的故事
把故事和她的基本信息保存到{OUT_PATH}/antonette.md,输出一个漂亮md文件
"""

async def main():
    # RAG (使用新的高级RAG模块)
    rag_system = AdvancedRAG(knowledge_dir=KNOWLEDGE_DIR)
    context = rag_system.retrieve_context(TASK)
    
    log_title('CONTEXT (from Advanced RAG)')
    print(context)

    # MCP Clients
    # 注意：LlamaIndex已经处理了文件读写，所以fileMCP可能不再需要用于RAG部分
    # 但我们仍然保留它，因为Agent可能需要它来执行写文件等任务
    fetch_mcp = MCPClient("mcp-server-fetch", "uvx", ['mcp-server-fetch'])
    file_mcp = MCPClient("mcp-server-file", "npx", ['-y', '@modelcontextprotocol/server-filesystem', OUT_PATH])

    # Agent
    agent = Agent('openai/gpt-4o-mini', [fetch_mcp, file_mcp], '', context)
    await agent.init()
    # 注意：在新的Agent.py实现中，invoke可能需要异步调用
    await agent.invoke(TASK) 
    await agent.close()


if __name__ == "__main__":
    # 如果您的Agent.py中的方法是异步的，请确保使用asyncio.run
    asyncio.run(main())