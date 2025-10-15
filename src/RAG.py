import os
import glob
import logging
from pathlib import Path
from typing import Optional, List
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.readers.unstructured import UnstructuredReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import CohereRerank
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 全局配置 ---
Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

PERSIST_DIR = "./storage"

class AdvancedRAG:
    """
    高级 RAG 系统，支持多种文档格式和重排序
    
    特性：
    - 使用 UnstructuredReader 支持多种文档格式
    - 自动创建和持久化向量索引
    - 集成 Cohere 重排序提升检索质量
    """
    
    def __init__(
        self, 
        knowledge_dir: str,
        chunk_size: int = 512,
        chunk_overlap: int = 20,
        persist_dir: str = PERSIST_DIR
    ):
        """
        初始化 RAG 系统
        
        Args:
            knowledge_dir: 知识库文件夹路径
            chunk_size: 文档分块大小
            chunk_overlap: 分块重叠大小
            persist_dir: 索引持久化路径
        """
        self.knowledge_dir = Path(knowledge_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_dir = persist_dir
        
        # 验证配置
        self._validate_config()
        
        # 加载或构建索引
        self.index = self._load_or_build_index()

    def _validate_config(self):
        """验证配置和环境变量"""
        if not self.knowledge_dir.exists():
            raise ValueError(f"知识库目录不存在: {self.knowledge_dir}")
        
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("未设置 OPENAI_API_KEY 环境变量")
        
        logger.info(f"知识库目录: {self.knowledge_dir}")
        logger.info(f"索引存储目录: {self.persist_dir}")

    def _get_supported_files(self) -> List[Path]:
        """
        获取支持的文件列表
        
        Returns:
            支持的文件路径列表
        """
        # Unstructured 支持的常见文件格式
        supported_extensions = [
            "*.txt", "*.pdf", "*.docx", "*.doc", 
            "*.pptx", "*.xlsx", "*.html", "*.md",
            "*.csv", "*.json", "*.xml"
        ]
        
        file_paths = []
        for ext in supported_extensions:
            file_paths.extend(self.knowledge_dir.glob(ext))
        
        logger.info(f"找到 {len(file_paths)} 个支持的文件")
        return file_paths

    def _load_or_build_index(self) -> VectorStoreIndex:
        """
        如果索引已存在，则从本地加载；否则，使用 Unstructured 创建并持久化索引
        """
        if os.path.exists(self.persist_dir):
            try:
                logger.info("从存储加载索引...")
                storage_context = StorageContext.from_defaults(
                    persist_dir=self.persist_dir
                )
                index = load_index_from_storage(storage_context)
                logger.info("索引加载成功")
                return index
            except Exception as e:
                logger.warning(f"加载索引失败: {e}，将重新构建索引")
        
        return self._build_index()

    def _build_index(self) -> VectorStoreIndex:
        """构建新的向量索引"""
        logger.info("开始构建索引...")
        
        # 1. 获取支持的文件
        file_paths = self._get_supported_files()
        if not file_paths:
            raise ValueError(f"在 {self.knowledge_dir} 中未找到支持的文件")
        
        # 2. 使用 UnstructuredReader 加载文档
        reader = UnstructuredReader()
        documents = []
        
        for i, file_path in enumerate(file_paths, 1):
            try:
                logger.info(f"处理文件 ({i}/{len(file_paths)}): {file_path.name}")
                docs = reader.load_data(file=str(file_path))
                documents.extend(docs)
            except Exception as e:
                logger.error(f"处理文件 {file_path.name} 时出错: {e}")
                continue
        
        if not documents:
            raise ValueError("未能成功加载任何文档")
        
        logger.info(f"成功加载 {len(documents)} 个文档")
        
        # 3. 解析文档为节点 (分块)
        parser = SentenceSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        nodes = parser.get_nodes_from_documents(documents)
        logger.info(f"创建了 {len(nodes)} 个节点")
        
        # 4. 创建并持久化索引
        logger.info("创建向量索引...")
        index = VectorStoreIndex(nodes, show_progress=True)
        index.storage_context.persist(persist_dir=self.persist_dir)
        logger.info(f"索引已保存到 {self.persist_dir}")
        
        return index

    def retrieve_context(
        self, 
        query: str, 
        top_k: int = 5,
        rerank: bool = True,
        rerank_top_n: int = 3
    ) -> str:
        """
        检索相关上下文
        
        Args:
            query: 查询文本
            top_k: 初始检索的文档数量
            rerank: 是否使用重排序
            rerank_top_n: 重排序后保留的文档数量
            
        Returns:
            拼接的上下文文本
        """
        postprocessors = []
        
        # 如果启用重排序且有 Cohere API key
        if rerank:
            cohere_api_key = os.getenv("COHERE_API_KEY")
            if cohere_api_key:
                postprocessors.append(
                    CohereRerank(api_key=cohere_api_key, top_n=rerank_top_n)
                )
                logger.info(f"启用 Cohere 重排序 (top_n={rerank_top_n})")
            else:
                logger.warning("未设置 COHERE_API_KEY，跳过重排序")
        
        # 创建查询引擎
        query_engine = self.index.as_query_engine(
            similarity_top_k=top_k,
            node_postprocessors=postprocessors
        )
        
        # 执行查询
        logger.info(f"查询: {query}")
        response = query_engine.query(query)
        
        # 提取上下文
        context_list = [node.get_content() for node in response.source_nodes]
        logger.info(f"检索到 {len(context_list)} 个相关片段")
        
        return "\n---\n".join(context_list)

    def query(self, question: str, **kwargs) -> str:
        """
        端到端查询接口
        
        Args:
            question: 用户问题
            **kwargs: 传递给 retrieve_context 的参数
            
        Returns:
            答案文本
        """
        query_engine = self.index.as_query_engine(**kwargs)
        response = query_engine.query(question)
        return str(response)

    def rebuild_index(self):
        """强制重建索引"""
        import shutil
        if os.path.exists(self.persist_dir):
            logger.info(f"删除现有索引: {self.persist_dir}")
            shutil.rmtree(self.persist_dir)
        self.index = self._build_index()


# 使用示例
if __name__ == "__main__":
    # 初始化 RAG 系统
    rag = AdvancedRAG(knowledge_dir="./knowledge_base")
    
    # 检索上下文
    query = "什么是机器学习？"
    context = rag.retrieve_context(query, top_k=5, rerank=True)
    print("检索到的上下文:")
    print(context)
    
    # 或者直接获取答案
    answer = rag.query(query)
    print("\n答案:")
    print(answer)