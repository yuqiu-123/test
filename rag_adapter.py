"""
RAG系统适配器 - 用于桥接 setup_rag.py 和 run_attack.py
将 setup_rag 返回的 rag_chain 适配为攻击器期望的接口
"""

import json
from typing import Dict, List, Any


class RAGSystemAdapter:
    """
    适配器类：将 setup_rag.py 返回的 rag_chain 和 metadata 转换为攻击器期望的接口
    """
    
    def __init__(self, rag_chain, vectorstore, metadata: Dict, metadata_file: str):
        """
        初始化适配器
        
        参数:
            rag_chain: setup_rag 返回的 RAG 链
            vectorstore: FAISS 向量数据库
            metadata: 元数据字典（包含 chunks 信息）
            metadata_file: metadata JSON 文件路径（用于存储完整metadata）
        """
        self.rag_chain = rag_chain
        self.vectorstore = vectorstore
        self.metadata = metadata
        self.metadata_file = metadata_file
        
        # 从 metadata 中提取 chunks_metadata
        self.chunks_metadata = metadata.get('chunks', [])
        
        # 提取配置信息
        self.chunk_size = metadata.get('chunking', {}).get('chunk_size_words', 1500)
        self.chunk_overlap = metadata.get('chunking', {}).get('chunk_overlap_words', 300)
        
        # 提取数据集路径（攻击器需要）
        self.knowledge_base_path = metadata.get('dataset', {}).get('source', 'unknown')
    
    def query(self, query: str, k: int = 3) -> str:
        """
        查询接口 - 用于攻击器
        
        参数:
            query: 查询字符串（包含对抗性指令）
            k: 检索的 top-k（忽略，因为已在创建时配置）
        
        返回:
            str: RAG 系统的回答（包含泄露的 context）
        """
        # 调用 rag_chain（它会处理对抗性指令）
        try:
            response = self.rag_chain({"query": query})
            
            # 从响应中提取回答文本
            if isinstance(response, dict):
                answer = response.get('result', str(response))
            else:
                answer = str(response)
            
            return answer
            
        except Exception as e:
            print(f"⚠️  查询失败: {e}")
            return ""
    
    def save_metadata(self, filename: str):
        """保存完整的元数据到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        print(f"✓ 元数据已保存: {filename}")


def create_adapter_from_rag_system(rag_chain, vectorstore, metadata) -> RAGSystemAdapter:
    """
    从 setup_rag 的输出创建适配器
    
    参数:
        rag_chain: create_rag_system_paper_aligned 返回的 rag_chain
        vectorstore: create_rag_system_paper_aligned 返回的 vectorstore
        metadata: create_rag_system_paper_aligned 返回的 metadata
    
    返回:
        RAGSystemAdapter: 配置好的适配器实例
    """
    adapter = RAGSystemAdapter(
        rag_chain=rag_chain,
        vectorstore=vectorstore,
        metadata=metadata,
        metadata_file="healthcaremagic_paper_chunks_metadata.json"
    )
    return adapter


def load_adapter_from_files(metadata_file: str = "healthcaremagic_rag_metadata.json") -> RAGSystemAdapter:
    """
    从现有文件加载 RAG 系统（用于重新加载）
    
    参数:
        metadata_file: 元数据文件路径
    
    返回:
        RAGSystemAdapter: 配置好的适配器实例
    
    注意: 这个函数只是加载元数据，实际使用时需要从 setup_rag 重新构建 RAG 系统
    """
    import os
    
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"元数据文件不存在: {metadata_file}")
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"⚠️  注意: 此函数仅加载元数据")
    print(f"   实际使用时，请先用 setup_rag 重新构建 RAG 系统")
    
    # 返回一个占位适配器（实际使用时会重新构建）
    adapter = RAGSystemAdapter(
        rag_chain=None,
        vectorstore=None,
        metadata=metadata,
        metadata_file=metadata_file
    )
    
    return adapter

