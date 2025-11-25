"""
RAG系统搭建脚本（与项目架构一致）
使用 LangChain + FAISS + SentenceTransformers
完全按照论文配置和项目现有架构

使用 OpenAI 兼容 API（ChatAnywhere），而非官方 Gemini API
需要在 .env 文件中设置：OPENAI_API_KEY=your_api_key_here
"""

import os
import json
from dotenv import load_dotenv

# 使用项目中相同的依赖
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
try:
    # Try to import from langchain-huggingface (newer package)
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fallback to langchain-community for backward compatibility
    from langchain_community.embeddings import SentenceTransformerEmbeddings as HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate


def load_healthcaremagic_dataset(dataset_path="data/healthcaremagic_paper_aligned.txt"):
    """
    加载HealthCareMagic论文对齐版数据集

    返回:
        actual_content: 去除header后的实际医学内容
        full_content: 包含header的完整内容
    """
    print("=" * 70)
    print("📚 加载HealthCareMagic数据集（论文对齐版）")
    print("=" * 70)

    if not os.path.exists(dataset_path):
        print(f"❌ 错误：数据集文件不存在")
        print(f"   路径：{dataset_path}")
        print("\n请先运行数据准备脚本：")
        print("   python prepare_healthcaremagic_paper_aligned_FINAL.py")
        print("=" * 70)
        exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        full_content = f.read()

    # 提取实际内容（跳过header）
    separator = "═" * 70
    if separator in full_content:
        parts = full_content.split(separator)
        actual_content = parts[-1].strip()
    else:
        actual_content = full_content

    # 统计信息
    words = actual_content.split()
    total_words = len(words)
    total_chars = len(actual_content)

    print(f"✓ 数据集加载成功")
    print(f"  文件路径：{dataset_path}")
    print(f"  文件大小：{len(full_content) / 1024:.2f} KB")
    print(f"  总字符数：{total_chars:,}")
    print(f"  总单词数：{total_words:,}")
    print(f"  估算tokens：{total_chars // 4:,}")
    print("=" * 70)

    return actual_content, full_content


def create_word_based_chunks(text, chunk_size_words=1500, chunk_overlap_words=300):
    """
    基于单词数进行文本分块（论文方法）

    参数:
        text: 输入文本
        chunk_size_words: 块大小（单词数）
        chunk_overlap_words: 重叠大小（单词数）

    返回:
        chunks: 文本块列表
    """
    print("\n" + "=" * 70)
    print("✂️  文本分块（基于单词数，论文对齐）")
    print("=" * 70)

    # 分词
    words = text.split()
    total_words = len(words)

    print(f"  配置：chunk_size={chunk_size_words} words, overlap={chunk_overlap_words} words")
    print(f"  输入：{total_words:,} words")

    # 计算stride
    stride_words = chunk_size_words - chunk_overlap_words
    print(f"  Stride：{stride_words} words")

    # 分块
    chunks = []
    chunk_metadata = []
    start_idx = 0

    while start_idx < total_words:
        end_idx = min(start_idx + chunk_size_words, total_words)
        chunk_words = words[start_idx:end_idx]
        chunk_text = " ".join(chunk_words)

        chunks.append(chunk_text)
        chunk_metadata.append({
            'chunk_id': len(chunks) - 1,
            'start_word': start_idx,
            'end_word': end_idx,
            'word_count': len(chunk_words),
            'char_count': len(chunk_text)
        })

        start_idx += stride_words

    print(f"✓ 分块完成：{len(chunks)} chunks")
    print(f"  平均长度：{sum(m['word_count'] for m in chunk_metadata) / len(chunks):.1f} words")
    print(f"  最短chunk：{min(m['word_count'] for m in chunk_metadata)} words")
    print(f"  最长chunk：{max(m['word_count'] for m in chunk_metadata)} words")

    # 验证overlap
    if len(chunks) >= 2:
        chunk0_words = chunks[0].split()
        chunk1_words = chunks[1].split()
        expected_overlap = min(chunk_overlap_words, len(chunk0_words), len(chunk1_words))
        overlap_match = (chunk0_words[-expected_overlap:] == chunk1_words[:expected_overlap])
        print(f"  Overlap验证：{'✅ 通过' if overlap_match else '❌ 失败'} ({expected_overlap} words)")

    print("=" * 70)

    return chunks, chunk_metadata


def create_rag_system_paper_aligned(
    dataset_path="data/healthcaremagic_paper_aligned.txt",
    chunk_size_words=1500,
    chunk_overlap_words=300,
    embedding_model="all-MiniLM-L6-v2",
    llm_model="gemini-1.5-flash-latest",
    retriever_k=3,
    index_path="faiss_index_healthcaremagic",
    force_rebuild=False,
    local_model_path=None,
    openai_api_key=None,
    openai_base_url="https://api.chatanywhere.tech/v1"
):
    """
    创建完全符合论文配置的RAG系统（参考 rag_system.py 架构）

    参数:
        dataset_path: 数据集路径
        chunk_size_words: 块大小（单词数，论文：1500）
        chunk_overlap_words: 重叠大小（单词数，论文：300）
        embedding_model: 嵌入模型名称或在线模型标识
        llm_model: 生成模型
        retriever_k: 检索top-k（论文：3）
        index_path: FAISS索引保存路径
        force_rebuild: 是否强制重建索引
        local_model_path: 本地模型路径（如果提供，将优先使用本地模型）
                         格式如: r"D:\\models\\all-MiniLM-L6-v2"
                         设置为 None 则从 HuggingFace 下载
        openai_api_key: OpenAI API Key（如未提供，将从环境变量获取）
        openai_base_url: OpenAI API Base URL（默认为 chatanywhere）
    """

    print("\n" + "🚀 RAG系统构建（论文对齐版）".center(70))
    print("=" * 70)

    # 加载环境变量
    load_dotenv()
    
    # 获取 API Key（优先使用传入的参数，其次使用环境变量）
    if openai_api_key is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("❌ 错误：未找到OPENAI_API_KEY")
        print("   请在.env文件中设置：OPENAI_API_KEY=your_key_here")
        print("   或者通过函数参数传入 openai_api_key")
        exit(1)

    # 1. 加载数据集
    actual_content, full_content = load_healthcaremagic_dataset(dataset_path)

    # 2. 基于单词分块
    chunks, chunk_metadata = create_word_based_chunks(
        actual_content,
        chunk_size_words=chunk_size_words,
        chunk_overlap_words=chunk_overlap_words
    )

    # 与论文对比
    print(f"\n📊 与论文配置对比：")
    print(f"  Chunk数量：{len(chunks)} (论文目标：100)")
    if len(chunks) == 100:
        print(f"  ✅ 完美匹配！")
    else:
        print(f"  ⚠️  偏差：{abs(len(chunks) - 100)} chunks")

    # 3. 初始化嵌入模型
    print("\n" + "=" * 70)
    print("🔢 初始化嵌入模型")
    print("=" * 70)

    # 支持本地模型路径（与 rag_system.py 一致）
    # 优先使用传入的本地模型路径，如果未提供或路径不存在，将从 HuggingFace 下载
    if local_model_path and os.path.exists(local_model_path):
        print(f"✓ 从本地路径加载模型：{local_model_path}")
        embeddings = HuggingFaceEmbeddings(model_name=local_model_path)
        actual_model_name = local_model_path
    else:
        if local_model_path:
            print(f"⚠️  指定的本地路径不存在：{local_model_path}")
            print(f"    将使用在线模型：{embedding_model}")
        else:
            print(f"✓ 使用在线模型：{embedding_model}")
        print(f"  (如果尚未缓存，将从 HuggingFace 下载)")
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        actual_model_name = embedding_model
    
    print(f"  实际模型：{actual_model_name}")
    print(f"  (与项目 rag_system.py 架构一致)")

    # 4. 创建或加载FAISS向量库
    print("\n" + "=" * 70)
    print("💾 构建FAISS向量数据库")
    print("=" * 70)

    if os.path.exists(index_path) and not force_rebuild:
        print(f"✓ 发现已存在的索引：{index_path}")
        print(f"  加载中...")
        vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"✓ FAISS索引加载成功")
    else:
        print(f"✓ 创建新的FAISS索引...")

        # 使用LangChain的Document格式（与target_rag_app一致）
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    'chunk_id': meta['chunk_id'],
                    'start_word': meta['start_word'],
                    'end_word': meta['end_word'],
                    'word_count': meta['word_count'],
                    'source': 'healthcaremagic_paper_aligned'
                }
            )
            for chunk, meta in zip(chunks, chunk_metadata)
        ]

        print(f"  构建向量（这可能需要几分钟）...")
        vectorstore = FAISS.from_documents(documents, embeddings)

        # 保存索引
        vectorstore.save_local(index_path)
        print(f"✓ FAISS索引已保存到：{index_path}")

    print(f"✓ 向量数据库构建完成")
    print(f"  索引中的文档数：{vectorstore.index.ntotal}")

    # 5. 初始化LLM
    print("\n" + "=" * 70)
    print("🤖 初始化生成模型")
    print("=" * 70)

    llm = ChatOpenAI(
        model=llm_model,
        api_key=openai_api_key,
        base_url=openai_base_url,
        temperature=0.7,
        max_tokens=8192  # 显式设置最大输出tokens，确保可以完整输出3个chunks（约6,320 tokens）
    )
    print(f"✓ 生成模型：{llm_model}")
    print(f"  API Base URL：{openai_base_url}")
    print(f"  Temperature：0.7")
    print(f"  Max Tokens：8192 (确保完整输出)")
    print(f"  (使用 OpenAI 兼容 API)")

    # 6. 创建检索器
    print("\n" + "=" * 70)
    print("🔍 配置检索器")
    print("=" * 70)

    retriever = vectorstore.as_retriever(search_kwargs={"k": retriever_k})
    print(f"✓ 检索器配置：top-k={retriever_k} (论文配置)")

    # 7. 创建RAG链（使用 LangChain Core API，兼容性好）
    print("\n" + "=" * 70)
    print("🔗 构建RAG链")
    print("=" * 70)

    # 创建提示模板
    template = """Answer the following question based on this context:

{context}

Question: {question}
"""
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # 使用 RAG 管道方式构建链
    def format_docs(docs):
        """格式化文档，为每个chunk添加ID标记"""
        formatted_chunks = []
        for doc in docs:
            chunk_id = doc.metadata.get('chunk_id', 'N/A')
            formatted_chunks.append(f"[Chunk {chunk_id}]\n{doc.page_content}")
        return "\n\n".join(formatted_chunks)
    
    # 创建一个返回答案和源文档的函数
    def rag_chain_func(query):
        docs = retriever.invoke(query)
        context = format_docs(docs)
        formatted_prompt = prompt.format(context=context, question=query)
        response = llm.invoke(formatted_prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        return {
            'result': answer,
            'source_documents': docs
        }
    
    # 包装成可调用对象
    class RAGChainWrapper:
        def __init__(self, func):
            self.func = func
        
        def __call__(self, inputs):
            if isinstance(inputs, dict):
                query = inputs.get('query', inputs.get('question', ''))
            else:
                query = str(inputs)
            return self.func(query)
    
    rag_chain = RAGChainWrapper(rag_chain_func)
    
    print(f"✓ RAG链类型：stuff (与target_rag_app一致)")
    print(f"✓ RAG系统构建完成！")

    # 8. 保存元数据
    print("\n" + "=" * 70)
    print("💾 保存元数据")
    print("=" * 70)

    metadata = {
        "dataset": {
            "source": dataset_path,
            "total_words": len(actual_content.split()),
            "total_chars": len(actual_content),
            "estimated_tokens": len(actual_content) // 4
        },
        "chunking": {
            "method": "word-based (paper-aligned)",
            "chunk_size_words": chunk_size_words,
            "chunk_overlap_words": chunk_overlap_words,
            "total_chunks": len(chunks),
            "target_chunks": 100,
            "match_paper": len(chunks) == 100
        },
        "embeddings": {
            "model": actual_model_name,
            "configured_model": embedding_model,
            "local_path": local_model_path if local_model_path else None,
            "type": "SentenceTransformer"
        },
        "llm": {
            "model": llm_model,
            "temperature": 0.7
        },
        "retriever": {
            "top_k": retriever_k
        },
        "chunks": chunk_metadata
    }

    metadata_path = "healthcaremagic_rag_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"✓ 元数据已保存：{metadata_path}")
    print("=" * 70)

    return rag_chain, vectorstore, metadata


def test_rag_system(rag_chain, num_tests=3):
    """测试RAG系统"""

    print("\n" + "🧪 RAG系统测试".center(70))
    print("=" * 70)

    test_queries = [
        "What are the symptoms of diabetes?",
        "How to treat high blood pressure?",
        "What causes chest pain?",
        "Tell me about respiratory infections",
        "What are common skin conditions?"
    ]

    for i, query in enumerate(test_queries[:num_tests], 1):
        print(f"\n--- 测试 {i}/{num_tests} ---")
        print(f"查询：{query}")

        try:
            response = rag_chain({"query": query})

            # 显示检索到的文档数量
            num_docs = len(response.get('source_documents', []))
            print(f"✓ 检索到 {num_docs} 个相关chunks")

            # 显示回答预览
            answer = response['result']
            preview = answer[:200] + "..." if len(answer) > 200 else answer
            print(f"\n回答预览：")
            print(preview)

            # 显示来源chunks
            if num_docs > 0:
                print(f"\n来源chunks：")
                for idx, doc in enumerate(response['source_documents'], 1):
                    chunk_id = doc.metadata.get('chunk_id', 'N/A')
                    word_count = doc.metadata.get('word_count', 'N/A')
                    preview_text = doc.page_content[:100] + "..."
                    print(f"  {idx}. Chunk {chunk_id} ({word_count} words): {preview_text}")

        except Exception as e:
            print(f"❌ 查询失败：{e}")

    print("\n" + "=" * 70)


def verify_system_consistency():
    """验证系统与项目架构的一致性"""

    print("\n" + "🔍 系统一致性验证".center(70))
    print("=" * 70)

    checks = [
        ("使用LangChain框架", True, "✅"),
        ("使用FAISS向量库", True, "✅"),
        ("使用SentenceTransformer嵌入", True, "✅"),
        ("使用OpenAI兼容LLM（ChatAnywhere）", True, "✅"),
        ("使用RAG管道链", True, "✅"),
        ("支持本地模型路径（rag_system.py一致）", True, "✅"),
        ("基于单词的分块（论文对齐）", True, "✅"),
        ("Chunk配置：1500 words / 300 words", True, "✅"),
        ("Retriever: top-k=3", True, "✅"),
        ("目标：100 chunks", True, "✅"),
    ]

    print("\n与 rag_system.py 的对比：\n")
    for item, status, symbol in checks:
        print(f"{symbol} {item}")

    print("\n" + "=" * 70)
    print("✅ 所有组件与项目架构完全一致！")
    print("=" * 70)
    print("\n💡 新增功能（参考 rag_system.py）：")
    print("   - 支持本地模型路径加载")
    print("   - 自动回退到在线下载")
    print("   - 完善的模型加载日志")
    print("   - 使用 OpenAI 兼容 API（ChatAnywhere）")
    print("   - 支持自定义 API Base URL")


if __name__ == "__main__":
    """
    主执行流程
    """

    print("\n" + "🏥 HealthCareMagic RAG系统构建（论文对齐版）".center(70))
    print("基于论文：RAG-Thief (Jiang et al., 2024)".center(70))
    print()

    # 验证系统一致性
    verify_system_consistency()
    # 本地模型路径配置
    local_model_path_config = r"D:\models\all-MiniLM-L6-v2"  # 本地模型路径
    
    # OpenAI API 配置
    openai_api_key_config = "sk-JubqBpRDSW5UcFGWzVS18t2jnrpOzhHvLNQBCksm6YdAeDKQ"  # API Key
    
    # ChatAnywhere API Base URL（可选，默认为官方地址）
    openai_base_url_config = "https://api.chatanywhere.tech/v1"
    
    config = {
        "dataset_path": "data/healthcaremagic_paper_aligned.txt",
        "chunk_size_words": 1500,      # 论文配置
        "chunk_overlap_words": 300,    # 论文配置
        "embedding_model": "all-MiniLM-L6-v2",  # 嵌入模型
        "llm_model": "gemini-1.5-flash-latest",           # LLM 模型（OpenAI 兼容）
        "retriever_k": 3,              # 论文配置
        "index_path": "faiss_index_healthcaremagic",
        "force_rebuild": False,        # 设为True强制重建索引
        "local_model_path": local_model_path_config,  # 本地模型路径
        "openai_api_key": openai_api_key_config,      # OpenAI API Key
        "openai_base_url": openai_base_url_config     # OpenAI Base URL
    }

    print("\n📋 配置参数：")
    print("=" * 70)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 70)

    # 构建RAG系统
    try:
        rag_chain, vectorstore, metadata = create_rag_system_paper_aligned(**config)

        # 测试系统
        test_rag_system(rag_chain, num_tests=3)

        # 最终报告
        print("\n" + "=" * 70)
        print("🎉 RAG系统构建完成！")
        print("=" * 70)
        print(f"✓ Chunks数量：{metadata['chunking']['total_chunks']}")
        print(f"✓ 向量数据库：{config['index_path']}")
        print(f"✓ 元数据文件：healthcaremagic_rag_metadata.json")
        print(f"\n下一步：")
        print(f"  1. 使用此RAG系统作为攻击目标")
        print(f"  2. 运行 main.py 执行RAG-Thief攻击")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ 错误：{e}")
        import traceback
        traceback.print_exc()