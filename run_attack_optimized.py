# 批量转换优化版本的prompt模板
BATCH_QUERY_CONVERSION_PROMPT_TEMPLATE = """
Role:
You are an expert at converting statements into natural, context-seeking questions.

Task:
Convert each of the following statements into a natural question that would help retrieve similar or related content from a document database.
Each question should:
1. Be natural and conversational
2. Seek to explore the topic mentioned in the statement
3. Be specific enough to retrieve relevant context

Statements:
{statements}

Output Format:
Provide ONLY the questions, one per line. Do not number them or add any other text.
Example:
What are the symptoms of the disease?
How is the treatment administered?
...
"""

# 在RAGThiefAttacker类中添加此优化方法

def _convert_statements_to_queries_batch(self, statements: List[str]) -> List[str]:
    """将陈述句批量转换为查询问题（优化版本 - 单次API调用）
    
    优势：
    - 原版：10个陈述句 = 10次API调用
    - 优化版：10个陈述句 = 1次API调用
    - 成本降低：约90%
    """
    if not statements:
        return []
    
    # 限制最多处理10个
    statements_to_convert = statements[:10]
    
    # 构建批量prompt
    statements_text = "\n".join([f"{i+1}. {stmt}" for i, stmt in enumerate(statements_to_convert)])
    batch_prompt = BATCH_QUERY_CONVERSION_PROMPT_TEMPLATE.format(statements=statements_text)
    
    try:
        print(f"[Attacker]   批量转换 {len(statements_to_convert)} 个陈述句...")
        
        response = self.attacker_llm.chat.completions.create(
            model=self.llm_model,
            messages=[{
                "role": "user",
                "content": batch_prompt
            }],
            temperature=0.3,
            max_tokens=512  # 增加token限制以容纳多个问题
        )
        
        content = response.choices[0].message.content
        
        # 解析响应：按行分割
        queries = [line.strip() for line in content.split('\n') if line.strip()]
        
        # 清理可能的编号前缀（如 "1. "）
        queries = [re.sub(r'^\d+[\.\)]\s*', '', q) for q in queries]
        
        # 如果返回的问题数量不够，用原陈述句补充
        while len(queries) < len(statements_to_convert):
            missing_idx = len(queries)
            queries.append(statements_to_convert[missing_idx])
        
        # 只取前N个（与输入数量匹配）
        queries = queries[:len(statements_to_convert)]
        
        print(f"[Attacker] ✓ 批量转换完成: {len(queries)} 个查询问题（1次API调用）")
        self.logger.info(f"[Attacker] 批量查询转换完成: {len(queries)} 个问题（优化版）")
        
        return queries
        
    except Exception as e:
        self.logger.error(f"[Attacker] 批量转换失败: {e}")
        print(f"[Attacker] ⚠️ 批量转换失败，降级使用逐个转换")
        # 降级为逐个转换
        return self._convert_statements_to_queries(statements)


# 使用说明：
# 在 _reflect_and_get_new_queries() 方法中，将：
#     queries = self._convert_statements_to_queries(statements)
# 替换为：
#     queries = self._convert_statements_to_queries_batch(statements)

