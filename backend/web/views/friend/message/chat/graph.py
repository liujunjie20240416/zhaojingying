import os
from time import localtime
from typing import TypedDict, Annotated, Sequence

import lancedb
from django.utils.timezone import now,localtime
from lancedb import query
from langchain_community.vectorstores import LanceDB
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import ToolNode

from web.documents.utils.custom_embeddings import CustomEmbeddings


class ChatGraph:
    @staticmethod
    def create_app():
        @tool
        def get_time()->str:
            """当需要查询精确时间时，调用此函数。返回格式为：[年-月-日 时:分:秒]"""
            return localtime(now()).strftime('%Y-%m-%d %H:%M:%S')
        @tool
        def search_knowledge_base(query:str)->str:
            '''当用户查询阿里云百炼平台的相关信息时，调用此函数。输入为要查询的问题，输出为查询结果。'''
            db = lancedb.connect('./web/documents/lancedb_storage')
            embeddings = CustomEmbeddings()
            vector_db = LanceDB(
                connection=db,
                embedding = embeddings,
                table_name = 'my_knowledge_base'
            )
            docs = vector_db.similarity_search(query,k=3)
            context='\n\n'.join([f'内容片段:{i+1}\n{doc.page_content}' for i,doc in enumerate(docs)])
            return f'从知识库中找到一下相关信息:\n\n{context}\n'

        def search_jingying(query:str)->str:
            """
    当用户正在与“晶莹”，“大白鹅”或其女友/前女友相关对象对话时，调用此工具。

    使用场景：
    - 对话对象名字中包含“晶莹”
    - 或当前对话设定为“女友 / 前女友”
    - 用户希望模拟特定人物（晶莹）的说话风格、语气或记忆

    功能说明：
    - 从本地向量数据库中检索与 query 最相关的内容（前3条）
    - 数据库存储的是和“晶莹”的聊天记录，语义中包含相关的对话风格、语气、记忆、历史信息
    - 返回内容用于帮助模型“模仿她的说话方式和表达习惯”
    如果用户询问'我们以前的事'、'我的喜好'或需要进行深度的情感互动，请使用此工具获取事实依据。

    使用规则：
    - 不要直接逐字复述检索结果
    - 应基于检索内容进行“人格模仿式重写”
    - 回答必须符合“虚拟女友”的语气（自然、温柔、带情绪）
    - 回答应像真实聊天，而不是知识库摘要

    输出要求：
    - 不要使用括号描写动作（例如：（轻轻握住你的手））
    - 不要写心理活动或舞台说明
    - 只输出自然对话内容

    参数：
    - query: 用户当前输入或对话内容，用于语义检索

    返回：
    - 与 query 相关的上下文信息（供模型参考，不是最终回答）

    注意：
    - 该工具返回的是“记忆与语料”，而不是最终回复
    - 模型需要基于这些信息生成符合角色设定的自然语言回复
    """
            db = lancedb.connect('./web/documents/lancedb_storage')
            embeddings = CustomEmbeddings()
            vector_db = LanceDB(
                connection=db,
                embedding = embeddings,
                table_name = 'jingying'
            )
            docs = vector_db.similarity_search(query,k=3)
            context = '\n\n'.join([
                f"【记忆片段 {i + 1}】 时间: {doc.metadata.get('time', '未知时间')}\n内容: {doc.page_content}"
                for i, doc in enumerate(docs)
            ])
            return f"从你的记忆库中找到了关于晶莹的以下相关片段：请注意：- 只用自然语言回复 - 不要添加任何括号动作描写'''\n\n{context}\n"
            #context='\n\n'.join([f'内容片段:{i+1}\n{doc.page_content}' for i,doc in enumerate(docs)])
            #return f'从知识库中找到一下相关信息:\n\n{context}\n'

        tools = [get_time,search_knowledge_base,search_jingying]

        llm = ChatOpenAI(
            model = 'deepseek-v3.2',
            openai_api_key = os.getenv('API_KEY'),
            openai_api_base = os.getenv('API_BASE'),
            streaming=True,
            model_kwargs={
                'stream_options':{
                    'include_usage':True,
                }
            }
        ).bind_tools(tools)

        class AgentState(TypedDict):
            messages: Annotated[Sequence[BaseMessage], add_messages]

        def model_call(state:AgentState)->AgentState:
            from pprint import pprint
            pprint(state['messages'])
            res = llm.invoke(state['messages'])
            return {'messages':[res]}

        def should_continue(state:AgentState)->str:
            last_message = state['messages'][-1]
            if last_message.tool_calls:
                return 'tools'
            return 'end'

        tool_node = ToolNode(tools)


        graph = StateGraph(AgentState)
        graph.add_node('agent',model_call)
        graph.add_node('tools',tool_node)

        graph.add_edge(START,'agent')
        graph.add_conditional_edges('agent',should_continue,{
            'tools':'tools',
            'end':END,
        })
        graph.add_edge('tools','agent')

        return graph.compile()