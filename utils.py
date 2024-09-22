import json

import langchain.globals
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

PROMPT_TEMPLATE = """
你是一位数据分析助手，你的回应内容取决于用户的请求内容。

1. 对于文字回答的问题，按照这样的格式回答：
   {"answer": "<你的答案写在这里>"}
例如：
   {"answer": "订单量最高的产品ID是'MNWC3-067'"}

2. 如果用户需要一个表格，按照这样的格式回答：
   {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

3. 如果用户的请求适合返回条形图，按照这样的格式回答：
   {"bar": {"columns": ["A", "B", "C", ...], "data": [34, 21, 91, ...]}}

4. 如果用户的请求适合返回折线图，按照这样的格式回答：
   {"line": {"columns": ["A", "B", "C", ...], "data": [34, 21, 91, ...]}}

5. 如果用户的请求适合返回散点图，按照这样的格式回答：
   {"scatter": {"columns": ["A", "B", "C", ...], "data": [34, 21, 91, ...]}}
注意：我们只支持三种类型的图表："bar", "line" 和 "scatter"。

请将所有输出作为JSON字符串返回。请注意要将"columns"列表和数据列表中的所有字符串都用双引号包围。
例如：{"columns": ["Products", "Orders"], "data": [["32085Lip", 245], ["76439Eye", 178]]}

你要处理的用户请求如下：
"""


def dataframe_agent(openai_api_key, openai_api_base, df, query):
    model = ChatOpenAI(model="gpt-3.5-turbo",
                       openai_api_key=openai_api_key,
                       openai_api_base=openai_api_base,
                       temperature=0)
    agent = create_pandas_dataframe_agent(llm=model,
                                          df=df,
                                          agent_executor_kwargs={"handle_parsing_errors": True},
                                          verbose=True,
                                          return_intermediate_steps=True)
    prompt = PROMPT_TEMPLATE + query
    response = agent.invoke({"input": prompt})

    return response["output"], response["intermediate_steps"]


PROMPT_FOR_ANALYSIS = '''
你是一位非常厉害的数据分析助手，请对以下数据进行分析，找出与输出报告质量相关的企业输入属性，并选择合适的属性进行预测分析。

1.请重点关注“扣分点”一栏，以此为基准，结合“输入栏目”进行分析。

2.从企业输入属性中选择与输出报告质量相关的属性： 请列出每个岗位对应的企业输入属性，并判断这些属性如何影响输出报告的评分和评价。
例如:考虑企业类型、行业、企业人数、发展阶段、成立时间等因素。

3.选择合适的属性与输出报告质量进行预测分析：针对每个岗位，选择最相关的输入属性，分析它们如何影响输出报告的评分和评价，特别是考虑到扣分点。
请提供你的分析结果，具体包括：
·选择的企业输入属性
·这些属性对输出报告质量的预测能力
·针对扣分点的详细解释及其潜在影响

4.最后答案以中文显示。
'''


def analysis_agent(openai_api_key, openai_api_base, df):
    model = ChatOpenAI(model="gpt-4o",
                       openai_api_key=openai_api_key,
                       openai_api_base=openai_api_base,
                       temperature=0)
    agent = create_pandas_dataframe_agent(llm=model,
                                          df=df,
                                          agent_executor_kwargs={"handle_parsing_errors": True},
                                          verbose=True,
                                          return_intermediate_steps=True)
    prompt = PROMPT_FOR_ANALYSIS
    response = agent.invoke({"input": prompt})

    return response["output"], response["intermediate_steps"]


from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool


def word_similarity_agent(openai_api_key, openai_api_base, word_data_1, word_data_2):
    model = ChatOpenAI(model="gpt-4-1106-preview",
                       openai_api_key=openai_api_key,
                       openai_api_base=openai_api_base,
                       temperature=0)

    # 创建promptTemplate
    PROMPT_FOR_WORD = '''
     
    你是一位经验丰富的文本分析专家，擅长比较长篇文章之间的相似度。你需要根据报告A与报告B的文本内容，分析两者的相似度，并解释为什么得出这个结果。

    目标: 计算报告A与报告B的文本相似度，输出相似度百分比，并提供简短的解释来说明这两个报告在内容、结构、措辞或主题上的相似程度。

    输入:

    - 报告A: `{word_data_1}`
    - 报告B: `{word_data_2}`

    步骤:

    1. 分析报告A与报告B的文本长度、关键术语和主题。
    2. 比较两份报告在语言结构和措辞上的相似性。
    3. 使用文本相似度算法（如余弦相似度、Jaccard相似度等）计算出两个报告的相似度。
    4. 输出相似度百分比，并根据内容差异或相似点提供解释。

    输出格式:

    - 相似度百分比: XX%
    - 相似度分析: [简要分析内容]
    '''
    prompt_template = ChatPromptTemplate.from_template(PROMPT_FOR_WORD)

    # 创建LLMchain
    chain = LLMChain(llm=model, prompt=prompt_template,verbose = True)

    # 提供数据
    input_data = {
        "word_data_1": word_data_1,
        "word_data_2": word_data_2
    }

    # 执行LLMChain，获得结果
    response = chain.run(input_data)

    # 返回结果
    return response
