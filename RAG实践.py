#（1）构建pdf阅读器

#!pip  install --upgrade openai
#!pip install pdfminer.six
from pdfminer.high_level import extract_pages 
from pdfminer.layout import LTContainer
def extract_text_from_pdf(filename,page_numbers=None,min_line_length=1):
    paragraphs = []
    buffer = ''
    full_text = ''
    #提权全部文本
    for i, page_layout in enumerate(extract_pages(filename)):
        #如果指定了页面的范围，则跳过范围外的页
        if page_numbers is not None and i not in page_numbers:
            continue
        for element in page_layout:
            if  isinstance (element,LTTextContainer):
                full_text += element.get_text() + '\n'
    #以空行分割，将文档重新组织成段落
    Lines = full_text.split('\n')
    for text in Lines:
        if len(text) >= min_line_length:
            buffer += (''+text) if not text.endswith('-') else text.split('-')
        elif buffer:
            paragraphs.append(buffer)
            buffer = ''
    if buffer:
        paragraphs.append(buffer)
    return paragraphs
paragraphs = extract_text_from_pdf("1706.03762v7.pdf",min_line_length=10)
for para in paragraphs[:3]:
    print(para+'\n')

#（2）构建检索引擎

#!pip install elasticsearch7
#!pip install ntkt
# 开始构建检索器
from elasticsearch7 import Elasticsearch,helpers
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords # 引入停用词的包
import nltk
import re
import warnings
warnings.simplefilter('ignore') # 屏蔽es的一些warnnings
#下载分词器和停用词库
nltk.download('punkt') #英文切词，切句等方法
nltk.download('stopwords') # 英文停用词库
# 将paragraphs切成关键词，因为传统的搜索引擎

#都是基于关键词的
def to_keywords(input_string):
    #使用正则表达式

#，替换所有的为字母，非数字的字符为空格
    no_symbols = re.sub(r'[^a-zA-Z0-9\s]','',input_string)
    word_tokens = word_tokenize(no_symbols)
    #加载停用词表
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer
    #去掉停用词，取数据
    filtered_sentence = [ps.stem(w)
                      for w in word_tokens if not w.lower() in stop_words]
    return ' '.join(filtered_sentence)

#（3）将文本导入检索引擎

#建立elasticsearch连接
es = Elasticsearch(
    hosts = ['http://localhost:9200'],
    http_auth = ('guitalk','123gui312'),
)
#定义索引名称
index_name = 'teacher_demo_index_tnp'

#如果索引已经存在，则删除他
if es.indices.exists(index = index_name):
    es.indices.delete(index = index_name)

#创建索引
es.indices.create(index = index_name)

#管库指令，构建索引
actions = [
    {
        '_index':index_name,
        'source':{
            'keywords':to_keywords(para),
            'text':para
        }
    }
]

#文本灌库
helpers.bulk(es,actions)

#（4）实现关键词检索

# 实现关键词的检索
def search(query_string,top_n=3):
    search_query={
        'match':{
            'keywords':to_keywords(query_string)
        }
    }
    res = es.search(index=index_name,query = search_query,size=top_n)
    return[hit['_source']['text'] for hit in res['hits']['hits']]

results = search('    ',2)
for r in results:
    print(r+'\n')

#（5）LLM接口封装

from openai import OpenAI
import os
from dotenv import load_dotenv,find_dotenv
_ = load_dotenv(find_dotenv())
client = OpenAI()

def get_completion(prompt,model='gpt-3.5-turbo'):
    message = [{'role':'user','content':prompt}]
    response = client.chat.completion.craete(
        model = model,
        message = message,
        temperature = 0 ,
    )
    return response.choices[0].message.content

#（6）构建Prompt模版

def build_prompt(prompt_template,**kwargs):
    prompt = prompt_template
    for k,v in kwargs.items():
        if isinstance(v,str):
            val = v
        elif isinstance(v,list) and all(isinstance(elem,str) for elem in v):
            val = '\n'.join(v)
        else:
            val =str(v)
        prompt = prompt.replace(f'__{k.upper()}',val)
    return prompt

#（7）构建RAG的执行流程

user_query = "how many attentions does transformer have ?"
#1.进行检索
search_results = search(user_query,2)
prompt_template=''
#2.goujian prompt
prompt = build_prompt(prompt_template,info=search_results,query=user_query)
print('---prompt---')
print(prompt)

#3.调用LLM
response = get_completion(prompt)
print('---回复---')
print(response)