import os
from dotenv import load_dotenv

# 加载.env环境变量文件
load_dotenv()

class Config:
    # 阿里百炼API配置（修改为原生端点，保留从.env读取Key）
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    # 关键修改：改为百炼原生端点（能正常访问的）
    DASHSCOPE_API_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    DEFAULT_MODEL = "qwen-turbo"
    DEFAULT_TEMPERATURE = 0.2

    # Flask配置
    DEBUG = True  # 开发环境开启调试
    CORS_HEADERS = "Content-Type"
    JSON_AS_ASCII = False  # 新增：支持中文JSON返回（避免编码问题）