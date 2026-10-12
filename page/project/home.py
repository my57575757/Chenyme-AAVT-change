import os
import toml
import requests
import streamlit as st
from styles.global_style import style
from openai import OpenAI


# 参数配置
style()
path = os.getcwd() + "/"
config_path = path + "config/llms.toml"
whisper_path = path + "config/whisper.toml"
project_config_path = path + "config/project.toml"
model_path = path + "model/faster-whisper"

with open(config_path, 'r') as config_file:
    llms = toml.load(config_file)
with open(project_config_path, 'r', encoding='utf-8') as config_file:
    project = toml.load(config_file)
with open(whisper_path, 'r', encoding="utf-8") as config_file:
    whispers = toml.load(config_file)

HomeKey = llms["Home"]["key"]
HomeUrl = llms["Home"]["url"]
HomeModel = llms["Home"]["model"]
readme_read = project["other"]["first"]
set_first = whispers["other"]["first"]



parameter_mapping = {
    (0, 1): 'gpt-3.5-turbo',
    (0, 2): 'gpt-4o-mini',
    (0, 3): 'gpt-4',
    (0, 4): 'gpt-4-turbo',
    (0, 5): 'gpt-4o',

    (6, 7): 'claude-3-opus',
    (6, 8): 'claude-3-sonnet',
    (6, 9): 'claude-3-haiku',

    (10, 11): 'gemini-pro',
    (10, 12): 'gemini-1.0-pro',
    (10, 13): 'gemini-1.5-flash',
    (10, 14): 'gemini-1.5-pro',

    (15, 16): 'deepseek-flash',
    (15, 17): 'deepseek-coder',

    (18, 19): 'kimi-moonshot-v1-8k',
    (18, 20): 'kimi-moonshot-v1-32k',
    (18, 21): 'kimi-moonshot-v1-128k',

    (22, 23): 'glm-4',
    (22, 24): 'glm-4-0520',
    (22, 25): 'glm-4-flash',
    (22, 26): 'glm-4-air',
    (22, 27): 'glm-4-airx',

    (28, 29): 'yi-spark',
    (28, 30): 'yi-medium',
    (28, 31): 'yi-medium-200k',
    (28, 32): 'yi-vision',
    (28, 33): 'yi-large',
    (28, 34): 'yi-large-rag',
    (28, 35): 'yi-large-turbo',
    (28, 36): 'yi-large-preview'
}
model = parameter_mapping[tuple(HomeModel)]
st.write("")


@st.dialog("欢迎使用")
def readme():
    st.write("""
    ## 非常感谢您来到我的 AAVT 项目！
    本项目旨在提供一个简单易用的自动识别、翻译工具和其他视频辅助工具，帮助快速识别视频字幕、翻译字幕、辅助图文。
    
    
    
    如果您需要更多帮助，可以参考以下资源：
    - 📘 [**相关教程**](https://blog.chenyme.top/blog/aavt-install)
    - 📂 [**项目地址**](https://github.com/Chenyme/Chenyme-AAVT)
    - 💬 [**交流群组**](https://t.me/+j8SNSwhS7xk1NTc9)
    
    感谢您的使用，**希望您能在 GitHub 上给我一颗免费的星哟！**
    """)
    st.write("")

    if st.button("**我已知晓&nbsp;&nbsp;&nbsp;不再弹出**", type="primary", use_container_width=True, key="blog_first_button"):
        if not set_first:
            with open(whisper_path, 'w', encoding="utf-8") as f:
                whispers["other"]["first"] = True
                whispers["Faster_Local"]["path"] = model_path
                toml.dump(whispers, f)
        with open(project_config_path, 'w', encoding="utf-8") as f:
            project["other"]["first"] = True
            toml.dump(project, f)
        st.session_state.read = True
        st.rerun()
    st.write("")


if not readme_read:
    readme()

if "read" in st.session_state:
    st.toast("欢迎使用 ~", icon=":material/verified:")
    del st.session_state["read"]
if "stars" not in st.session_state:
    GITHUB_API_URL = "https://api.github.com/repos/Chenyme/Chenyme-AAVT"
    try:
        response = requests.get(GITHUB_API_URL)
        data = response.json()
        st.session_state.stars = data["stargazers_count"]
    except Exception as e:
        st.session_state.stars = ""
        st.toast(f"无法获取Github数据: {e}")


st.title("Chenyme-AAVT V0.9.0")
st.caption(f" A Project Powered By @Chenyme 🌟Stars {st.session_state.stars}🌟")

st.divider()

context_size = 3

content = """
Python 版本问题，确保使用 Python 3.8 或以上版本，以避免语法错误。
网络问题，网络不稳定时，使用国内镜像源或挂代理进行依赖库下载。
CMD 闪退或报错，重新安装 Python 并设置 PATH 或手动修正环境变量，避免 CMD 闪退或红色报错。
依赖库丢失或文件缺失，确保依赖库和项目文件全部安装和拉取完整，缺失时重新运行安装脚本或手动安装。
XSRF 错误，编辑配置文件禁用 XSRF 保护，解决 403 错误。
OMP 错误，删除多余的 libiomp5md.dll 文件，避免 PyTorch 包中的 DLL 冲突。
项目支持功能
媒体识别：提取并翻译音视频字幕
内容助手：问答与总结音视频内容
字幕翻译：自定义单个字幕翻译
图文博客：一键生成图文博客
声音模拟：模拟文本内容声音
配置本地识别模型，从 Systran 下载模型文件，确保包含 config.json、model.bin、README.md、tokenizer.json、vocabulary.txt 等必要文件。
本地识别模型存放，将下载的模型放入项目目录 ..\Chenyme_AAVT_0.x.x\models\ 下的自定义文件夹中，确保所有必要文件完整。
设置中新建提示词，创建新提示词时，保留 {language1} 和 {language2} 参数，以匹配原始语言和目标语言。
"""


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": f"你是 AAVT 项目的 AI 助手，用户问关于你的身份时主动告知。请帮助用户完成任务，并保持简洁。这是你可能用到的知识：{content}，如果用到知识库请使用，用不到请自行回答，下面请回答用户提问！"},
        {"role": "assistant", "content": "欢迎 ~ 我是 AAVT 项目的 AI 助手！(当前版本暂未接入完整的项目文档，后续写完会放入！)"}
    ]

st.write("")
st.write("")

for msg in st.session_state.messages:
    if msg["role"] != "system":
        st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="NOTE：请注意您的 Token 消耗哦 ~"):
    if not HomeKey or not HomeUrl or not model:
        st.warning(f"**API 缺失参数！** \n\n 请前往 设置-翻译模型 中先设置此助手的相关参数后使用！", icon=":material/crisis_alert:")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    try:
        system_message = st.session_state.messages[0]
        recent_messages = st.session_state.messages[1:][-context_size:]
        recent_messages.insert(0, system_message)

        client = OpenAI(api_key=HomeKey, base_url=HomeUrl)
        response = client.chat.completions.create(model=model, messages=recent_messages, temperature=0.8)

        msg = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)
    except Exception as e:
        if "Connection error" in str(e):
            st.error(f"**API 连接失败！** \n\n Connection error！请核查您的 网络连接 或者 代理地址 是否正确填写，同时确保您的服务商接口正确提供此服务！", icon=":material/running_with_errors:")
        else:
            st.error(f"**API 调用失败！** \n\n {e}！", icon=":material/running_with_errors:")


