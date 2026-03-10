import os
import toml
import streamlit as st
from utils.get_font import get_font_data
from styles.global_style import style

style(True)
get_font_data()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 修复OMP

path = os.getcwd() + "/"
project_config_path = path + "config/project.toml"

with open(project_config_path, 'r', encoding='utf-8') as config_file:
    project = toml.load(config_file)
open_protect = project["protect"]["open"]
password_protect = project["protect"]["password"]


if not open_protect:
    st.session_state.verify = True

if "verify" in st.session_state:
    # 使用传统导航方式（兼容 Streamlit 1.37.1）
    page_names = ["主页", "设置", "音频识别", "视频识别", "字幕翻译", "声音克隆", "辅助工具"]
    
    if "current_page" not in st.session_state:
        st.session_state.current_page = "主页"
    
    with st.sidebar:
        st.title("导航")
        selected = st.radio("选择页面", page_names, index=page_names.index(st.session_state.current_page))
        st.session_state.current_page = selected
    
    # 执行选中的页面
    page_files = {
        "主页": "page/project/home.py",
        "设置": "page/setting/set.py",
        "音频识别": "page/project/audio.py",
        "视频识别": "page/project/video.py",
        "字幕翻译": "page/project/translate.py",
        "声音克隆": "page/tests/test.py",
        "辅助工具": "page/tests/tools.py",
    }
    
    import runpy
    runpy.run_path(page_files[selected], run_name="__main__")
else:
    st.write("# ")
    st.markdown("<h1 style='text-align: center;'>Chenyme-AAVT V0.9.0</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: grey; font-size: 12px;'>&copy; 2024 Powered By @Chenyme</p>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col2:
        st.write("")
        st.write("")
        st.info("##### 请登录 \n\n 请验证密码后使用本项目！", icon=":material/lock:")
        st.write("")
        st.write("")
        password_protect2 = st.text_input("mm", placeholder="验证密码", label_visibility="collapsed")
        st.write("")
        st.write("")
        if st.button("**验证**", type="primary", use_container_width=True):
            if password_protect2 == password_protect:
                st.session_state.verify = True
                st.rerun()
            else:
                st.toast("**错误！** \n\n 请重新输入您的密码！", icon=":material/lock:")
