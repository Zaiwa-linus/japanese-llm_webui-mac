import streamlit as st
from Script import generate, models
import mlx.core as mx

# モデルとトークナイザのロード状態を保持するためのセッション状態の初期化
if 'model_loaded' not in st.session_state:
    st.session_state['model_loaded'] = False

# サイドバーにロードボタンの設定
with st.sidebar:
    gguf = 'mistral-7b-v0.1.Q8_0.gguf'
    repo = 'TheBloke/Mistral-7B-v0.1-GGUF'
    load_button = st.button("Load Model")
    if load_button:
        st.session_state['model_loaded'] = False  # ロード前にフラグをリセット
        with st.spinner('Model loading...'):
            mx.random.seed(100)  # 乱数生成器のシードを設定
            st.session_state['model'], st.session_state['tokenizer'] = models.load(gguf, repo)  # モデルとトークナイザをロード
            st.session_state['model_loaded'] = True
            st.success('Model loaded successfully!')

# UI要素の定義（ユーザー入力など）
prompt = st.text_input("Prompt:", "")
max_tokens = st.slider("Max tokens:", min_value=1, max_value=1000, value=100)
temp = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.0)

# Generateボタン（モデルがロードされている場合のみ有効）
if st.session_state['model_loaded']:
    if st.button("Generate"):
        output_container = st.empty()  # 出力表示用のプレースホルダー
        for generated_text in generate.generate_tokens_iter(st.session_state['model'], st.session_state['tokenizer'], prompt, max_tokens, temp):
            # プレースホルダーの内容を更新
            output_container.text_area("Generated Text:", value=generated_text, height=300)
else:
    # モデルがロードされていない場合、Generateボタンを無効化
    st.write("Please load the model first using the 'Load Model' button on the sidebar.")
