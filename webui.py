import streamlit as st
from Script import generate
import mlx.core as mx
from Script import models

# UI要素の定義（ユーザー入力など）
prompt = st.text_input("Prompt:", "ここにプロンプトを入力してください")
max_tokens = st.slider("Max tokens:", min_value=1, max_value=1000, value=100)
temp = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0)

if st.button("Generate"):
    gguf ='mistral-7b-v0.1.Q8_0.gguf'
    repo = 'TheBloke/Mistral-7B-v0.1-GGUF'
    prompt = 'Write a quicksort in Python'
    max_token = 100

    mx.random.seed(100)  # 乱数生成器のシードを設定
    model, tokenizer = models.load(gguf, repo)  # モデルとトークナイザをロード

    # tokens = generate.generate_tokens(model, tokenizer, prompt, max_tokens, temp)  # モデルとトークナイザは事前に定義されていると仮定
    # generated_text = tokenizer.decode(tokens)
    # st.text_area("Generated Text:", value=generated_text, height=300)

    output_container = st.empty()  # 出力表示用のプレースホルダー
    for generated_text in generate.generate_tokens_iter(model, tokenizer, prompt, max_tokens, temp):
        # プレースホルダーの内容を更新
        output_container.text_area("Generated Text:", value=generated_text, height=300)
