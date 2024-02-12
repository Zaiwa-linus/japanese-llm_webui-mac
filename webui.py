import streamlit as st

# サイドバーにテキスト入力を追加
user_input = st.sidebar.text_input("名前を入力してください")

# サイドバーにボタンを追加
if st.sidebar.button('挨拶'):
    st.sidebar.write(f'こんにちは、{user_input}さん！')
