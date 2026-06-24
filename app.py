import streamlit as st

st.title("RAG APP")

with open("data\sample.txt", "r") as file:
    text = file.read()

st.write(text)
