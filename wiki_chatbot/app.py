import streamlit as st
from scraper import scrape_url
from chatbot import create_qa_system

# Streamlit page setup
st.set_page_config(page_title="Wiki Chatbot", layout="wide")
st.title("Wikipedia Chatbot — LangChain + Groq")
st.caption("Paste a Wikipedia link, and ask questions based on that page.")

# Store QA system in Streamlit session state
if 'qa_system' not in st.session_state:
    st.session_state['qa_system'] = None

# Input URL
url = st.text_input("🔗 Paste Wikipedia / Web Link here:")

# Scrape & Load Button
if st.button("Scrape & Load"):
    if url:
        with st.spinner("🔍 Scraping website and preparing chatbot..."):
            try:
                chunks = scrape_url(url)
                st.session_state['qa_system'] = create_qa_system(chunks)
                st.success(f"✅ Scraping done! {len(chunks)} chunks loaded.")
            except Exception as e:
                st.error(f"❌ Error while scraping or creating QA system: {e}")
    else:
        st.error("⚠️ Please paste a URL.")

# Ask question
query = st.text_input("Ask your question:")

if st.button("Ask") and query:
    qa = st.session_state.get('qa_system')
    if qa:
        with st.spinner("🤖 Getting answer from Groq LLM..."):
            try:
                result = qa.invoke(query)

                # ✅ Handle result properly (AIMessage or dict)
                if hasattr(result, "content"):
                    st.markdown("**Answer:**")
                    st.write(result.content)
                elif isinstance(result, dict) and "result" in result:
                    st.markdown("**Answer:**")
                    st.write(result["result"])
                else:
                    st.error("⚠️ Unexpected output format from model.")

            except Exception as e:
                st.error(f"❌ Error while getting answer: {e}")
    else:
        st.error("⚠️ Please scrape & load a URL first.")
