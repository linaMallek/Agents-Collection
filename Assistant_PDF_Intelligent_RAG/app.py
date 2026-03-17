import tempfile
import os
import streamlit as st
from src.ingest import create_db
from src.rag_agent import build_agent

st.set_page_config(page_title='PDF Assistant', page_icon='📄')
st.title('📄 PDF Assistant')

# ── Sidebar: file upload ──────────────────────────────────────────────────────
with st.sidebar:
    st.header('Upload your document')
    uploaded_file = st.file_uploader(
        'Supported formats: PDF, TXT',
        type=['pdf', 'txt'],
    )

# ── Session state initialisation ─────────────────────────────────────────────
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_file' not in st.session_state:
    st.session_state.current_file = None

# ── Build agent when a new file is uploaded ───────────────────────────────────
if uploaded_file is not None and uploaded_file.name != st.session_state.current_file:
    with st.spinner('Reading and indexing your document…'):
        suffix = '.pdf' if uploaded_file.name.endswith('.pdf') else '.txt'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            vector_store = create_db(tmp_path)
            st.session_state.agent = build_agent(vector_store)
            st.session_state.current_file = uploaded_file.name
            st.session_state.messages = []
        finally:
            os.unlink(tmp_path)
    st.success(f'✅ "{uploaded_file.name}" indexed. Ask away!')


if st.session_state.agent is None:
    st.info('Upload a document in the sidebar to get started.')
else:
    # Render existing conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    # Accept new user input
    user_input = st.chat_input('Ask a question about your document…')
    if user_input:
        st.session_state.messages.append({'role': 'user', 'content': user_input})
        with st.chat_message('user'):
            st.markdown(user_input)

        with st.chat_message('assistant'):
            with st.spinner('Thinking…'):
                lc_messages = [
                    {'role': m['role'], 'content': m['content']}
                    for m in st.session_state.messages
                ]
                response = st.session_state.agent.invoke({'messages': lc_messages},
                                                        {"configurable": {"thread_id": "1"}},)
                
                answer = response['messages'][-1].content
            st.markdown(answer)

        st.session_state.messages.append({'role': 'assistant', 'content': answer})

