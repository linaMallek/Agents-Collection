import tempfile
import os
import streamlit as st
from src.ingest import create_db
from src.rag_agent import build_agent

import tempfile
import os
import streamlit as st
from src.ingest import create_db
from src.rag_agent import build_agent

st.set_page_config(page_title='PDF Assistant', page_icon='📄')
st.title('📄 PDF Assistant')

# ── Sidebar: file upload ──────────────────────────────────────────────────────
with st.sidebar:
    st.header('Upload your documents')
    uploaded_files = st.file_uploader(          
        'Supported formats: PDF, TXT',
        type=['pdf', 'txt'],
        accept_multiple_files=True,             
    )

# ── Session state initialisation ─────────────────────────────────────────────
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_files' not in st.session_state:
    st.session_state.current_files = []        

# ── Build agent when new files are uploaded ───────────────────────────────────
if uploaded_files:
    uploaded_names = sorted([f.name for f in uploaded_files])

    if uploaded_names != st.session_state.current_files:   
        with st.spinner('Reading and indexing your documents…'):
            tmp_paths = []
            try:
                # save all files to temp paths
                for uploaded_file in uploaded_files:
                    suffix = '.pdf' if uploaded_file.name.endswith('.pdf') else '.txt'
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_paths.append(tmp.name)

                vector_store = create_db(tmp_paths)         
                st.session_state.agent = build_agent(vector_store)
                st.session_state.current_files = uploaded_names
                st.session_state.messages = []
            finally:
                for path in tmp_paths:                      
                    os.unlink(path)

        st.success(f'✅ {len(uploaded_files)} file(s) indexed. Ask away!')
        
        # ✅ show which files are loaded in sidebar
        with st.sidebar:
            st.markdown("**Indexed files:**")
            for name in uploaded_names:
                st.markdown(f"- 📄 {name}")


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

