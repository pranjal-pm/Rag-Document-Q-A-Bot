"""
Streamlit web application for Policy Document Q&A Bot
"""
import os
import sys
from pathlib import Path

os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '0'
os.environ.setdefault('ACCELERATE_USE_CPU', '1')
os.environ.setdefault('ACCELERATE_NO_CPU_MEMORY_EFFICIENT', '1')

import streamlit as st
import re
import html
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.rag_pipeline import RAGPipeline
from src.auth import UserAuth
from src.config import (
    PAGE_TITLE, PAGE_ICON, VECTOR_DB_PATH, METADATA_PATH,
    DEFAULT_LLM_PROVIDER, USE_LLM_BY_DEFAULT, OPENAI_API_KEY,
    OPENAI_MODEL, TEMPERATURE, MAX_TOKENS
)

try:
    from streamlit_chat import message
    HAS_STREAMLIT_CHAT = True
except ImportError:
    HAS_STREAMLIT_CHAT = False
    def message(content, is_user=False, key=None):
        if is_user:
            st.markdown(f"**You:** {content}")
        else:
            st.markdown(f"**Bot:** {content}")


# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    h1 {
        color: #1f77b4;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Chat message styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 20px;
        height: 3rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1f77b4;
    }
    
    /* Status badges */
    .status-badge {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    /* Example questions */
    .example-question {
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s;
        border: none;
    }
    
    .example-question:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Info boxes */
    .info-box {
        padding: 1rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        margin: 1rem 0;
    }
    
    /* Chat container */
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    /* Scrollbar styling */
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    
    /* Auth form styling */
    .auth-container {
        max-width: 500px;
        margin: 2rem auto;
        padding: 2rem;
        background: white;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .auth-input {
        margin: 1rem 0;
    }
    
    /* User badge */
    .user-badge {
        padding: 0.5rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        display: inline-block;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

auth = UserAuth()

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.session_token = None

if 'rag_pipeline' not in st.session_state:
    try:
        st.session_state.rag_pipeline = RAGPipeline(
            use_openai_llm=False,
            openai_api_key=None,
            llm_provider="none"
        )
        st.session_state.initialized = True
    except Exception as e:
        st.session_state.initialized = False
        st.session_state.error = str(e)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'use_llm' not in st.session_state:
    st.session_state.use_llm = USE_LLM_BY_DEFAULT

if 'llm_provider' not in st.session_state:
    st.session_state.llm_provider = DEFAULT_LLM_PROVIDER

if 'auto_save_history' not in st.session_state:
    st.session_state.auto_save_history = True


def is_clarification_request(query: str, rag_pipeline, use_llm: bool = False, llm_provider: str = "none") -> bool:
    """Detect if user is asking for clarification"""
    query_lower = query.lower().strip()
    
    clarification_patterns = [
        "don't understand", "didn't understand", "not understand", "can't understand",
        "don't get it", "didn't get it", "not clear", "unclear",
        'explain easier', 'explain simply', 'explain in simple', 'simple explanation',
        'easy explanation', 'make it easy', 'make it simple', 'simplify',
        'explain again', 'explain better', 'explain more', 'clarify',
        'easy version', 'simpler version', 'easier answer', 'simple answer'
    ]
    
    if any(pattern in query_lower for pattern in clarification_patterns):
        return True
    
    if use_llm and llm_provider == "openai" and rag_pipeline and hasattr(rag_pipeline, 'openai_client') and rag_pipeline.openai_client:
        try:
            detection_prompt = f"""Analyze if the user is asking for a simpler or easier explanation of a previous answer.

User query: "{query}"

Respond with ONLY "YES" if the user wants:
- A simpler/easier explanation
- Clarification because they didn't understand
- An easier version of the answer
- To understand something better

Respond with ONLY "NO" if it's a new question or different request.

Response:"""
            
            response = rag_pipeline.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that detects user intent. Respond with only YES or NO."},
                    {"role": "user", "content": detection_prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            result = response.choices[0].message.content.strip().upper()
            return "YES" in result
        except:
            pass
    
    return False


def get_last_bot_response():
    """Get the last bot response from chat history"""
    for chat in reversed(st.session_state.chat_history):
        if chat['type'] == 'bot':
            return chat
    return None


def get_chat_history_file(user_id: str = None) -> Path:
    """Get the path to the chat history file for a user"""
    history_dir = Path(__file__).parent / "chat_history"
    history_dir.mkdir(exist_ok=True)
    
    if user_id:
        filename = f"chat_history_{user_id}.json"
    else:
        filename = "chat_history_default.json"
    
    return history_dir / filename


def save_chat_history(chat_history: list, user_id: str = None) -> bool:
    """Save chat history to a JSON file"""
    try:
        history_file = get_chat_history_file(user_id)
        # Prepare data for saving
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'chat_history': chat_history,
            'count': len(chat_history)
        }
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        pass
        return False


def load_chat_history(user_id: str = None) -> list:
    """Load chat history from a JSON file"""
    try:
        history_file = get_chat_history_file(user_id)
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('chat_history', [])
        return []
    except Exception as e:
        pass
        return []


def export_chat_history(chat_history: list) -> str:
    """Export chat history as a formatted text string"""
    if not chat_history:
        return "No chat history to export."
    
    lines = ["=" * 60, "Chat History Export", "=" * 60, ""]
    
    for i, chat in enumerate(chat_history, 1):
        if chat['type'] == 'user':
            lines.append(f"User ({i}):")
            lines.append(chat['content'])
        else:
            lines.append(f"\nBot ({i}):")
            lines.append(chat['content'])
            if chat.get('sources'):
                lines.append(f"Sources: {', '.join(chat['sources'])}")
        lines.append("\n" + "-" * 60 + "\n")
    
    return "\n".join(lines)


def clean_html_text(text: str) -> str:
    """Remove HTML tags and decode HTML entities"""
    if not text:
        return ""
    
    text = str(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = html.unescape(text)
    text = re.sub(r'&[a-zA-Z]+;', '', text)
    text = re.sub(r'&#\d+;', '', text)
    text = re.sub(r'&#x[0-9a-fA-F]+;', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def generate_simplified_explanation(original_answer: str, query: str, chunks: list, rag_pipeline) -> str:
    """Generate a simpler explanation"""
    return f"Let me explain this more simply:\n\n{original_answer}\n\nThis information comes directly from your documents. If you need more details, please ask a more specific question!"


def show_login_page():
    """Display login page"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h1 style="color: #667eea; font-size: 3rem; margin-bottom: 0.5rem;">🔐 Welcome Back</h1>
        <p style="color: #666; font-size: 1.2rem;">Login to access your Policy Document Q&A Bot</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.container():
            st.markdown("### Login")
            
            username = st.text_input("Username or Email", key="login_username", placeholder="Enter your username or email")
            password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")
            
            col_login, col_register = st.columns(2)
            
            with col_login:
                if st.button("🚀 Login", use_container_width=True, type="primary"):
                    if username and password:
                        success, user_data, message = auth.login_user(username, password)
                        if success:
                            st.session_state.authenticated = True
                            st.session_state.user = user_data
                            st.session_state.session_token = user_data['session_token']
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.warning("Please fill in all fields")
            
            with col_register:
                if st.button("📝 Register", use_container_width=True):
                    st.session_state.show_register = True
                    st.rerun()
            
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; color: #666; padding: 1rem;">
                <p>Don't have an account? Click Register to create one.</p>
            </div>
            """, unsafe_allow_html=True)


def show_register_page():
    """Display registration page"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h1 style="color: #667eea; font-size: 3rem; margin-bottom: 0.5rem;">✨ Create Account</h1>
        <p style="color: #666; font-size: 1.2rem;">Register to start using Policy Document Q&A Bot</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.container():
            st.markdown("### Registration Form")
            
            full_name = st.text_input("Full Name", key="reg_full_name", placeholder="Enter your full name (optional)")
            username = st.text_input("Username *", key="reg_username", placeholder="Choose a username (min 3 characters)")
            email = st.text_input("Email *", key="reg_email", placeholder="Enter your email address")
            password = st.text_input("Password *", type="password", key="reg_password", placeholder="Create a password (min 6 characters)")
            confirm_password = st.text_input("Confirm Password *", type="password", key="reg_confirm_password", placeholder="Confirm your password")
            
            st.markdown("*Required fields")
            
            col_register, col_back = st.columns(2)
            
            with col_register:
                if st.button("✅ Register", use_container_width=True, type="primary"):
                    # Validation
                    if not username or not email or not password:
                        st.error("Please fill in all required fields")
                    elif password != confirm_password:
                        st.error("Passwords do not match")
                    elif len(password) < 6:
                        st.error("Password must be at least 6 characters long")
                    elif len(username) < 3:
                        st.error("Username must be at least 3 characters long")
                    else:
                        success, message = auth.register_user(username, email, password, full_name)
                        if success:
                            st.success(message)
                            st.info("You can now login with your credentials")
                            st.session_state.show_register = False
                            st.rerun()
                        else:
                            st.error(message)
            
            with col_back:
                if st.button("← Back to Login", use_container_width=True):
                    st.session_state.show_register = False
                    st.rerun()


def main():
    """Main application"""
    if 'show_register' not in st.session_state:
        st.session_state.show_register = False
    
    if st.session_state.authenticated and st.session_state.session_token:
        user_data = auth.verify_session(st.session_state.session_token)
        if not user_data:
            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.session_token = None
    
    if not st.session_state.authenticated:
        if st.session_state.show_register:
            show_register_page()
        else:
            show_login_page()
        return
    
    user = st.session_state.user
    
    if 'chat_history_loaded' not in st.session_state:
        user_id = user.get('username', 'default')
        saved_history = load_chat_history(user_id)
        if saved_history and not st.session_state.chat_history:
            st.session_state.chat_history = saved_history
        st.session_state.chat_history_loaded = True
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2.5rem; 
                border-radius: 15px; 
                margin-bottom: 2rem;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h1 style="color: white; margin: 0; font-size: 2.8rem; font-weight: 700;">📄 Document Q&A Assistant</h1>
        <p style="color: rgba(255,255,255,0.95); font-size: 1.3rem; margin-top: 0.8rem; font-weight: 300;">
            Intelligent answers from your document collection
        </p>
        <div style="margin-top: 1.5rem;">
            <span style="background: rgba(255,255,255,0.25); padding: 0.6rem 1.2rem; border-radius: 25px; color: white; font-weight: 500;">
                👤 {user.get('full_name', user.get('username', 'User'))}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); 
                    padding: 1.5rem; 
                    border-radius: 10px; 
                    margin-bottom: 1.5rem;
                    border: 1px solid #81c784;">
            <h3 style="color: #2e7d32; margin: 0 0 1rem 0; font-size: 1.3rem; font-weight: 600;">📚 Available Document Context</h3>
            <p style="color: #555; font-size: 0.95rem; margin-bottom: 1rem;">
                You can ask questions about the following policy areas:
            </p>
            <div style="display: flex; flex-wrap: wrap; gap: 0.8rem;">
                <span style="background: white; padding: 0.6rem 1.2rem; border-radius: 20px; color: #2e7d32; font-weight: 500; font-size: 0.9rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    💰 Financial Policies
                </span>
                <span style="background: white; padding: 0.6rem 1.2rem; border-radius: 20px; color: #2e7d32; font-weight: 500; font-size: 0.9rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    📊 Economic Policies
                </span>
                <span style="background: white; padding: 0.6rem 1.2rem; border-radius: 20px; color: #2e7d32; font-weight: 500; font-size: 0.9rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    ⚖️ Legal Policies
                </span>
                <span style="background: white; padding: 0.6rem 1.2rem; border-radius: 20px; color: #2e7d32; font-weight: 500; font-size: 0.9rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    🏛️ Regulatory Policies
                </span>
                <span style="background: white; padding: 0.6rem 1.2rem; border-radius: 20px; color: #2e7d32; font-weight: 500; font-size: 0.9rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    📋 Administrative Policies
                </span>
                <span style="background: white; padding: 0.6rem 1.2rem; border-radius: 20px; color: #2e7d32; font-weight: 500; font-size: 0.9rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    🔒 Compliance Policies
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar - Simplified and Professional
    with st.sidebar:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.2rem; 
                    border-radius: 10px; 
                    color: white;
                    margin-bottom: 1.5rem;
                    text-align: center;">
            <h4 style="color: white; margin: 0; font-weight: 600;">👤 {user.get('full_name', user.get('username', 'User'))}</h4>
            <p style="color: rgba(255,255,255,0.85); margin: 0.5rem 0 0 0; font-size: 0.85rem;">{user.get('email', '')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if not VECTOR_DB_PATH.exists():
            st.error("⚠️ **Database not found**")
            st.info("Run `python ingest_documents.py` to process documents.")
            return
        
        if st.session_state.initialized:
            st.info("📄 **Document-Only Mode**")
            st.caption("Answers are retrieved directly from your documents")
        
        st.session_state.use_llm = False  # Always disabled - document-only mode
        
        st.markdown("---")
        
        # Chat History Management Section
        st.markdown("### 💾 Chat History")
        
        auto_save = st.checkbox(
            "💾 Auto-save chat history",
            value=st.session_state.get('auto_save_history', True),
            help="Automatically save chat history when you close the app"
        )
        st.session_state.auto_save_history = auto_save
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save", use_container_width=True, help="Save current chat history"):
                if st.session_state.authenticated and st.session_state.user:
                    user_id = st.session_state.user.get('username', 'default')
                    if save_chat_history(st.session_state.chat_history, user_id):
                        st.success("✅ Saved!")
                    else:
                        st.error("❌ Failed to save")
                else:
                    if save_chat_history(st.session_state.chat_history):
                        st.success("✅ Saved!")
                    else:
                        st.error("❌ Failed to save")
        
        with col2:
            if st.button("📂 Load", use_container_width=True):
                if st.session_state.authenticated and st.session_state.user:
                    user_id = st.session_state.user.get('username', 'default')
                    loaded = load_chat_history(user_id)
                else:
                    loaded = load_chat_history()
                
                if loaded:
                    st.session_state.chat_history = loaded
                    st.success(f"✅ Loaded {len(loaded)} messages!")
                    st.rerun()
                else:
                    st.info("ℹ️ No saved history found")
        
        if st.button("📥 Export", use_container_width=True):
            export_text = export_chat_history(st.session_state.chat_history)
            if export_text:
                st.download_button(
                    label="⬇️ Download Chat History",
                    data=export_text,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
        st.markdown("---")
        
        # Action buttons
        if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
            st.session_state.chat_history = []
            if st.session_state.auto_save_history:
                if st.session_state.authenticated and st.session_state.user:
                    user_id = st.session_state.user.get('username', 'default')
                    save_chat_history([], user_id)
                else:
                    save_chat_history([])
            st.rerun()
        
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            if st.session_state.session_token:
                auth.logout_user(st.session_state.session_token)
            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.session_token = None
            st.session_state.chat_history = []
            st.rerun()
    
    if not st.session_state.initialized:
        st.error(f"❌ **Error initializing RAG pipeline:** {st.session_state.get('error', 'Unknown error')}")
        return
    
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="margin: 2rem 0 1.5rem 0;">
            <h3 style="color: #333; margin-bottom: 0.5rem; font-size: 1.5rem;">Try These Questions</h3>
            <p style="color: #666; font-size: 0.95rem;">Click any question to get started</p>
        </div>
        """, unsafe_allow_html=True)
        
        example_questions = [
            "What is a Calderbank offer?",
            "How are legal costs awarded?",
            "Explain indemnity costs",
            "What are the requirements for cost awards?",
            "What is the difference between party costs and indemnity costs?"
        ]
        
        cols = st.columns(2)
        for idx, question in enumerate(example_questions):
            with cols[idx % 2]:
                if st.button(f"{question}", key=f"example_{idx}", use_container_width=True):
                    st.session_state.example_question = question
                    st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
    
    if st.session_state.chat_history:
        st.markdown("### 💬 Conversation")
        chat_container = st.container()
        with chat_container:
            for i, chat in enumerate(st.session_state.chat_history):
                if chat['type'] == 'user':
                    with st.chat_message("user"):
                        st.write(chat['content'])
                else:
                    answer_content = clean_html_text(chat['content'])
                    sources = chat.get('sources', [])
                    is_simplified = chat.get('is_simplified', False)
                    
                    with st.chat_message("assistant"):
                        if is_simplified:
                            st.success("✨ Simplified Explanation")
                        st.markdown(answer_content)
                        if sources:
                            sources_text = ", ".join(sources[:3])
                            st.caption(f"📚 Sources: {sources_text}")
    else:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #999; margin: 2rem 0;">
            <p style="font-size: 1rem;">Your conversation will appear here</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Ask Your Question")
    
    default_value = ""
    if 'example_question' in st.session_state:
        default_value = st.session_state.example_question
        del st.session_state.example_question
    
    query = st.text_input("", value=default_value, placeholder="Enter your question here... ", key="query_input", label_visibility="collapsed")
    
    col1, col2, col3 = st.columns([2, 2, 8])
    with col1:
        submit_button = st.button("Submit", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    if submit_button and query:
        with st.spinner("🔍 Searching documents and generating answer..."):
            try:
                use_llm = st.session_state.get('use_llm', False) and st.session_state.get('llm_provider', 'none') != "none"
                llm_provider = st.session_state.get('llm_provider', 'none') or 'none'
                if is_clarification_request(query, st.session_state.rag_pipeline, use_llm, llm_provider):
                    last_bot_response = get_last_bot_response()
                    
                    if last_bot_response:
                        simplified_answer = generate_simplified_explanation(
                            original_answer=last_bot_response.get('content', ''),
                            query=last_bot_response.get('query', ''),
                            chunks=last_bot_response.get('chunks', []),
                            rag_pipeline=st.session_state.rag_pipeline
                        )
                        
                        st.session_state.chat_history.append({'type': 'user', 'content': query})
                        st.session_state.chat_history.append({
                            'type': 'bot',
                            'content': clean_html_text(simplified_answer),
                            'sources': last_bot_response.get('sources', []),
                            'chunks': last_bot_response.get('chunks', []),
                            'query': last_bot_response.get('query', ''),
                            'is_simplified': True
                        })
                        
                        if st.session_state.get('auto_save_history', True):
                            if st.session_state.authenticated and st.session_state.user:
                                user_id = st.session_state.user.get('username', 'default')
                                save_chat_history(st.session_state.chat_history, user_id)
                            else:
                                save_chat_history(st.session_state.chat_history)
                    else:
                        st.session_state.chat_history.append({'type': 'user', 'content': query})
                        st.session_state.chat_history.append({
                            'type': 'bot',
                            'content': "I don't have a previous answer to simplify. Please ask me a question first, and then I can explain it more simply if needed! 😊"
                        })
                else:
                    result = st.session_state.rag_pipeline.query(query, use_llm=False, llm_provider=None)
                    
                    st.session_state.chat_history.append({'type': 'user', 'content': query})
                    st.session_state.chat_history.append({
                        'type': 'bot',
                        'content': clean_html_text(result['answer']),
                        'sources': result.get('sources', []),
                        'chunks': result.get('chunks', []),
                        'query': result.get('query', query)
                    })
                    
                    if st.session_state.get('auto_save_history', True):
                        if st.session_state.authenticated and st.session_state.user:
                            user_id = st.session_state.user.get('username', 'default')
                            save_chat_history(st.session_state.chat_history, user_id)
                        else:
                            save_chat_history(st.session_state.chat_history)
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ **Error processing query:** {e}")
                st.info("Please try again or rephrase your question.")
    


if __name__ == "__main__":
    main()

