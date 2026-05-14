import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import anthropic

# --- Setup (runs once) ---
@st.cache_resource
def load_resources():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("boris")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    claude = anthropic.Anthropic()
    return collection, model, claude

st.markdown("""
<style>
/* Background */
.stApp { background-color: #0f0f0f; }

/* Chat messages */
.stChatMessage { background-color: #1a1a1a; border-radius: 12px; }

/* Input box */
.stChatInput input { background-color: #1a1a1a; color: white; border: 1px solid #333; }

/* Title */
h1 { color: white; font-family: 'Georgia', serif; }

/* Caption */
.stCaption { color: #888; }
</style>
""", unsafe_allow_html=True)

collection, model, claude = load_resources()

# --- UI ---
st.title("Ask Boris")
st.caption("Ask me anything about design, product, AI, and hiring. Answers based on my writing.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle input
if prompt := st.chat_input("What do you think about..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve relevant documents
    vector = model.encode([prompt]).tolist()
    results = collection.query(query_embeddings=vector, n_results=5)
    
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    
    context = ""
    sources = []
    for doc, meta in zip(docs, metas):
        context += f"\n---\n{doc}\n"
        if meta.get("url"):
            title = meta.get("title") or "Post"
            date = meta.get("date", "")[:10]
            url = meta["url"]
            sources.append(f"[{title[:60]}]({url}) — {date}")

    # Build prompt
    system = """You are Boris Kehr — a designer and ML student based in Stockholm with 15+ years of experience in UX, product design, and now AI/ML. 
Answer questions based only on the context provided. Be direct and opinionated, in Boris's voice. 
If the context doesn't contain enough information to answer, say so honestly.
Always answer in the same language the question was asked in."""

    user_prompt = f"""Context from Boris's writing:
{context}

Question: {prompt}"""

    # Call Claude
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                system=system,
                messages=[{"role": "user", "content": user_prompt}]
            )
            answer = response.content[0].text
            
            if sources:
                answer += "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources)
            
            st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})