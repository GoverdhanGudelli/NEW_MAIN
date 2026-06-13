import os
import glob
import warnings
import streamlit as st
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Zyro HR Help Desk", page_icon="🏢")
st.title("🏢 Zyro Dynamics HR Help Desk")
st.caption("Ask me anything about Zyro Dynamics HR policies")

# ← Add sidebar here
with st.sidebar:
    st.header("📋 About")
    st.write("Zyro Dynamics HR Help Desk")
    st.write("Powered by RAG + LLaMA 3.3")
    st.divider()
    st.write("**Available Policies:**")
    policies = [
        "📄 Employee Handbook",
        "🏖️ Leave Policy",
        "🏠 Work From Home",
        "⚖️ Code of Conduct",
        "📊 Performance Review",
        "💰 Compensation & Benefits",
        "🔒 IT & Data Security",
        "🛡️ POSH Policy",
        "✈️ Travel & Expense",
        "🚀 Onboarding & Separation"
    ]
    for p in policies:
        st.write(p)
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

@st.cache_resource(show_spinner="📚 Building knowledge base...")
def load_pipeline():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"

    PDF_DIR = "data/"
    documents = []
    for path in glob.glob(PDF_DIR + "*.pdf"):
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
 chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 7, "fetch_k": 20}
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=st.secrets["GROQ_API_KEY"]
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an HR assistant for Zyro Dynamics Pvt. Ltd.
Answer ONLY using the provided context from HR policy documents.
If context does not contain the answer, say you don't have that information.
Be accurate and professional.Answer in detail with specific numbers, dates, 
and policy names from the context."

Context:
{context}"""),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(
            f"[Source: {os.path.basename(d.metadata.get('source','unknown'))}, "
            f"Page {d.metadata.get('page','?')}]\n{d.page_content}"
            for d in docs
        )

    pipeline = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return pipeline, retriever, llm

pipeline, retriever, llm = load_pipeline()

REFUSAL = "I can only answer HR-related questions from Zyro Dynamics policy documents. Please ask about leave, salary, WFH, performance, conduct, or benefits."

def is_in_scope(question):
    out_of_scope = ["stock price", "cricket", "weather", "recipe",
                    "movie", "politics", "sports", "investment"]
    if any(kw in question.lower() for kw in out_of_scope):
        return False
    check_prompt = ChatPromptTemplate.from_messages([
        ("human", """Does this question relate to HR, company policy, leave,
salary, benefits, conduct, performance, onboarding, travel expenses, WFH, or IT security?
Reply ONLY: IN_SCOPE or OUT_OF_SCOPE
Question: {question}""")
    ])
    chain = check_prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question}).strip()
    return result == "IN_SCOPE"

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                for s in msg["sources"]:
                    st.write(f"- **{s['file']}** — Page {s['page']}")

if user_input := st.chat_input("Ask an HR question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            if not is_in_scope(user_input):
                answer = REFUSAL
                sources = []
            else:
                source_docs = retriever.invoke(user_input)
                answer = pipeline.invoke(user_input)
                sources = [
                    {
                        "file": os.path.basename(d.metadata.get("source", "unknown")),
                        "page": d.metadata.get("page", "?")
                    }
                    for d in source_docs
                ]
        st.markdown(answer)
        if sources:
            with st.expander("📄 Sources"):
                for s in sources:
                    st.write(f"- **{s['file']}** — Page {s['page']}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })