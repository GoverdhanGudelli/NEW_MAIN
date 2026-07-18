app_code = '''
import os
import glob
import warnings
import streamlit as st
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

st.set_page_config(page_title="Zyro HR Help Desk", page_icon="\U0001F3E2")
st.title("\U0001F3E2 Zyro Dynamics HR Help Desk")
st.caption("Ask me anything about Zyro Dynamics HR policies")

# Same values as the notebook (Cell 4 CHUNK_SIZE / CHUNK_OVERLAP) so the
# deployed app and the notebook retrieve identically.
CHUNK_SIZE = 900
CHUNK_OVERLAP = 100
RETRIEVAL_K = 5
RETRIEVAL_FETCH_K = 20

@st.cache_resource(show_spinner="Building knowledge base...")
def load_pipeline():
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    # os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"

    PDF_DIR = "data/"
    documents = []
    for path in glob.glob(PDF_DIR + "*.pdf"):
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\\n\\n", "\\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)

    # Same embedding model as the notebook -- bge-base-en-v1.5, not the
    # weaker all-MiniLM-L6-v2 that was here before.
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = RETRIEVAL_FETCH_K

    dense_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVAL_FETCH_K, "fetch_k": RETRIEVAL_FETCH_K * 2}
    )

    def _doc_key(doc):
        return (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:50])

    def reciprocal_rank_fusion(result_lists, weights, k=60):
        scores = {}
        doc_lookup = {}
        for docs, weight in zip(result_lists, weights):
            for rank, doc in enumerate(docs):
                key = _doc_key(doc)
                doc_lookup[key] = doc
                scores[key] = scores.get(key, 0.0) + weight * (1.0 / (k + rank + 1))
        ranked_keys = sorted(scores.keys(), key=lambda kk: scores[kk], reverse=True)
        return [doc_lookup[kk] for kk in ranked_keys]

    def fused_retrieve(question):
        bm25_docs = bm25_retriever.invoke(question)
        dense_docs = dense_retriever.invoke(question)
        return reciprocal_rank_fusion([bm25_docs, dense_docs], weights=[0.4, 0.6])

    reranker = CrossEncoder("BAAI/bge-reranker-base", device="cpu")

    def rerank(question, docs, top_k=RETRIEVAL_K):
        if not docs:
            return docs
        pairs = [[question, d.page_content] for d in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[:top_k]]

    class RerankingRetriever:
        def __init__(self, top_k=RETRIEVAL_K):
            self.top_k = top_k

        def invoke(self, question):
            candidates = fused_retrieve(question)
            return rerank(question, candidates, self.top_k)

    retriever = RerankingRetriever(top_k=RETRIEVAL_K)

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=700,
        api_key=st.secrets["GROQ_API_KEY"]
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an HR assistant for Zyro Dynamics Pvt. Ltd.
Answer ONLY using the provided context from HR policy documents.
If context does not contain the answer, say you don\'t have that information.
Be concise, accurate and professional.

Context:
{context}"""),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\\n\\n".join(
            f"[Source: {os.path.basename(d.metadata.get(\'source\',\'unknown\'))}, "
            f"Page {d.metadata.get(\'page\',\'?\')}]\\n{d.page_content}"
            for d in docs
        )

    retrieve_and_format = RunnableLambda(lambda q: format_docs(retriever.invoke(q)))

    pipeline = (
        {"context": retrieve_and_format, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return pipeline, retriever, llm

pipeline, retriever, llm = load_pipeline()

REFUSAL = "I can only answer HR-related questions from Zyro Dynamics policy documents. Please ask about leave, salary, WFH, performance, conduct, or benefits."

OUT_OF_SCOPE_KEYWORDS = ["stock price", "cricket", "weather", "recipe",
                "movie", "politics", "sports", "investment"]

def is_in_scope(question):
    if any(kw in question.lower() for kw in OUT_OF_SCOPE_KEYWORDS):
        return False
    check_prompt = ChatPromptTemplate.from_messages([
        ("human", """Does this question relate to HR, company policy, leave,
salary, benefits, conduct, performance, onboarding, travel expenses, WFH, or IT security?
Reply ONLY: IN_SCOPE or OUT_OF_SCOPE
Question: {question}""")
    ])
    chain = check_prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question}).strip().upper()
    # Default to IN_SCOPE on an unexpected/malformed response -- refusing a
    # genuine HR question is a worse failure than occasionally answering
    # a borderline one.
    if "OUT_OF_SCOPE" in result:
        return False
    return True

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(f"- **{s[\'file\']}** -- Page {s[\'page\']}")

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
            with st.expander("Sources"):
                for s in sources:
                    st.write(f"- **{s[\'file\']}** -- Page {s[\'page\']}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
'''

with open("app.py", "w") as f:
    f.write(app_code.strip())

print("app.py created.")
