"""Build the LangChain RetrievalQA chain over the FAISS index."""
import os
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from ingest import INDEX_PATH

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Use the following context to answer the question accurately and concisely.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
)


def build_qa_chain() -> RetrievalQA:
    embeddings = OpenAIEmbeddings()
    store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = store.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT},
    )
