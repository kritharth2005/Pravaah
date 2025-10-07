from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from vector import (
    add_to_chroma,
    clear_database,
    get_embedding_function,
    load_documents,
    spilt_documents,
)

load_dotenv()

def query_rag(query_text: str):
    vector_store = Chroma(
        persist_directory="chroma_langchain_db",
        embedding_function=get_embedding_function(),
    )

    PROMPT_TEMPLATE = """
        You are an AI legal assistant. Your task is to provide a clear and concise answer based exclusively on the provided legal context.

        CONTEXT:
        {context}
        ---
        QUESTION: {question}

        INSTRUCTIONS:
        1.  Answer the question using only the information from the CONTEXT above.
        2.  Cite the specific sections or clauses you are referencing in your answer.
        3.  If the context does not contain the answer, state that the information is not available in the provided text.
        4.  Conclude your response with the disclaimer: "This is an AI-generated summary for informational purposes only and does not constitute legal advice. Please consult a qualified legal professional."

    """

    results = vector_store.similarity_search_with_score(query=query_text, k=7)
    # print(results)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    print("=" * 55)
    print(prompt)
    print("=" * 55)

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    response = model.invoke(prompt)


    print("=" * 55)
    print(response.content)
    print("=" * 55)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response.content}\nSource: {sources}"
    print("=" * 55)
    print(formatted_response)
    print("=" * 55)


def load_vector_store():
    documents = load_documents()
    chunks = spilt_documents(documents)
    add_to_chroma(chunks)
    return {"message": "Done"}


def delete_vector_store():
    clear_database()
    return {"message": "Done"}

if __name__ == "__main__":
    print(load_vector_store())
    query = """
    After being fired from his job, Rohan sends a series of threatening emails to his former manager in Mangaluru. The emails do not threaten physical harm but state that Rohan will 'ruin his reputation' by posting fabricated, defamatory stories about him online if he is not paid a severance of ₹5,00,000. Does this act constitute 'criminal intimidation' under the BNS? Explain your reasoning by analyzing the nature of the threat required for this offense.
"""
    query_rag(query_text=query)
    # clear_database()