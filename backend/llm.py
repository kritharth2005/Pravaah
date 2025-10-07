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

def human_summarizer(query_text: str):
    vector_store = Chroma(
        persist_directory="chroma_langchain_db",
        embedding_function=get_embedding_function(),
    )

    PROMPT_TEMPLATE = """
    You are an AI assistant that explains legal topics in simple, everyday language. Your task is to answer the user's question clearly, based only on the text provided.

    CONTEXT:
    {context}
    ---
    QUESTION: {question}

    INSTRUCTIONS:
    1.  **Explain the answer in simple terms.** Avoid legal jargon. If you must use a legal term, explain it immediately.
    2.  Base your entire answer only on the information from the CONTEXT above.
    3.  If the context does not contain the answer, state that the information is not available in the provided text.
    4.  Conclude your response with a simple disclaimer: "Please remember, this is a simplified explanation for informational purposes and not legal advice. Always consult a legal professional for serious matters."

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
    # print(load_vector_store())
    query = """
    As per the principles enshrined in Section 11 of the Code of Civil Procedure, the doctrine of Res Judicata acts as a procedural bar against a court trying any suit or issue in which the matter directly and substantially in issue has been directly and substantially in issue in a former suit between the same parties, or between parties under whom they or any of them claim, litigating under the same title, in a Court competent to try such subsequent suit, and has been heard and finally decided by such Court. This is to ensure finality to litigation and prevent multiplicity of proceedings.
"""
    human_summarizer(query_text=query)
    # clear_database()