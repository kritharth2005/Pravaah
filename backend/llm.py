import asyncio
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
from multilingual import generate_audio_output

load_dotenv()


async def human_summarizer(query_text: str, lang: str):
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

    audio_path = await generate_audio_output(response.text, lang)

    return response.text, audio_path


async def human_advisor(query_text: str, lang: str):
    vector_store = Chroma(
        persist_directory="chroma_langchain_db",
        embedding_function=get_embedding_function(),
    )

    PROMPT_TEMPLATE = """
        ## ROLE & GOAL ##
        You are an AI Legal Advisor. Your goal is to analyze the user's situation based *exclusively* on the provided legal CONTEXT. You must break down the legal rules and apply them to the user's question in a clear, step-by-step manner using simple language.

        ## CONTEXT ##
        {context}

        ## USER'S QUESTION ##
        {question}

        ## INSTRUCTIONS & RULES ##
        1.  **Analyze, Don't Just Define:** Do not just explain the law. Apply the rules from the CONTEXT directly to the facts in the USER'S QUESTION.
        2.  **Simple Language is Crucial:** Explain everything in plain, everyday English. Avoid legal jargon. If you must use a legal term from the context, explain it immediately (e.g., "'Liability' just means who is legally at fault.").
        3.  **Follow a Clear Structure:** Organize your answer into the following sections:
            * **The Legal Question:** Briefly restate the user's main issue.
            * **The Relevant Law:** Explain the specific rule from the CONTEXT that applies.
            * **How the Law Applies to Your Situation:** This is the most important part. Connect the legal rule directly to the user's scenario.
            * **Conclusion:** Give a straightforward concluding thought based on your analysis.
        4.  **Strictly Context-Based:** Your entire analysis must be based ONLY on the provided CONTEXT. Do not use any outside knowledge.
        5.  **Handle Missing Information:** If the CONTEXT does not contain the information to answer the question, you must clearly state: "The provided text does not have the information needed to answer this question."

        ## CRITICAL DISCLAIMER ##
        You MUST end every response with the following disclaimer, exactly as written:
        "**Disclaimer:** I am an AI assistant, not a lawyer. This analysis is for informational purposes only, based on the text provided, and is not a substitute for professional legal advice. You should consult with a qualified legal professional for your specific situation."
    """

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20,
        },
    )

    results = retriever.get_relevant_documents(query_text)

    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
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

    audio_path = await generate_audio_output(response.text, lang)

    return response.text, audio_path


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
My elderly mother, Leela, who lives in Mangaluru, recently sold her valuable beachfront property to her long-time financial advisor, Suresh, for a price that is far below the market value. My mother is 80 years old, has been in poor health, and completely trusted Suresh with all her financial decisions. I feel he took advantage of her trust and her age. We want to challenge the sale. Based on Indian law, is there a principle that can help us argue that the contract is unfair because of the relationship between my mother and her advisor?
"""
    # human_summarizer(query_text=query)
    asyncio.run(human_advisor(query_text=query, lang="hin"))
# clear_database()
