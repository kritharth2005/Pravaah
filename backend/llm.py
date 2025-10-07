import asyncio
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from torch import cuda
import os

from vector import (
    add_to_chroma,
    clear_database,
    get_embedding_function,
    load_documents,
    spilt_documents,
)
from multilingual import generate_audio_output

load_dotenv()

vector_store = Chroma(
    persist_directory="chroma_langchain_db",
    embedding_function=get_embedding_function(),
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


async def human_summarizer(query_text: str, lang: str):

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
    5.  **Start the explanation directly.** Do not begin your response with phrases like "Based on the information provided," or "According to the text."

    """

    results = vector_store.similarity_search_with_score(query=query_text, k=7)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    print("=" * 55)
    print(prompt)
    print("=" * 55)

    response = model.invoke(prompt)

    print("=" * 55)
    print(response.content)
    print("=" * 55)

    restext, audio_path = await generate_audio_output(response.content, lang)

    return restext, audio_path


async def professional_summarizer(query_text: str, lang: str):

    PROMPT_TEMPLATE = """
        ## ROLE & GOAL ##
        You are an AI Legal Analyst. Your goal is to provide a concise and technically accurate summary of the legal principles contained within the provided CONTEXT for a professional legal audience.

        ## CONTEXT ##
        {context}

        ## USER'S QUESTION ##
        {question}

        ## INSTRUCTIONS & RULES ##
        1.  **Technical & Precise Language:** Summarize the legal text using precise legal terminology. Do not simplify or explain jargon; the audience is expected to understand it.
        2.  **Structured Summary:** Structure your response logically. Begin with the core legal principle, then enumerate the essential elements, conditions, or exceptions as presented in the text.
        3.  **Strictly Context-Based:** Your entire summary must be derived exclusively from the provided CONTEXT. Do not infer or add information not present in the text.
        4.  **Cite Sections:** You must cite the specific section numbers or clauses referenced in the context.
        5.  **Handle Missing Information:** If the CONTEXT does not contain the information relevant to the question, state that the information is not available in the provided text.
        6.  **Start the explanation directly.** Do not begin your response with phrases like "Based on the information provided," or "According to the text."

        ## PROFESSIONAL DISCLAIMER ##
        You MUST end every response with the following disclaimer, exactly as written:
        "**Disclaimer:** This AI-generated summary is for informational and preliminary review purposes only and is not a substitute for a complete reading of the source text or independent legal analysis."
        """

    results = vector_store.similarity_search_with_score(query=query_text, k=7)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    print("=" * 55)
    print(prompt)
    print("=" * 55)

    response = model.invoke(prompt)

    print("=" * 55)
    print(response.content)
    print("=" * 55)

    restext, audio_path = await generate_audio_output(response.content, lang)

    return restext, audio_path


async def human_advisor(query_text: str, lang: str):

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
        7.  **Start the explanation directly.** Do not begin your response with phrases like "Based on the information provided," or "According to the text."

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

    response = model.invoke(prompt)

    print("=" * 55)
    print(response.content)
    print("=" * 55)

    restext, audio_path = await generate_audio_output(response.content, lang)

    return restext, audio_path


async def professional_advisor(query_text: str, lang: str):

    PROMPT_TEMPLATE = """
        ## ROLE & GOAL ##
        You are a Specialist AI Legal Analyst. Your function is to provide a detailed and technical legal analysis for a legal professional. Your goal is to dissect the user's query, apply the relevant statutory provisions from the provided CONTEXT, and outline the legal reasoning, potential arguments, and conclusions.

        ## CONTEXT ##
        {context}

        ## USER'S QUERY ##
        {question}

        ## INSTRUCTIONS & RULES ##
        1.  **Technical & Precise Language:** Use accurate legal terminology and formal language appropriate for a lawyer or advocate. Do not simplify legal concepts.
        2.  **In-Depth Analysis:** Your analysis must be thorough. Go beyond a surface-level application. Identify the essential elements of the relevant legal provisions and meticulously apply them to the facts of the case.
        3.  **Identify Strengths and Weaknesses:** If possible, based on the context, identify potential counter-arguments or weaknesses in the legal position.
        4.  **Structure Your Response (IRAC Method):** Organize your analysis into the following formal sections:
            * **Issue:** Concisely state the central legal question(s) presented by the user's query.
            * **Rule:** State the relevant legal rule(s) and cite the specific sections from the CONTEXT verbatim.
            * **Application:** This is the core of your analysis. Systematically apply the rule to the facts. Analyze each element of the statute and connect it to the corresponding facts in the query.
            * **Conclusion:** Provide a reasoned legal conclusion based on your application of the rule to the facts.
        5.  **Strictly Context-Based:** Your entire analysis must be derived exclusively from the provided CONTEXT. Do not infer principles or cite case law not present in the text.
        6.  **Handle Missing Information:** If the CONTEXT is insufficient to form a complete analysis, explicitly state what information is missing and how it impacts the conclusion.
        7.  **Start the explanation directly.** Do not begin your response with phrases like "Based on the information provided," or "According to the text."

        ## PROFESSIONAL DISCLAIMER ##
        You MUST end every response with the following disclaimer, exactly as written:
        "**Disclaimer:** This AI-generated analysis is for informational and preliminary review purposes only. It is not a substitute for independent professional legal judgment and should not be cited as legal authority. Always conduct your own comprehensive research."
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

    response = model.invoke(prompt)

    print("=" * 55)
    print(response.content)
    print("=" * 55)

    restext, audio_path = await generate_audio_output(response.content, lang)

    return restext, audio_path


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
#     query = """
#     Provide a technical summary of the grounds upon which a 'Perpetual Injunction' can be granted, as enumerated in the Specific Relief Act, 1963
# """
    # human_summarizer(query_text=query)
    # asyncio.run(professional_summarizer(query_text=query, lang="mal"))
# clear_database()
