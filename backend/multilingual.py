import asyncio
import edge_tts
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def translate(prompt, lang):
    """
    Translates a given text to a specified language using LangChain and Google's Gemini.
    """
    # Note: I've corrected the model name to "gemini-1.5-flash"
    # as "gemini-2.5-flash" is not a valid model name.
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    # A more structured prompt template separates instructions from user input.
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert language translator. Your task is to translate the user's text into the specified language. Return only the translated text and nothing else."),
            ("human", "Translate the following text to {lang}:\n\n{text_to_translate}"),
        ]
    )

    # The parser ensures we get a clean string as the final output.
    output_parser = StrOutputParser()

    # Create the chain by piping the components together.
    chain = prompt_template | llm | output_parser

    # Invoke the chain with the required variables.
    translated_text = chain.invoke({
        "text_to_translate": prompt,
        "lang": lang
    })

    return translated_text


def translater(lang, script):
    model = None
    tran_script = None
    match lang:
        case "eng":
            model = "en-US-AriaNeural"
            tran_script = script
        case "hin":
            tran_script = translate(script, "hindi")
            model = "hi-IN-MadhurNeural"
        case "kan":
            tran_script = translate(script, "kannada")
            model = "kn-IN-SapnaNeural"
        case "tam":
            tran_script = translate(script, "tamil")
            model = "ta-IN-PallaviNeural"
        case "mal":
            tran_script = translate(script, "malayalam")
            model = "ml-IN-MidhunNeural"
        case "tel":
            tran_script = translate(script, "telugu")
            model = "te-IN-MohanNeural"

    return tran_script, model


async def generate_tts(text, voice_model):
    if not os.path.exists("static"):
        os.makedirs("static")

    output_file = "static/output.mp3"
    rate = "+10%"  # Adjusts the speed (negative values slow it down)

    communicate = edge_tts.Communicate(text, voice_model, rate=rate)
    await communicate.save(output_file)
    print(f"Speech saved as {output_file}")


async def generate_audio_output(text: str, lang: str):
    restext, voice_model = translater(lang=lang, script=text)
    await generate_tts(restext, voice_model)
    return restext, "static/output.mp3"