import asyncio
import edge_tts
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def translate(prompt, lang):
    system_prompt = """
        You are a language translator.
        You will take in a prompt to translate and the language to which it shloud be translated.
        Just return the translated text.
    """

    model = genai.GenerativeModel("gemini-2.5-flash")
    chat = model.start_chat()
    chat.send_message(system_prompt)

    user_input = f"{prompt}: translate this to this language: {lang}"
    resp = chat.send_message(user_input)

    return resp.text


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