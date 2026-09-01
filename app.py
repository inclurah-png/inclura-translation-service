# =======================================================
# Inclura Translation Service
# MADLAD-400 3B + CTranslate2
# =======================================================

import os

import ctranslate2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentencepiece import SentencePieceProcessor
from huggingface_hub import snapshot_download


# =======================================================
# Configuration
# =======================================================

MODEL_NAME = os.getenv(
    "TRANSLATION_MODEL",
    "santhosh/madlad400-3b-ct2",
)

MAX_TEXT_LENGTH = int(
    os.getenv(
        "MAX_TEXT_LENGTH",
        "5000",
    )
)


# =======================================================
# Language mapping
# =======================================================
#
# MADLAD uses the target language in the form:
#
# <2LANGUAGE>
#
# Examples:
#
# <2yo> Yoruba
# <2ig> Igbo
# <2hau> Hausa
#
# =======================================================

SUPPORTED_LANGUAGES = {
    "en",
    "es",
    "fr",
    "pt",
    "de",
    "ar",
    "zh",
    "zh-TW",
    "ja",
    "hi",
    "ru",
    "it",
    "nl",
    "sw",
    "yo",
    "ig",
    "ha",
    "pcm",
    "ko",
    "vi",
    "th",
    "id",
    "ms",
    "bn",
    "tr",
    "af",
    "am",
    "zu",
    "xh",
    "so",
    "st",
    "tn",
    "ts",
    "ss",
    "rw",
    "lg",
    "ln",
    "ff",
    "wo",
    "tw",
    "bm",
    "ee",
}


# =======================================================
# FastAPI
# =======================================================

app = FastAPI(
    title="Inclura Translation Service",
    version="1.0.0",
)


# =======================================================
# Request schema
# =======================================================

class TranslationRequest(BaseModel):
    text: str
    target: str


# =======================================================
# Model loading
# =======================================================
#
# The model is loaded once when the service starts.
#
# This follows the CTranslate2 loading approach
# documented for MADLAD-400.
# =======================================================

print(
    "Loading translation model:",
    MODEL_NAME,
)

model_path = snapshot_download(
    MODEL_NAME
)

print(
    "Model downloaded:",
    model_path,
)


tokenizer = SentencePieceProcessor()

tokenizer.load(
    f"{model_path}/sentencepiece.model"
)


translator = ctranslate2.Translator(
    model_path
)

print(
    "Inclura Translation Service ready."
)


# =======================================================
# Health
# =======================================================

@app.get("/")
def health():
    return {
        "service":
            "Inclura Translation Service",

        "model":
            MODEL_NAME,

        "status":
            "running",

        "languages":
            len(SUPPORTED_LANGUAGES),
    }


# =======================================================
# Translation
# =======================================================

@app.post("/translate")
def translate(
    request: TranslationRequest
):

    text = request.text.strip()

    target = (
        request.target
        .strip()
        .lower()
    )


    # ---------------------------------------------------
    # Validation
    # ---------------------------------------------------

    if not text:

        raise HTTPException(
            status_code=400,
            detail="Text is required.",
        )


    if len(text) > MAX_TEXT_LENGTH:

        raise HTTPException(
            status_code=413,
            detail=(
                "Text exceeds the maximum "
                "allowed length."
            ),
        )


    if target == "zh-tw":

        target = "zh-TW"


    if target not in SUPPORTED_LANGUAGES:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported target language: "
                f"{target}"
            ),
        )


    # ---------------------------------------------------
    # MADLAD target token
    # ---------------------------------------------------

    input_text = (
        f"<2{target}> {text}"
    )


    input_tokens = tokenizer.encode(
        input_text,
        out_type=str,
    )


    # ---------------------------------------------------
    # Translation
    # ---------------------------------------------------

    results = translator.translate_batch(
        [input_tokens],

        batch_type="tokens",

        beam_size=1,

        no_repeat_ngram_size=1,

        repetition_penalty=2,

        max_batch_size=1024,
    )


    if not results:

        raise HTTPException(
            status_code=502,
            detail=(
                "Translation model "
                "returned no result."
            ),
        )


    hypotheses = (
        results[0].hypotheses
    )


    if not hypotheses:

        raise HTTPException(
            status_code=502,
            detail=(
                "Translation model "
                "returned no translation."
            ),
        )


    translated_text = (
        tokenizer.decode(
            hypotheses[0]
        )
        .strip()
    )


    if not translated_text:

        raise HTTPException(
            status_code=502,
            detail=(
                "Translation model "
                "returned empty text."
            ),
        )


    return {
        "translatedText":
            translated_text,

        "targetLanguage":
            target,

        "model":
            "MADLAD-400-3B-CT2",

        "provider":
            "Inclura Translation Service",
}
    }
