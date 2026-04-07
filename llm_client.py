"""
LLM Client — Google Gemini only.

Setup:
    pip install google-generativeai
    set GEMINI_API_KEY=your-key-here        (Windows)
    export GEMINI_API_KEY=your-key-here     (Mac/Linux)

Free tier: 15 req/min, 1M tokens/day.
Get key : https://aistudio.google.com/apikey
"""

import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash"

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

LLM_PROVIDER = "gemini" if (GEMINI_API_KEY and GEMINI_AVAILABLE) else "none"


def llm_complete(
    user: str,
    system: str = "",
    max_tokens: int = 1000,
    temperature: float = 0.7,
) -> str | None:
    """Call Gemini. Returns response text or None."""
    if LLM_PROVIDER == "none":
        return None
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config=genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
            system_instruction=system if system else None,
        )
        return model.generate_content(user).text
    except Exception as e:
        print(f"  [Gemini] API error: {e}")
        return None


def llm_status() -> str:
    if LLM_PROVIDER == "gemini":
        return f"Gemini ({GEMINI_MODEL}) — key: ...{GEMINI_API_KEY[-6:]}"
    msgs = []
    if not GEMINI_AVAILABLE:
        msgs.append("pip install google-generativeai")
    if not GEMINI_API_KEY:
        msgs.append("set GEMINI_API_KEY")
    return "No LLM — " + " | ".join(msgs)


if __name__ == "__main__":
    print(f"Provider : {LLM_PROVIDER}")
    print(f"Status   : {llm_status()}")
    if LLM_PROVIDER == "none":
        print()
        print("Get a free Gemini key at: https://aistudio.google.com/apikey")
        print("Then: pip install google-generativeai")
    else:
        result = llm_complete(
            system="Be concise.",
            user="Say exactly: Gemini connection working",
            max_tokens=20,
        )
        print(f"Test     : {result}")
