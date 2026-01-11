import os
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig

load_dotenv()
# Create Gemini client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def _build_context(chunks):
    blocks = []
    for c in chunks:
        blocks.append(
            f"[Page {c.page} | {c.modality.upper()}]\n{c.content}"
        )
    return "\n\n".join(blocks)


def _build_prompt(context: str, question: str):
    return f"""
You are a document analysis assistant.

STRICT RULES:
- Answer ONLY using the provided context.
- If the answer is not present, say:
  "Information not found in the document."
- Cite sources using (Page X).
- Do NOT use external knowledge.

Context:
{context}

Question:
{question}

Answer:
"""


def answer_question(chunks, question: str):
    context = _build_context(chunks)
    prompt = _build_prompt(context, question)

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=512
        )
    )

    return response.text
