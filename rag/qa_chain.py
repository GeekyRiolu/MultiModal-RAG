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
You are a document question-answering system.

Answer the question using ONLY the information provided in the context.

Rules:
- Base every statement strictly on the context
- Do NOT use outside knowledge
- Do NOT speculate
- If the answer is not present, say:
  "Information not found in the document."

Answering guidelines:
- Provide a detailed and well-explained answer
- Break the answer into logical paragraphs or bullet points
- Clearly explain relationships, causes, or implications if mentioned
- Cite relevant pages using (Page X)

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


def generate_document_summary(chunks):
    context = _build_context(chunks[:10])

    prompt = f"""
Using ONLY the information in the context below, generate a clear and
concise summary of the document.

Guidelines:
- Do NOT add external knowledge
- Do NOT assume intent or audience
- Focus on key themes, risks, and findings
- Use bullet points (5–8 bullets)
- If information is missing, do not invent it

Context:
{context}

Summary:
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config={
            "temperature": 0.2,
            "max_output_tokens": 400,
        }
    )

    return response.text
