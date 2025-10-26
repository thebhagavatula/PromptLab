import os
from openai import OpenAI
from difflib import SequenceMatcher
import argparse
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_PROMPT = """Question: {question}

Answer the question step-by-step, show your reasoning and conclusion."""

SELF_CRITIQUE_PROMPT = """You produced this answer:

{answer}

Now, check your answer step-by-step. Identify any incorrect facts, contradictions, or unjustified leaps. 
For each issue found, explain why it is problematic and propose a corrected/clarified answer. 
If the original answer is correct, say "No issues found" and restate the concise final answer.
"""

def call_model(prompt, model="gpt-4o-mini", temperature=1, max_tokens=600):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            timeout=20
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[!] API Error: {e}")
        return "Error generating response."

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def run_debug(question, model="gpt-4o-mini"):
    base = BASE_PROMPT.format(question=question)
    print(">>> Asking base question...\n")
    ans = call_model(base, model=model, temperature=1)
    print(">>> Answering base question...\n")
    print(ans, "\n")
    time.sleep(0.8)

    print(">>> Asking model to self-critique...\n")
    critique_prompt = SELF_CRITIQUE_PROMPT.format(answer=ans)
    critique = call_model(critique_prompt, model=model, temperature=1)
    print(">>> Answering critique...\n")
    print(critique, "\n")

    # Quick fix: handle empty / placeholder answers
    if not ans.strip() or ans.lower().startswith(("mock_response", "error", "i don't have the answer")):
        sim = 0.0
        score = 0.0
    else:
        sim = similarity(ans, critique)
        issues_found = ("no issues found" not in critique.lower())
        score = 0.5 + 0.5*(1 - sim)
        if issues_found:
            score -= 0.2
        score = max(0.0, min(1.0, score))

    print(f"Similarity between answer and critique text (rough): {sim:.3f}")
    print(f"Heuristic consistency score (0=bad to 1=excellent): {score:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", required=True, help="Question to ask the model (wrap in quotes)")
    parser.add_argument("--model", default="gpt-4o-mini", help="model id to call")
    args = parser.parse_args()
    run_debug(args.q, model=args.model)
