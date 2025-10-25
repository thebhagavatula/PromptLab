import os
import openai
from difflib import SequenceMatcher
import argparse
import time
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

BASE_PROMPT = """Question: {question}
Answer the question step-by-step, show your reasoning and conclusion."""

SELF_CRITIQUE_PROMPT = """You produced this answer:

{answer}

Now, check your answer step-by-step. Identify any incorrect facts, contradictions, or unjustified points. For each issue found, Explain why is is problematic and propose a clarified/corrected answer. If the original answer is more apt, say "No issues found" and restate the concise final answer.
"""

def callModel(prompt, model="gpt-4o-mini", temperature=0.3, max_tokens=600):
    resp = openai.Completion.create(
        model=model,
        messages=[{"role":"user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp["choices"][0]["message"]["content"].strip()

def similarity (a, b):
    return SequenceMatcher(None, a, b).ratio()

def runDebug(question, model="gpt-4o-mini"):
    base = BASE_PROMPT.format(question=question)
    print(">>> Asking base question...\n")
    ans = callModel(base, model=model, temperature=0.3)
    print(">>> Answering base question...\n")
    print(ans, "\n")
    time.sleep(0.7)

    print(">>> Asking model to self-critique...\n")
    critiquePrompt = SELF_CRITIQUE_PROMPT.format(answer=ans)
    critique = callModel(critiquePrompt, model=model, temperature=0.3)
    print(">>> Critique's answer...\n")
    print(critique, "\n")

    sim = similarity(ans, critique)
    print(f"Similarity between answer and critique text (rough): {sim:.3f}")

    #simple heuristic
    issuesFound = ("no issues found" not in critique.lower())
    score = 0.5 + (0.5 * (1 - sim)) #lower similarity -> more changes -> lower consistency
    if issuesFound:
        score -= 0.2
    score = max(0.0, min(1.0, score))
    print(f"Heuristic consistency score (0 = bad to 1 = excellent): {score:.2f}")

    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--q", required=True, help="Question to ask the model (wrap in quotes)")
        parser.add_argument("--model", default="gpt-4o-mini", help="Model to call")
        args = parser.parse_args()
        runDebug(args.q, args.model)
