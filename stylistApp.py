# style_stylist_app.py
import os
import openai
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TONE_PROMPTS = {
    "formal": "Rewrite the following prompt in a formal, professional tone while keeping the intent identical:",
    "empathetic": "Rewrite the following prompt with an empathetic and warm tone suitable for a close friend:",
    "minimalist": "Rewrite the following prompt in a concise, minimalist style (short sentences, minimal adjectives):",
    "old-money-elegant": "Rewrite the following prompt in an elegant, old-money aesthetic tone: restrained, courteous, and polished:",
}

def transform_prompt(user_input, tone):
    modifier = TONE_PROMPTS.get(tone, TONE_PROMPTS["formal"])
    full_prompt = f"{modifier}\n\n\"{user_input}\"\n\nProduce only the rewritten prompt."
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":full_prompt}],
        temperature=1,
        max_tokens=200
    )
    return resp.choices[0].message.content.strip()

def generate_variations(user_input):
    results = {}
    for tone in TONE_PROMPTS:
        results[tone] = transform_prompt(user_input, tone)
    return results

with gr.Blocks() as demo:
    gr.Markdown("## Emotion-Tuned Prompt Stylist — rewrite prompts into multiple tones")
    inp = gr.Textbox(lines=3, placeholder="Enter a prompt (e.g. 'Write an apology email for missing a birthday')", label="Prompt")
    btn = gr.Button("Generate Styles")
    out_boxes = {tone: gr.Textbox(label=tone, interactive=False) for tone in TONE_PROMPTS}
    def on_click(text):
        return [generate_variations(text)[t] for t in TONE_PROMPTS]
    btn.click(on_click, inputs=[inp], outputs=list(out_boxes.values()))

if __name__ == "__main__":
    demo.launch()
