import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
from io import BytesIO
from openai import OpenAI
import re

# --- Setup OpenAI Client ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Streamlit Config ---
st.set_page_config(page_title="Retirement Planning AI Advisor", layout="centered")
st.title("🤖 Retirement Planning AI Advisor")
st.write("""
Ask questions about **investing or retiring in Australia**, and the AI will analyze and visualize your situation.

Example questions you can ask:
- "Can I retire at 60 with 200k AUD and spend 40k a year?"
- "How much should I save annually to retire comfortably in Sydney?"
- "Is it better to invest in real estate or superannuation in Australia?"
- "What’s the expected return of Australian pension funds?"
""")

# --- User Prompt ---
prompt = st.text_area("💬 Ask your question about investing/retiring in Australia:")

# --- Extract Parameters from Prompt ---
def extract_parameters(prompt_text):
    extract_prompt = f"""
You are a financial AI agent. Extract the following 5 numbers from this user question if available:
- current age
- retirement age
- current savings (AUD)
- annual contribution (AUD)
- annual expense/spending in retirement (AUD)

Respond ONLY in JSON like this:
{{"age": ..., "retire_age": ..., "savings": ..., "annual_contrib": ..., "annual_expense": ...}}. Use null for any missing value.
Question: "{prompt_text}"
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": extract_prompt}],
            temperature=0,
            max_tokens=150,
        )
        match = re.search(r"\{.*\}", response.choices[0].message.content, re.DOTALL)
        if match:
            import json
            return json.loads(match.group())
        else:
            return {}
    except Exception as e:
        return {}

# --- AI Explanation ---
def get_ai_explanation(prompt_text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful retirement advisor focusing on Australia."},
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.2,
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# --- Simulation Function ---
def simulate_growth(start, annual_contrib, retire_age, age, expense, rate=0.06, inflation=0.02, lifespan=88):
    balance = start
    balances = []
    years_to_retire = max(retire_age - age, 0)
    years_after = max(lifespan - retire_age, 0)

    for _ in range(years_to_retire):
        balance *= (1 + rate - inflation)
        balance += annual_contrib
        balances.append(balance)

    for _ in range(years_after):
        balance *= (1 + rate - inflation)
        balance -= expense
        balances.append(balance)
        if balance <= 0:
            break

    return balances, age

# --- Plot ---
def generate_plot(sim_result, start_age, retire_age):
    total_years = len(sim_result)
    age_labels = list(range(start_age, start_age + total_years))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(age_labels, sim_result, marker='o', linewidth=2.5, color="#0066CC", label="Projected Balance")
    ax.axvline(x=retire_age, linestyle='--', color='gray', linewidth=1.5)
    ax.text(retire_age + 0.5, max(sim_result)*0.9, 'Retirement Age', color='gray')

    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("#FAFAFA")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title("Projected Pension Balance Over Time", fontsize=16, fontweight='bold')
    ax.set_xlabel("Age", fontsize=13)
    ax.set_ylabel("Pension Balance (AUD)", fontsize=13)
    ax.tick_params(axis='both', labelsize=11)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend()

    return fig

# --- Download ---
def convert_plot_to_bytes(fig, filetype='png'):
    buf = BytesIO()
    fig.savefig(buf, format=filetype, bbox_inches='tight')
    buf.seek(0)
    return buf

# --- Run Button ---
if st.button("🔍 Analyze"):
    if prompt.strip():
        params = extract_parameters(prompt)
        explanation = get_ai_explanation(prompt)

        # Set defaults if missing
        age = params.get("age") or 35
        retire_age = params.get("retire_age") or 65
        savings = params.get("savings") or 120000
        contrib = params.get("annual_contrib") or 10000
        expense = params.get("annual_expense") or 50000

        st.markdown("### 🤖 AI Advisor’s Response")
        st.write(explanation)

        st.markdown("### 📈 Projected Pension Balance (Simulation)")
        sim_result, start_age = simulate_growth(
            start=savings,
            annual_contrib=contrib,
            retire_age=retire_age,
            age=age,
            expense=expense
        )
        fig = generate_plot(sim_result, start_age, retire_age)
        st.pyplot(fig)

        # Downloads
        png_buf = convert_plot_to_bytes(fig, 'png')
        pdf_buf = convert_plot_to_bytes(fig, 'pdf')
        st.download_button("📸 Download Chart (PNG)", data=png_buf, file_name="retirement_projection.png", mime="image/png")
        st.download_button("📄 Download Report (PDF)", data=pdf_buf, file_name="retirement_projection.pdf", mime="application/pdf")

        # Final Message
        if sim_result[-1] <= 0:
            st.error(f"⚠️ Your savings may run out by age {start_age + len(sim_result) - 1}.")
        else:
            st.success(f"✅ Your savings may last until age {start_age + len(sim_result) - 1}.")
    else:
        st.warning("Please enter a question about Australian retirement or investment.")
