import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os
from io import BytesIO
from openai import OpenAI
import matplotlib.ticker as ticker
import re

# --- Setup OpenAI client ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Streamlit Config ---
st.set_page_config(page_title="Retirement Planning AI Advisor", layout="centered")
st.title("🤖 Retirement Planning AI Advisor")
st.write("""
Ask questions about **investing or retiring in Australia**, and the AI will analyze and visualize your situation.

Example questions you can ask:
- "Can I retire at 60 with 200k AUD and spend 40k a year?"
- "How much should I save annually to retire comfortably in Sydney?"
- "If I invest $10k per year until 67, how much will I have?"
- "Will $50,000 a year last until I'm 85?"
- "I’m 40 with 150k in super, plan to retire at 63 and spend 55k per year."
""")

# --- User Prompt ---
prompt = st.text_area("💬 Ask your question about investing/retiring in Australia:", placeholder="About investing/retiring in Australia")

# --- Regex-based parameter extraction ---
def extract_params(text):
    age_match = re.search(r'age\s*(?:is\s*)?(\d{2})', text, re.IGNORECASE)
    retire_match = re.search(r'retire (?:at|by)?\s*(\d{2})', text, re.IGNORECASE)
    savings_match = re.search(r'(\d{1,3}[, ]?\d{3,})(?:\s*(?:AUD|dollars|\$))?', text, re.IGNORECASE)
    contrib_match = re.search(r'(?:save|contribut(?:e|ing))\s*(?:\$)?(\d{1,3}[, ]?\d{3,})', text, re.IGNORECASE)
    expense_match = re.search(r'(?:spend|expense|spending)\s*(?:\$)?(\d{1,3}[, ]?\d{3,})', text, re.IGNORECASE)

    def clean(val):
        return int(val.replace(',', '').replace(' ', '')) if val else None

    return {
        "age": clean(age_match.group(1)) if age_match else None,
        "retire_age": clean(retire_match.group(1)) if retire_match else None,
        "savings": clean(savings_match.group(1)) if savings_match else None,
        "annual_contrib": clean(contrib_match.group(1)) if contrib_match else None,
        "annual_expense": clean(expense_match.group(1)) if expense_match else None
    }

# --- AI Advisor Function ---
def get_ai_explanation(prompt_text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional Australian retirement financial advisor."},
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.3,
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error calling AI model: {e}"

# --- Pension Simulation Function ---
def simulate_growth(start, annual_contrib, retire_age, age, expense, rate=0.06, inflation=0.02, lifespan=88):
    balance = start
    balances = []
    years_to_retire = retire_age - age
    years_after = lifespan - retire_age

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

# --- Generate Professional Plot ---
def generate_plot(sim_result, start_age, retire_age):
    total_years = len(sim_result)
    age_labels = list(range(start_age, start_age + total_years))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(age_labels, sim_result, marker='o', linewidth=2.5, color="#0066CC", label="Projected Balance")
    ax.axvline(x=retire_age, linestyle='--', color='gray', linewidth=1.5)
    ax.text(retire_age + 0.5, max(sim_result)*0.9, 'Retirement Age', color='gray', fontsize=12)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
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
    fig.tight_layout()
    return fig

# --- Download Function ---
def convert_plot_to_bytes(fig, filetype='png'):
    buf = BytesIO()
    fig.savefig(buf, format=filetype, bbox_inches='tight')
    buf.seek(0)
    return buf

# --- Main Logic ---
if st.button("🔍 Analyze"):
    if prompt.strip():
        # Step 1: Extract parameters dynamically
        params = extract_params(prompt)
        st.markdown("#### 🧾 Extracted parameters")
        st.json(params)

        # Step 2: Apply defaults for missing values
        age = params.get("age") or 35
        retire_age = params.get("retire_age") or 65
        savings = params.get("savings") or 120000
        annual_contrib = params.get("annual_contrib") or 10000
        annual_expense = params.get("annual_expense") or 50000

        # Step 3: AI Text Analysis
        st.markdown("### 🤖 AI Advisor’s Response")
        result = get_ai_explanation(prompt)
        st.write(result)

        # Step 4: Dynamic Simulation
        st.markdown("### 📈 Projected Pension Balance (Simulation)")
        sim_result, start_age = simulate_growth(
            start=savings,
            annual_contrib=annual_contrib,
            retire_age=retire_age,
            age=age,
            expense=annual_expense
        )
        fig = generate_plot(sim_result, start_age, retire_age)
        st.pyplot(fig)

        # Step 5: Export Buttons
        png_buf = convert_plot_to_bytes(fig, 'png')
        pdf_buf = convert_plot_to_bytes(fig, 'pdf')
        st.download_button("📸 Download Chart (PNG)", data=png_buf, file_name="retirement_projection.png", mime="image/png")
        st.download_button("📄 Download Report (PDF)", data=pdf_buf, file_name="retirement_projection.pdf", mime="application/pdf")

        # Step 6: Result Summary
        if sim_result[-1] <= 0:
            st.error(f"⚠️ Your savings may run out by age {start_age + len(sim_result) - 1}.")
        else:
            st.success(f"✅ Your savings may last until age {start_age + len(sim_result) - 1}.")
    else:
        st.warning("Please enter a question about Australian retirement or investment.")
