import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os
from io import BytesIO
from openai import OpenAI
import matplotlib.ticker as ticker



# --- Setup OpenAI client ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Streamlit Config ---
st.set_page_config(page_title="Retirement Planning AI Advisor", layout="centered")
st.title("🤖 Retirement Planning AI Advisor")
st.write("""
Ask questions about **investing or retiring in Australia**, and the AI will analyze and visualize your situation.

Example questions you can ask:
- "Can I retire at 65 with 120k AUD in super?"
- "How much should I save annually to retire comfortably in Sydney?"
- "Is it better to invest in real estate or superannuation in Australia?"
- "How long will my savings last if I retire at 60?"
- "What’s the expected return of Australian pension funds?"
""")

# --- User Prompt ---
prompt = st.text_area("💬 Ask your question about investing/retiring in Australia:", placeholder="About investing/retiring in Australia")

# --- AI Assistant Function ---
def get_ai_explanation(prompt_text):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful retirement financial advisor focusing on Australia."},
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.6,
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error calling AI model: {e}"

# --- Pension Simulation Function ---
def simulate_growth(start=120000, annual_contrib=10000, retire_age=65, age=35, expense=50000, rate=0.06, inflation=0.02, lifespan=88):
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

def generate_plot(sim_result, start_age):


    total_years = len(sim_result)
    age_labels = list(range(start_age, start_age + total_years))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(age_labels, sim_result, marker='o', linewidth=2.5, color="#0066CC", label="Projected Balance")
    ax.axvline(x=65, linestyle='--', color='gray', linewidth=1.5)
    ax.text(65 + 0.5, max(sim_result)*0.85, 'Retirement Age', color='gray', fontsize=12)


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

    max_idx = np.argmax(sim_result)
    peak_y = sim_result[max_idx]
    peak_x = age_labels[max_idx]
    ax.annotate(
        f"Peak: ${int(peak_y):,}",
        xy=(peak_x, peak_y),
        xytext=(peak_x + 1, peak_y * 0.92),  # 向下移动注释
        arrowprops=dict(arrowstyle='->', color='green', lw=2),
        fontsize=12,
        color='green'
    )

    ax.legend(fontsize=11)
    fig.tight_layout()  # ✅ 防止所有重叠

    return fig


# --- Download Function ---
def convert_plot_to_bytes(fig, filetype='png'):
    buf = BytesIO()
    if filetype == 'png':
        fig.savefig(buf, format='png', bbox_inches='tight')
    else:
        fig.savefig(buf, format='pdf', bbox_inches='tight')
    buf.seek(0)
    return buf

# --- Main Logic ---
if st.button("🔍 Analyze"):
    if prompt.strip():
        st.markdown("### 🤖 AI Advisor’s Response")
        result = get_ai_explanation(prompt)
        st.write(result)

        st.markdown("### 📈 Projected Pension Balance (Simulation)")
        sim_result, start_age = simulate_growth()
        fig = generate_plot(sim_result, start_age)
        st.pyplot(fig)

        # --- Export Buttons ---
        png_buf = convert_plot_to_bytes(fig, 'png')
        pdf_buf = convert_plot_to_bytes(fig, 'pdf')

        st.download_button("📸 Download Chart (PNG)", data=png_buf, file_name="retirement_projection.png", mime="image/png")
        st.download_button("📄 Download Report (PDF)", data=pdf_buf, file_name="retirement_projection.pdf", mime="application/pdf")

        # --- Result message ---
        if sim_result[-1] <= 0:
            st.error(f"⚠️ Your savings may run out by age {start_age + len(sim_result) - 1}.")
        else:
            st.success(f"✅ Your savings may last until age {start_age + len(sim_result) - 1}.")
    else:
        st.warning("Please enter a question about Australian retirement or investment.")
