# Symbolic Dynamics Engine

Discretises ETF returns into symbolic sequences (Up/Down/Neutral). Computes topological entropy, forbidden words, and entropy rate. The predictability score combines entropy rate and forbidden word ratio – lower entropy and fewer forbidden words indicate more predictable dynamics, which aligns with momentum/trend signals.

- **Alphabet:** U (return > 0.5σ), D (return < -0.5σ), N (otherwise)
- **Measures:** topological entropy, entropy rate (bits), forbidden word count
- **Score:** (1 - entropy_rate) × (1 - forbidden_ratio) – higher = more predictable
- **Windows:** 63, 252, 504, 1008, 2016 days (best per ETF)
- **Output:** top 3 ETFs per universe by predictability score

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
