# Deploying FinSight

## 1. Streamlit Community Cloud (free, easiest)

1. Push the repo to GitHub (already done).
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**.
3. Repository `batsyahan/FinSight`, branch `main`, main file `app/streamlit_app.py`.
4. **Advanced settings → Secrets**, paste (TOML):
   ```toml
   FINSIGHT_LLM = "openai"
   OPENAI_BASE_URL = "https://api.groq.com/openai/v1"
   OPENAI_API_KEY = "gsk_..."
   FINSIGHT_MODEL = "openai/gpt-oss-120b"
   FINSIGHT_DATA = "yfinance"
   ```
   Streamlit exposes root-level secrets as environment variables, which is what FinSight reads.
5. Deploy. Dependencies come from `requirements.txt`. Put the public URL at the top of the README.

> A public app spends your API quota whenever anyone uses it. For a portfolio demo, consider
> `FINSIGHT_LLM = "mock"` with `FINSIGHT_DATA = "yfinance"` (real data, no LLM cost), or keep the
> app private and share it with reviewers only.

## 2. Docker (API + UI + Redis)

```bash
cp .env.example .env        # fill in your keys
docker compose up --build   # API on http://localhost:8000/docs, UI on http://localhost:8501
```

## 3. Hugging Face Spaces (Docker)

Create a Space with the **Docker** SDK, push this repo to it, and add your keys as Space secrets.
Spaces expect port 7860, so add to the Space's Dockerfile (or set in Space settings):
```dockerfile
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=7860"]
```

## 4. Any VM / Render / Railway / Fly.io

The `Dockerfile` serves the REST API (`uvicorn finsight.api:app`) on port 8000 with a health
check at `/healthz`. Set `FINSIGHT_API_KEY` so the API is not open to everyone.
