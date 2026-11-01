"""Settings: model, data source, workflow limits and display preferences (persisted locally).
API keys are never stored by the app: set them in .env or your host's secrets."""

import os

import streamlit as st

from finsight.ui import components as C
from finsight.ui import data as D
from finsight.ui import shell

store, s = shell.store(), shell.settings()
saved = store.get_setting("app_settings", {}) or {}

st.html(C.page_header("Settings", "Saved locally in the app database. Environment variables provide the defaults."))
left, right = st.columns([1.3, 1], gap="medium")
with left:
    with st.container(key="glass_set_main"):
        st.html(C.panel_head("Research workflow", "bot"))
        backends = ["mock", "openai", "anthropic"]
        llm = st.selectbox("LLM backend", backends, index=backends.index(s.llm) if s.llm in backends else 0,
                           help="mock = deterministic offline model; openai = any OpenAI-compatible API (Groq, "
                                "OpenAI, Ollama...); anthropic = Claude", key="st_llm")
        model = st.text_input("Model", s.model, key="st_model", placeholder="e.g. openai/gpt-oss-120b")
        data = st.selectbox("Market data", ["sample", "yfinance"], index=1 if s.data == "yfinance" else 0, key="st_data")
        max_rev = st.slider("Max Writer to Critic revisions", 0, 4, s.max_revisions, key="st_rev")
        bench = st.selectbox("Overview benchmark", list(D.BENCHMARKS),
                             index=list(D.BENCHMARKS).index(store.get_setting("benchmark", "NIFTY 50")), key="st_bench")
        initials = st.text_input("Profile initials", store.get_setting("user_initials", "FS"), max_chars=2,
                                 key="st_init")
        if st.button("Save settings", type="primary", icon=":material/save:", key="st_save"):
            store.set_setting("app_settings", {"llm": llm, "model": model.strip(), "data": data, "max_revisions": max_rev})
            store.set_setting("benchmark", bench)
            store.set_setting("user_initials", (initials or "FS").strip()[:2].upper())
            st.cache_data.clear()
            st.toast("Settings saved")
            st.rerun()
with right:
    with st.container(key="glass_set_keys"):
        st.html(C.panel_head("Credentials", "shield", "slate"))
        for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY", "SEC_USER_AGENT", "FINSIGHT_EMBED_MODEL"):
            present = bool(os.getenv(var))
            st.html(f'<div class="fs-row"><span class="k">{var}</span>{C.pill("set" if present else "not set", "ok" if present else "neutral")}</div>')
        st.html(C.note("Keys are read from the environment (.env locally, Secrets on Streamlit Cloud) and are never "
                       "written to the database or shown here."))
    with st.container(key="glass_set_maint"):
        st.html(C.panel_head("Maintenance", "database", "slate"))
        st.html(C.kv_rows([("Database", str(store.path)), ("Saved overrides", ", ".join(f"{k}={v}" for k, v in saved.items()) or "none")]))
        if st.button("Clear cached market data", icon=":material/delete_sweep:", key="st_clear"):
            st.cache_data.clear()
            st.toast("Cache cleared")
        if st.button("Reset to environment defaults", icon=":material/restart_alt:", key="st_reset"):
            store.set_setting("app_settings", {})
            st.cache_data.clear()
            st.rerun()
