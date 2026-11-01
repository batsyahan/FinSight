"""Design tokens and global CSS for the finsight dashboard.

Styling hooks: CSS variables on :root, the documented ``st-key-<key>`` classes that Streamlit adds to
keyed containers (``glass_*`` panels, ``topnav``, ``rail``), and a handful of ``data-testid`` attributes.
No reliance on generated class names.
"""

from __future__ import annotations

TOKENS = {
    "bg": "#101211",
    "surface": "#171A19",
    "surface_2": "#1E2120",
    "glass": "rgba(27, 30, 29, 0.72)",
    "border": "rgba(255, 255, 255, 0.07)",
    "border_strong": "rgba(255, 255, 255, 0.12)",
    "lime": "#C5F65A",
    "lime_soft": "rgba(197, 246, 90, 0.13)",
    "sage": "#8FA88A",
    "amber": "#F5AE45",
    "amber_soft": "rgba(245, 174, 69, 0.14)",
    "coral": "#EE6B61",
    "coral_soft": "rgba(238, 107, 97, 0.14)",
    "slate": "#7D8A99",
    "text": "#ECEDE7",
    "muted": "#8D928E",
    "faint": "#5E635F",
}
RADIUS = {"sm": "8px", "md": "12px", "lg": "16px", "xl": "20px", "pill": "999px"}
SPACE = {"1": "4px", "2": "8px", "3": "12px", "4": "16px", "5": "20px", "6": "24px", "8": "32px"}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {
  --bg: %(bg)s; --surface: %(surface)s; --surface-2: %(surface_2)s; --glass: %(glass)s;
  --border: %(border)s; --border-strong: %(border_strong)s; --lime: %(lime)s; --lime-soft: %(lime_soft)s;
  --sage: %(sage)s; --amber: %(amber)s; --amber-soft: %(amber_soft)s; --coral: %(coral)s;
  --coral-soft: %(coral_soft)s; --slate: %(slate)s; --text: %(text)s; --muted: %(muted)s; --faint: %(faint)s;
  --r-sm: 8px; --r-md: 12px; --r-lg: 16px; --r-xl: 20px;
}
html, body, .stApp, button, input, textarea, select { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }
.stApp {
  background:
    radial-gradient(1200px 600px at 85%% -10%%, rgba(197,246,90,0.045), transparent 60%%),
    radial-gradient(900px 500px at -10%% 110%%, rgba(143,168,138,0.05), transparent 60%%),
    var(--bg);
  color: var(--text);
}
header[data-testid="stHeader"] { background: transparent; height: 0; }
[data-testid="stMainBlockContainer"], .block-container { padding-top: 18px; padding-bottom: 32px; max-width: 1480px; }
[data-testid="stSidebar"] { background: rgba(16,18,17,0.92); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: 8px; }
[data-testid="stSidebarNav"] { display: none; }

/* ---------- glass panels: st.container(key="glass_...") ---------- */
div[class*="st-key-glass"] {
  background: var(--glass); border: 1px solid var(--border); border-radius: var(--r-lg);
  padding: 18px 20px; backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  box-shadow: 0 1px 0 rgba(255,255,255,0.03) inset, 0 12px 32px rgba(0,0,0,0.28);
}
div[class*="st-key-glass"] div[class*="st-key-glass"] { box-shadow: none; background: rgba(255,255,255,0.025); }

/* ---------- top navigation ---------- */
.st-key-topbar { padding: 2px 0 10px 0; }
.st-key-topnav { background: rgba(255,255,255,0.035); border: 1px solid var(--border); border-radius: 999px;
  padding: 4px 6px; }
.st-key-topnav [data-testid="stHorizontalBlock"] { gap: 2px; flex-wrap: nowrap; }
.st-key-topnav a { border-radius: 999px; padding: 4px 10px; justify-content: center; }
.st-key-topnav a:hover { background: rgba(255,255,255,0.06); }
.st-key-topnav a p, .st-key-topnav a span { font-size: 13px; color: var(--muted); white-space: nowrap; }
.fs-pill-active { display: block; text-align: center; background: rgba(255,255,255,0.10); color: var(--text);
  border-radius: 999px; padding: 6px 10px; font-size: 13px; font-weight: 500; white-space: nowrap; }

/* ---------- sidebar rail ---------- */
.st-key-rail a { border-radius: 10px; padding: 6px 10px; }
.st-key-rail a:hover { background: rgba(255,255,255,0.05); }
.st-key-rail a p, .st-key-rail a span { color: var(--muted); font-size: 13.5px; }
.fs-rail-active { display: flex; align-items: center; gap: 10px; background: var(--lime-soft); color: var(--lime);
  border-radius: 10px; padding: 8px 12px; font-size: 13.5px; font-weight: 600; margin: 2px 0; }

/* ---------- buttons and inputs ---------- */
[data-testid="stBaseButton-primary"] { background: var(--lime); color: #111410; border: none; font-weight: 600;
  border-radius: 999px; }
[data-testid="stBaseButton-primary"]:hover { background: #d4fb7a; color: #111410; }
[data-testid="stBaseButton-secondary"] { border-radius: 999px; border: 1px solid var(--border-strong);
  background: rgba(255,255,255,0.03); }
[data-testid="stBaseButton-secondary"]:hover { border-color: var(--lime); color: var(--lime); }
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input { border-radius: 10px; }
[data-testid="stTabs"] [role="tablist"] { gap: 4px; border-bottom: none; }
[data-testid="stTabs"] button[role="tab"] { border-radius: 999px; padding: 6px 14px; background: rgba(255,255,255,0.03); }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { background: var(--lime-soft); color: var(--lime); }
[data-testid="stTabs"] [data-baseweb="tab-highlight"], [data-testid="stTabs"] [data-baseweb="tab-border"] { display: none; }
[data-testid="stExpander"] details { border: 1px solid var(--border); border-radius: var(--r-md); background: rgba(255,255,255,0.02); }
[data-testid="stDataFrame"] { border-radius: var(--r-md); overflow: hidden; }

/* ---------- components (finsight.ui.components) ---------- */
.fs-brand { display: flex; align-items: center; gap: 10px; font-weight: 600; font-size: 20px; letter-spacing: -0.2px; }
.fs-brand .dot { color: var(--lime); }
.fs-title { font-size: 30px; font-weight: 600; line-height: 1.1; letter-spacing: -0.6px; margin: 2px 0 6px 0; }
.fs-sub { color: var(--muted); font-size: 13px; margin: 0 0 6px 0; }
.fs-back { color: var(--muted); font-size: 12.5px; margin-bottom: 2px; }
.fs-panel-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 10px; }
.fs-panel-title { display: flex; align-items: center; gap: 10px; font-weight: 600; font-size: 15px; }
.fs-icon { display: inline-flex; align-items: center; justify-content: center; width: 30px; height: 30px;
  border-radius: 9px; background: var(--lime-soft); color: var(--lime); flex: none; }
.fs-icon.amber { background: var(--amber-soft); color: var(--amber); }
.fs-icon.coral { background: var(--coral-soft); color: var(--coral); }
.fs-icon.slate { background: rgba(125,138,153,0.16); color: #A9B6C4; }
.fs-icon.lg { width: 44px; height: 44px; border-radius: 12px; }
.fs-kpi { display: flex; align-items: center; gap: 14px; background: var(--glass); border: 1px solid var(--border);
  border-radius: var(--r-lg); padding: 14px 16px; min-height: 84px; }
.fs-kpi .v { font-size: 24px; font-weight: 600; line-height: 1.1; }
.fs-kpi .l { color: var(--muted); font-size: 12.5px; line-height: 1.3; }
.fs-kpi .s { font-size: 12px; }
.fs-big { font-size: 30px; font-weight: 600; letter-spacing: -0.5px; }
.fs-muted { color: var(--muted); } .fs-faint { color: var(--faint); }
.fs-up { color: var(--lime); } .fs-down { color: var(--coral); } .fs-warn { color: var(--amber); }
.fs-small { font-size: 12px; } .fs-xs { font-size: 11px; }
.fs-pill { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 11.5px; font-weight: 600;
  border: 1px solid transparent; white-space: nowrap; }
.fs-pill.ok { background: var(--lime-soft); color: var(--lime); border-color: rgba(197,246,90,0.25); }
.fs-pill.warn { background: var(--amber-soft); color: var(--amber); border-color: rgba(245,174,69,0.28); }
.fs-pill.bad { background: var(--coral-soft); color: var(--coral); border-color: rgba(238,107,97,0.28); }
.fs-pill.neutral { background: rgba(255,255,255,0.05); color: var(--muted); border-color: var(--border); }
.fs-pill.info { background: rgba(125,138,153,0.15); color: #B5C1CE; border-color: rgba(125,138,153,0.3); }
.fs-chip { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 12px; color: var(--muted);
  background: rgba(255,255,255,0.04); border: 1px solid var(--border); margin-right: 6px; }
.fs-tl { position: relative; margin: 4px 0 0 0; }
.fs-tl-item { display: grid; grid-template-columns: 18px 1fr auto; gap: 10px; align-items: start; padding: 7px 0; position: relative; }
.fs-tl-item:not(:last-child)::before { content: ""; position: absolute; left: 6px; top: 22px; bottom: -8px; width: 2px;
  background: rgba(255,255,255,0.08); }
.fs-tl-dot { width: 14px; height: 14px; border-radius: 50%%; margin-top: 3px; border: 3px solid rgba(197,246,90,0.25);
  background: var(--lime); box-sizing: border-box; }
.fs-tl-dot.warn { background: var(--amber); border-color: rgba(245,174,69,0.25); }
.fs-tl-dot.bad { background: var(--coral); border-color: rgba(238,107,97,0.25); }
.fs-tl-dot.idle { background: #3a3e3c; border-color: rgba(255,255,255,0.06); }
.fs-tl-dot.run { background: var(--lime); animation: fs-pulse 1.4s ease-in-out infinite; }
@keyframes fs-pulse { 50%% { box-shadow: 0 0 0 5px rgba(197,246,90,0.14); } }
.fs-tl-name { font-weight: 600; font-size: 13.5px; }
.fs-tl-desc { color: var(--muted); font-size: 12px; line-height: 1.35; }
.fs-table { width: 100%%; border-collapse: collapse; font-size: 13px; }
.fs-table th { text-align: left; color: var(--muted); font-weight: 500; font-size: 11.5px; padding: 6px 8px;
  border-bottom: 1px solid var(--border); }
.fs-table td { padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.04); }
.fs-table td.num, .fs-table th.num { text-align: right; font-variant-numeric: tabular-nums; }
.fs-row { display: flex; justify-content: space-between; align-items: center; padding: 6px 0;
  border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 13px; }
.fs-row:last-child { border-bottom: none; }
.fs-row .k { color: var(--muted); } .fs-row .v { font-weight: 600; font-variant-numeric: tabular-nums; }
.fs-list-item { display: flex; gap: 12px; align-items: flex-start; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.04); }
.fs-list-item:last-child { border-bottom: none; }
.fs-list-item a { color: var(--text); text-decoration: none; } .fs-list-item a:hover { color: var(--lime); }
.fs-bullet { display: flex; gap: 10px; align-items: flex-start; padding: 5px 0; font-size: 13.5px; line-height: 1.45; }
.fs-bullet::before { content: ""; width: 7px; height: 7px; border-radius: 50%%; background: var(--lime); margin-top: 7px; flex: none; }
.fs-bar { height: 7px; border-radius: 999px; background: rgba(255,255,255,0.07); overflow: hidden; }
.fs-bar > span { display: block; height: 100%%; background: var(--lime); border-radius: 999px; }
.fs-empty { border: 1px dashed var(--border-strong); border-radius: var(--r-md); padding: 18px; text-align: center;
  color: var(--muted); font-size: 13px; }
.fs-memo h1 { font-size: 22px; } .fs-memo h2 { font-size: 15.5px; border-left: 3px solid var(--lime); padding-left: 10px;
  margin-top: 18px; } .fs-memo p, .fs-memo li { font-size: 13.8px; line-height: 1.6; color: #D8DAD3; }
.fs-memo abbr.cite { color: var(--lime); text-decoration: none; border-bottom: 1px dotted var(--lime); cursor: help; }
.fs-check { display: flex; gap: 10px; align-items: flex-start; padding: 6px 0; font-size: 13px; }
.fs-avatar { width: 34px; height: 34px; border-radius: 50%%; background: var(--lime); color: #111410; display: flex;
  align-items: center; justify-content: center; font-weight: 700; font-size: 13px; margin-left: auto; }
.fs-note { font-size: 11.5px; color: var(--faint); margin-top: 6px; }
@media (max-width: 900px) {
  .fs-title { font-size: 24px; } .fs-big { font-size: 24px; }
  .st-key-topnav { overflow-x: auto; }
  .fs-table { display: block; overflow-x: auto; white-space: nowrap; }
}
""" % TOKENS


def css() -> str:
    return f"<style>{CSS}</style>"
