# Infosys Limited (INFY.NS): Research Memo

## Summary
Infosys Limited is a large Indian technology services firm offering consulting, digital, AI, cloud, and engineering services across multiple industries [S1]. The company reported FY 2026 revenue of **USD 20.16 billion** and net income of **USD 3.31 billion** [S2]. Operating cash flow was **USD 4.04 billion** and free cash flow **USD 3.73 billion**, while total debt stood at **USD 0.97 billion** [S2]. The balance sheet shows equity of **USD 9.79 billion**, current assets of **USD 10.91 billion** versus current liabilities of **USD 5.52 billion**, indicating solid liquidity [S2]. The stock trades at **INR 1,185** with an EPS of **75.1** [S2]. Over the past two years the total return was **‑30.83%** with a CAGR of **‑17.11%** and a Sharpe ratio of **‑0.72**, reflecting weak risk‑adjusted performance [S5].

## Business Overview
Infosys provides a broad portfolio of services, including applied and generative AI, sustainability, blockchain, IoT, cybersecurity, digital supply chain, and cloud platforms such as Microsoft and SAP [S1]. Its diversified offering spans consulting, technology outsourcing, and digital transformation for clients worldwide, positioning the firm in high‑growth segments of the global IT services market.

## Financial Health
- **Revenue & Profitability:** FY 2026 revenue of **USD 20.16 billion** and net income of **USD 3.31 billion** [S2]; revenue growth of **4.57%** [S7]; net margin **16.44%** and ROE **33.85%** [S7].  
- **Cash Flow:** Operating cash flow of **USD 4.04 billion** and free cash flow of **USD 3.73 billion** [S2]; free‑cash‑flow margin **18.52%** [S7].  
- **Balance Sheet:** Total equity **USD 9.79 billion**, current assets **USD 10.91 billion** versus current liabilities **USD 5.52 billion**, yielding a current ratio of **1.98** [S2, S7]; total debt **USD 0.97 billion**, debt‑to‑equity **0.1** [S2, S7].  
- **Valuation:** Forward‑looking P/E **15.78** and market price **INR 1,185** per share with EPS **75.1** [S2, S7].

## Price Performance and Backtest
- **Historical Returns:** Total return over the last two years is **‑30.83%** with a CAGR of **‑17.11%** and annualized volatility of **28.7%** [S5].  
- **Risk Metrics:** Maximum drawdown of **‑48.17%** (as of July 1 2026) and a Sharpe ratio of **‑0.72**, indicating returns have not covered the risk‑free rate after adjusting for volatility [S5].  
- **Strategy Backtest:** A 50/200 SMA crossover strategy underperformed buy‑and‑hold by **0.31%**, delivering **‑18.16%** versus **‑17.85%**, and was in the market only **19.3%** of the time [S6].

## Key Risks
- **Market‑price risk:** Deep drawdown flag (**‑48.17%**) and poor risk‑adjusted return dominate the risk profile, reflecting high equity market volatility and weak performance relative to risk‑free rates [S5, S8].  
- **Financial risk:** Mitigated by low leverage (debt‑to‑equity **0.1**), strong liquidity (current ratio **1.98**), and robust cash generation [S2, S7].  
- **Qualitative risk:** No recent governance, regulatory, or management‑change concerns identified [S3, S4].

## Outlook
**Neutral** – The company’s fundamentals are solid, with healthy profitability, strong cash flow, and a low‑debt balance sheet [S2, S7]. However, recent market performance has been poor, with a large drawdown and negative Sharpe ratio, suggesting that price recovery may be challenging in the near term [S5, S8]. The neutral view reflects confidence in the business model but acknowledges significant market‑price risk.

## Disclaimer
This memo is for educational purposes only and does not constitute financial advice or a recommendation to buy or sell any securities.

---

### Evidence

- **[S1]** INFY.NS company profile (`get_company_profile`)
- **[S2]** INFY.NS financial statements (`get_fundamentals`)
- **[S3]** INFY.NS recent news (`get_news`)
- **[S4]** INFY.NS recent news (`get_news`)
- **[S5]** INFY.NS price statistics (`get_price_stats`)
- **[S6]** INFY.NS SMA crossover backtest (`run_backtest`)
- **[S7]** INFY.NS fundamental ratios (`compute_ratios`)
- **[S8]** INFY.NS rule-based risk flags (`risk_rules`)
