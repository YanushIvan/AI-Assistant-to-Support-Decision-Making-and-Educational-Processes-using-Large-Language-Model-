"""
Appends new EEX-sourced QA pairs to energy_data_structured.jsonl only.
Source: https://www.eex.com/en/markets (and sub-pages)
"""
import json, pathlib

BASE = pathlib.Path(__file__).parent.parent
STRUCTURED_PATH = BASE / "energy_data_structured.jsonl"


def make_completion(plain: str, key_facts: list[str],
                    risk_level: str, risk_note: str,
                    confidence: str, confidence_note: str) -> str:
    facts_block = "\n".join(f"• {f}" for f in key_facts)
    return (
        f"ANSWER: {plain}\n\n"
        f"KEY FACTS:\n{facts_block}\n\n"
        f"RISK LEVEL: {risk_level}\n"
        f"→ {risk_note}\n\n"
        f"CONFIDENCE: {confidence}\n"
        f"→ {confidence_note}"
    )


entries = [

    # ── EEX — GENERAL ──────────────────────────────────────────────────────

    {
        "prompt": "What is the **European Energy Exchange (EEX)** and who owns it?",
        "completion": "The **European Energy Exchange (EEX)** is a leading commodity exchange headquartered in **Leipzig, Germany** (Augustusplatz 9). It operates regulated markets for **power, natural gas, environmental products (EU ETS), energy certificates, agricultural commodities, and hydrogen**. EEX is part of **Deutsche Börse Group** and sits at the centre of the **EEX Group**, which includes EPEX SPOT (spot power), Nodal Exchange (US power futures), ECC (clearing), PXE (Central Europe), and GET Baltic, among others. All derivatives traded on EEX are cleared through **European Commodity Clearing AG (ECC)**.",
        "key_facts": [
            "EEX is headquartered in Leipzig, Germany; part of Deutsche Börse Group",
            "Markets covered: power, natural gas, environmentals, energy certificates, agriculturals, hydrogen",
            "EEX Group entities include EEX, EPEX SPOT, Nodal Exchange, ECC, PXE, GET Baltic",
            "Clearing is handled exclusively by European Commodity Clearing AG (ECC)",
        ],
        "risk_level": "Low",
        "risk_note": "Institutional background — risk applies to counterparties not cleared through ECC.",
        "confidence": "High",
        "confidence_note": "EEX official website, markets overview page.",
    },

    {
        "prompt": "What is **European Commodity Clearing AG (ECC)** and what role does it play in EEX markets?",
        "completion": "**European Commodity Clearing AG (ECC)** is the central clearing house of the EEX Group, specialising in **energy and commodity products**. ECC acts as the **central counterparty (CCP)** for all derivatives traded on EEX, eliminating bilateral counterparty risk. It handles **physical settlement** of gas and GO futures, manages collateral and margin requirements, supports **cross-margining** across power and emissions products, and processes **same-day registry transfers** for EU ETS allowances. ECC's role is critical for both capital efficiency and systemic risk reduction in European energy markets.",
        "key_facts": [
            "ECC is the central counterparty (CCP) for all EEX derivatives",
            "Manages physical settlement of gas futures and Guarantees of Origin",
            "Offers cross-margining between power, gas, and emissions products",
            "Enables same-day registry transfer of EU Emission Allowances (EUAs)",
        ],
        "risk_level": "Low",
        "risk_note": "CCP structure significantly reduces counterparty risk for all market participants.",
        "confidence": "High",
        "confidence_note": "EEX markets overview and product pages (ECC referenced throughout).",
    },

    # ── POWER FUTURES ──────────────────────────────────────────────────────

    {
        "prompt": "What are the **standard trading parameters** for EEX European Power Futures (delivery rate, tick size, trading hours)?",
        "completion": "For all European EEX Power Futures, the standard parameters are: **delivery rate of 1 MW**, **tick size of €0.01/MWh** (or GBP 0.01/MWh for GB Power Futures), and **trading hours of 08:00–18:00 CET**. Trade registration also runs 08:00–18:00 CET where available. Japanese Power Futures use ¥0.01/kWh and trade 16:00–18:00 JST (exchange), with registration available 09:00–18:00 JST.",
        "key_facts": [
            "Standard delivery rate: 1 MW for all power futures",
            "Standard tick size: €0.01/MWh (Europe), GBP 0.01/MWh (GB), ¥0.01/kWh (Japan)",
            "Trading hours: 08:00–18:00 CET (European products)",
            "Japanese Power Futures: exchange trading 16:00–18:00 JST; registration 09:00–18:00 JST",
        ],
        "risk_level": "Low",
        "risk_note": "Standard operational parameters; risk arises from incorrect order specification.",
        "confidence": "High",
        "confidence_note": "EEX Power Futures product specifications page.",
    },

    {
        "prompt": "What **maturities** are available for **German Power Futures** on EEX, and why do they extend further than most other countries?",
        "completion": "German Power Futures on EEX offer the broadest maturity spectrum among European products: **next 11 days, next 2 weekends, current + next 4 weeks, current + next 24 months** (longest month horizon in the EEX suite), **11 quarters, and 10 calendar years**. France, the Netherlands, Italy, and Spain also reach 10 calendar years but have shorter month horizons. The deep liquidity and central position of the German market (the reference price zone for Central Europe) justifies the extended 24-month forward curve, enabling PPA pricing and long-term hedging for generators and utilities.",
        "key_facts": [
            "German Power Futures maturities: 11 days, 2 weekends, 4 weeks, 24 months, 11 quarters, 10 years",
            "24-month forward month horizon is the longest in the EEX European power suite",
            "10-year year contracts available (German, French, Dutch, Italian, Spanish, Nordic System)",
            "Settlement: cash and/or physical (by physical fulfilment for German)",
        ],
        "risk_level": "Medium",
        "risk_note": "Long-dated illiquid contracts carry mark-to-model and liquidity risk.",
        "confidence": "High",
        "confidence_note": "EEX Power Futures product specifications page.",
    },

    {
        "prompt": "How does **cascading** work for EEX Power Futures, and what is the **Future-to-Spot Service (FTS)**?",
        "completion": "**Cascading** is the methodology by which a longer-dated EEX Power Future is progressively split into shorter-dated contracts as delivery approaches. For example, a **year future** cascades into **quarterly futures** at the start of the delivery year, and a **quarterly future** cascades into **monthly futures**. This is managed automatically by EEX's clearing house, **ECC**.\n\nThe **Future-to-Spot Service (FTS)** extends this further by converting eligible futures positions into bids for the corresponding **EPEX SPOT day-ahead auction**. Eligible products include week and month futures for Germany, France, Netherlands, GB, and others. EEX initiates the service automatically (requiring a subscription via service order), or participants can handle it manually — though manual initiation is not available for Danish, Finnish, Swedish, and Norwegian futures.",
        "key_facts": [
            "Cascading: year → quarterly → monthly futures managed automatically by ECC",
            "FTS: converts futures positions into day-ahead EPEX SPOT auction bids",
            "FTS eligible: Germany, France, Netherlands, GB, Norway (NO1/NO2), Sweden, Hungary, and others",
            "EEX Initiation Service (automatic) requires subscription; manual option available for most markets",
        ],
        "risk_level": "Medium",
        "risk_note": "Incorrect FTS setup can result in unintended day-ahead positions.",
        "confidence": "High",
        "confidence_note": "EEX Power Futures product page — cascading and FTS sections.",
    },

    {
        "prompt": "What is **Location Spread Trading** on EEX Power Futures, and what are some of the key spread pairs available?",
        "completion": "**Location Spread Trading** on EEX allows participants to efficiently trade the **price differential between two delivery areas** as a single instrument, with both legs executed simultaneously at a guaranteed spread. When traded, two underlying futures positions are created — one in each delivery area. Available maturities are typically **month, quarter, and year**; some pairs also offer **week** maturities.\n\nKey spread pairs include: **Germany vs. France, Germany vs. Netherlands, Germany vs. Austria, Germany vs. Czech Republic, Germany vs. Poland, Germany vs. Nordic (System), Italy vs. France, Italy vs. Germany, Italy vs. Greece, Netherlands vs. Belgium, Spanish vs. French**, and numerous Central-Eastern European cross-border pairs (e.g., Hungarian vs. Bulgarian, Hungarian vs. Romanian, Czech vs. Slovak). Base load is available for all pairs; Peak load contracts additionally for selected liquid pairs.",
        "key_facts": [
            "Location spreads trade the price difference between two delivery zones as a single instrument",
            "Both legs execute simultaneously at a guaranteed spread — avoids leg risk",
            "30+ spread pairs available, including Germany vs. France, Italy vs. Germany, Dutch vs. Belgian",
            "Standard maturities: Month, Quarter, Year; some pairs include Week contracts",
        ],
        "risk_level": "Medium",
        "risk_note": "Spread risk is lower than outright positions but still sensitive to fundamental divergence between zones.",
        "confidence": "High",
        "confidence_note": "EEX Power Futures — Location Spread Trading section.",
    },

    {
        "prompt": "What is the **GB Power Future** on EEX, and how does it differ structurally from continental European Power Futures?",
        "completion": "The **GB Power Future** on EEX is the only European power futures contract priced in **GBP** (tick size GBP 0.01/MWh) rather than euros. Its maturity structure is also uniquely shorter: **next 11 days, 2 weekends, current + next 4 weeks, 3 months, 4 quarters, 4 seasons, and 2 calendar years** — compared to 10 years for the largest continental markets. GB is the only EEX power market with dedicated **season** maturities (reflecting market convention in British energy trading). Settlement is cash and/or physical, and Trade Registration is supported. The 2-year maximum year horizon is notably shorter than the 6–10 year range for continental European products.",
        "key_facts": [
            "GB Power Future is the only EEX European contract priced in GBP (not EUR)",
            "Maximum year maturity: 2 years (vs. 6–10 years for most continental markets)",
            "Unique 'season' maturities (4 seasons) — not offered for any other European market",
            "Month horizon: only 3 months forward (vs. 6–24 months for continental products)",
        ],
        "risk_level": "Medium",
        "risk_note": "Shorter maturity curve limits long-term hedging utility for GB market participants.",
        "confidence": "High",
        "confidence_note": "EEX Power Futures product specifications page.",
    },

    # ── NORDIC POWER ───────────────────────────────────────────────────────

    {
        "prompt": "How does EEX handle **Nordic Power market** trading, including bidding zones and the **Nordic System Price**?",
        "completion": "EEX offers Nordic Power futures across **9 bidding zones**: five Norwegian zones (NO1 Oslo, NO2 Kristiansand, NO3 Trondheim, NO4 Tromsø, NO5 Bergen) and four Swedish zones (SE1 Luleå, SE2 Sundsvall, SE3 Stockholm, SE4 Malmö), plus a **Nordic System Price Future** covering the system as a whole. The Nordic System Price Future is available in maturities from current/next 4 weeks up to **10 calendar years**. All Nordic futures are **cash settled** at €0.01/MWh tick. Underlying indices are provided by **EPEX SPOT**, described as a NEMO-harmonised reference. The **Future-to-Spot Service (FTS)** is available for week and month futures across all 12 Nordic zones.",
        "key_facts": [
            "9 Nordic bidding zones: NO1–NO5 (Norway) + SE1–SE4 (Sweden), plus Nordic System",
            "Nordic System Price Future: maturities up to 10 years; cash settled",
            "All Nordic futures are cash settled at €0.01/MWh",
            "EPEX SPOT provides NEMO-harmonised reference prices for all Nordic zones",
        ],
        "risk_level": "Medium",
        "risk_note": "Intra-Nordic zone price spreads can be significant; zonal futures are needed alongside system price.",
        "confidence": "High",
        "confidence_note": "EEX Nordic Power Markets page and Power Futures specifications.",
    },

    {
        "prompt": "What are **EEX Implied EPADs** and why are they important for Nordic power market participants?",
        "completion": "**EPAD (Electricity Price Area Differentials)** instruments capture the **price difference between a specific Nordic bidding zone and the Nordic System Price**. Traditional EPADs are traded OTC or on Nord Pool. EEX offers an **implied EPAD** structure: by simultaneously trading a **Nordic System Price Future** and a **Nordic Zonal Future** (e.g., SE3 Stockholm) on EEX, a participant replicates the EPAD payoff while benefiting from a **single margin requirement**, exchange-cleared counterparty risk elimination, and spread order book transparency. This is critical for Nordic generators (predominantly hydro and wind) who receive revenues at their local area price but hedge at the liquid system price, creating area price mismatch exposure.",
        "key_facts": [
            "EPAD = price difference between a Nordic bidding zone and the Nordic System Price",
            "EEX implied EPADs: simultaneous position in System Price Future + Zonal Future",
            "Benefits: single margin requirement, CCP clearing, spread order book transparency",
            "Key exposure for hydro and wind generators locked to local area prices",
        ],
        "risk_level": "Medium",
        "risk_note": "Area price risk is material for Nordic generators during transmission constraints and low water periods.",
        "confidence": "High",
        "confidence_note": "EEX Nordic Power Markets page — EPAD section.",
    },

    # ── POWER OPTIONS ──────────────────────────────────────────────────────

    {
        "prompt": "What **exercise style** do EEX Power Options use, and how does the **futures-style premium** work?",
        "completion": "EEX Power Options are **European-style** options — they can only be exercised at expiry, not before. However, they trade as **'futures-style'** options: the premium is **not paid upfront** but is instead broken into **daily variation margin payments** throughout the option's life. Upon exercise, the holder receives the corresponding **futures position** in the underlying power deliverable. This structure helps participants assess **daily P&L exposure** more clearly and is operationally consistent with how EEX manages futures margining through ECC. Options are available on German, French, Italian, Spanish, and Japanese (Tokyo and Kansai) power markets.",
        "key_facts": [
            "EEX Power Options are European-style (exercise at expiry only)",
            "Futures-style premium: no upfront payment — daily variation margin replaces single premium",
            "On exercise, participant receives the corresponding futures position",
            "Markets covered: Germany, France, Italy, Spain, Japan (Tokyo / Kansai)",
        ],
        "risk_level": "Medium",
        "risk_note": "Daily margin calls on futures-style options require careful liquidity management.",
        "confidence": "High",
        "confidence_note": "EEX Power Options product page.",
    },

    {
        "prompt": "What are the **maturities** available for **German Power Options** on EEX, and how are expiry dates structured for short-dated options?",
        "completion": "German Power Options on EEX offer the widest maturity range: **24 months, 6 quarters, and 4 calendar years**. Short-dated ('German Short Dated') options link to underlying year futures with **four expiry dates per delivery year**, falling at each quarter-end: end of March (expiry for the Apr delivery option), end of June (Jul), end of September (Oct), and end of December (Jan). This creates up to **12 simultaneous expiry dates** covering the next 3–4 delivery years. French, Italian, and Spanish options offer a narrower range: **5 months, 2 quarters, 2 years**. Japanese options (Tokyo and Kansai) cover **current + next 8 full months**, with average price options (settled against JEPX spot averages) launched in 2025.",
        "key_facts": [
            "German Power Options: 24 months, 6 quarters, 4 years — widest maturity range on EEX",
            "Up to 12 simultaneous expiry dates covering 3–4 delivery years",
            "Short-dated expiry structure: 4 expiries per year (end of each quarter)",
            "Japanese Average Price Options launched 2025: settled against JEPX monthly spot average",
        ],
        "risk_level": "Medium",
        "risk_note": "Multi-expiry structure requires careful options book management to avoid unintended exercise.",
        "confidence": "High",
        "confidence_note": "EEX Power Options product specifications page.",
    },

    # ── RENEWABLE ENERGY HEDGING ─────────────────────────────────────────

    {
        "prompt": "What is **'shape risk'** or **profile risk** for solar power generators, and how does EEX's **Mon-Sun Peak Future** address it?",
        "completion": "**Shape risk** (also called profile risk) arises when the generation profile of an asset does not match the profile of the standard hedging instrument. For **solar power**, generation occurs seven days a week including weekends, but standard **Peak futures** cover only **Monday–Friday peak hours**. This creates a **mismatch** for solar producers: they receive weekend revenues that are unhedged by Mon–Fri Peak contracts. EEX's **Mon-Sun Peak Future** (currently available for the Spanish market) covers power **from 08:00–20:00 CET every day of the week**, including weekends — directly matching the solar generation profile and eliminating weekend shape risk. Maturities range from daily to calendar year contracts.",
        "key_facts": [
            "Shape/profile risk: mismatch between asset generation profile and standard contract profile",
            "Solar generators produce on weekends; standard Mon–Fri Peak futures leave weekends unhedged",
            "Mon-Sun Peak Future covers 08:00–20:00 CET Monday through Sunday",
            "Currently available for the Spanish power market; maturities from day to calendar year",
        ],
        "risk_level": "High",
        "risk_note": "Unhedged weekend shape risk can cause significant revenue shortfalls for solar assets in spot-price environments.",
        "confidence": "High",
        "confidence_note": "EEX Renewable Energy Hedging page — Mon-Sun Peak Futures section.",
    },

    {
        "prompt": "What does **Cal+10** mean on EEX, and for which **markets** are 10-year power futures available?",
        "completion": "'**Cal+10**' refers to power futures extending up to **10 calendar years forward** from the current date. These long-dated contracts are essential for **Power Purchase Agreement (PPA) pricing and hedging**, providing a publicly observable, exchange-cleared price reference for assets whose revenues extend a decade into the future. A daily settlement price curve out to Cal+10 enables both asset valuation and bilateral PPA price transparency. Markets on EEX with Cal+10 maturities include: **Germany, France, Netherlands, Italy, Spain, and the Nordic System Price**. EEX Group's risk analytics subsidiary **Lacima** provides structured PPA pricing tools integrated with these curves.",
        "key_facts": [
            "Cal+10: 10 calendar-year forward maturity available on EEX",
            "Available markets: Germany, France, Netherlands, Italy, Spain, Nordic System",
            "Used for PPA pricing, long-term asset valuation, and long-dated hedging",
            "Daily settlement price curve to Cal+10 supports bilateral PPA transparency",
        ],
        "risk_level": "Medium",
        "risk_note": "Cal+10 contracts are typically illiquid — mark-to-market uses settlement curves, not traded prices.",
        "confidence": "High",
        "confidence_note": "EEX Renewable Energy Hedging and Power Futures pages.",
    },

    # ── NATURAL GAS ────────────────────────────────────────────────────────

    {
        "prompt": "What **gas hubs** are accessible through EEX's physical gas derivatives and gas spot markets?",
        "completion": "EEX provides access to **10 gas hubs** across Europe through its physical gas derivatives: **CEGH CZ VTP** (Czech Republic), **CEGH VTP** (Austrian Central European Gas Hub), **ETF** (Zeebrugge Trading Facility, Belgium), **THE** (Trading Hub Europe, Germany — successor to NCG and GASPOOL), **NBP** (National Balancing Point, UK), **PEG** (Point d'Échange de Gaz, France), **PSV** (Punto di Scambio Virtuale, Italy), **PVB** (Punto Virtual de Balance, Spain), **TTF** (Title Transfer Facility, Netherlands — the dominant European benchmark hub), and **ZTP** (Zeebrugge Trading Point, Belgium). Gas spot trading runs **24/7** covering within-day, day-ahead, and hourly contracts on THE, TTF, and PEG hubs.",
        "key_facts": [
            "10 gas hubs: CEGH CZ VTP, CEGH VTP, ETF, THE, NBP, PEG, PSV, PVB, TTF, ZTP",
            "TTF (Netherlands) is the dominant European benchmark hub",
            "THE = Trading Hub Europe (Germany), successor to NCG and GASPOOL",
            "Gas spot trading is 24/7; within-day, day-ahead, and hourly contracts available",
        ],
        "risk_level": "Medium",
        "risk_note": "Hub basis (spread between TTF and other hubs) can vary significantly during supply disruptions.",
        "confidence": "High",
        "confidence_note": "EEX Physical Gas Derivatives and Gas Spot product pages.",
    },

    {
        "prompt": "What are **Spark Spreads** on EEX and what do they allow power generators to hedge?",
        "completion": "A **Spark Spread** is the margin a gas-fired power plant earns as the **difference between the electricity price it can sell and the gas cost it must pay** to generate that electricity (adjusted for plant efficiency). On EEX, spark spreads can be traded by combining **financially settled EEX Power Futures** with **EEX Physical Gas Futures** in the same portfolio cleared through ECC. This enables gas-fired generators to **lock in the generation margin** directly, rather than hedging power and gas independently and leaving themselves exposed to basis and correlation risk between the two. Spark spreads are a fundamental hedging tool for **Combined Cycle Gas Turbine (CCGT) operators** and for **virtual power plant structures**.",
        "key_facts": [
            "Spark spread = electricity price minus gas fuel cost (adjusted for plant efficiency)",
            "EEX enables spark spread trading by combining power futures with physical gas futures",
            "Both legs are cleared through ECC — mitigating counterparty risk",
            "Key hedging tool for CCGT operators and virtual power plants",
        ],
        "risk_level": "High",
        "risk_note": "Unhedged spark spreads expose gas-fired generators to double-sided commodity price risk.",
        "confidence": "High",
        "confidence_note": "EEX Physical Gas Derivatives page — Spark Spreads section.",
    },

    {
        "prompt": "What is the **EGSI** (European Gas Spot Index) and what are **EGSI Financial Gas Futures** on EEX?",
        "completion": "The **European Gas Spot Index (EGSI)** is a reference price index for European natural gas spot markets. **EGSI Financial Gas Futures** on EEX are **cash-settled** derivatives whose final settlement is determined by the EGSI index value at expiry, rather than by physical gas delivery. This contrasts with EEX's physical gas derivatives, which require actual gas delivery at a nominated hub. EGSI Financial Gas Futures are suitable for participants who want **financial exposure and price hedging** without the operational complexity of physical delivery — for example, financial firms, industrial gas consumers hedging procurement costs, or traders managing book exposure rather than physical positions.",
        "key_facts": [
            "EGSI = European Gas Spot Index — a reference price for European gas spot markets",
            "EGSI Financial Gas Futures are cash-settled (no physical delivery)",
            "Physical gas derivatives on EEX are physically delivered at nominated hubs",
            "EGSI futures suit financial participants and hedgers without delivery infrastructure",
        ],
        "risk_level": "Medium",
        "risk_note": "Cash-settled instruments eliminate delivery risk but introduce index basis risk vs physical hubs.",
        "confidence": "High",
        "confidence_note": "EEX Natural Gas page — EGSI Financial Gas Futures section.",
    },

    # ── EU ETS ─────────────────────────────────────────────────────────────

    {
        "prompt": "What is EEX's role in the **EU Emissions Trading System (EU ETS)**, and what does it mean that it is 'the largest environmental market in the world'?",
        "completion": "EEX describes itself as **the leading exchange for the EU ETS**, operating both the **primary market** (auctions on behalf of EU member states) and the **secondary market** (spot, futures, and options). The EU ETS is the world's largest carbon market by volume and value, covering power generation, heavy industry, and aviation within the EU. Through EEX, **100% of EU member state allowances are auctioned** via the primary market using the **M7 Auction System**. EEX has successfully conducted **more than 2,000 emissions auctions**. Secondary market trading uses the **Eurex T7 matching engine** with access via TT, Trayport, Fidessa, and Fastfill platforms.",
        "key_facts": [
            "EEX: leading exchange for EU ETS — both primary (auctions) and secondary (spot/futures/options)",
            "100% of EU member state ETS allowances are auctioned through EEX",
            "More than 2,000 emissions auctions successfully conducted by EEX",
            "Secondary market uses Eurex T7 matching engine; accessible via TT, Trayport, Fidessa",
        ],
        "risk_level": "Medium",
        "risk_note": "EU ETS carbon price volatility can be high; regulatory changes (e.g., cap reductions, ETS2) affect pricing.",
        "confidence": "High",
        "confidence_note": "EEX Environmentals page, EU ETS product description.",
    },

    {
        "prompt": "What **capital efficiency advantages** does EEX offer for participants trading **EU ETS emission allowances (EUAs)** alongside other energy products?",
        "completion": "EEX offers several capital efficiency mechanisms for EU ETS participants:\n\n**1. Cross-margining:** EUAs can be margined alongside other EEX Group commodities (power, gas) through ECC, reducing the total margin required compared to holding separate positions on multiple exchanges.\n\n**2. EUAs as collateral:** ECC accepts EUAs as collateral, allowing participants to use their allowance inventory to meet margin obligations across their broader portfolio without needing to post additional cash or bonds.\n\n**3. Same-day registry transfer:** EUAs can be delivered and confirmed via same-day registry transfer, reducing settlement risk and float costs.\n\n**4. Efficient and free delivery:** ECC processes EUA deliveries directly via registry transfer, with EEX describing delivery as 'efficient and free'.",
        "key_facts": [
            "Cross-margining: EUAs margined alongside power and gas products at ECC",
            "EUAs accepted as collateral — reduces cash margin requirements",
            "Same-day registry transfer available for EU ETS spot and futures settlement",
            "ECC delivery of EUAs described as free by EEX",
        ],
        "risk_level": "Medium",
        "risk_note": "Capital efficiency benefits disappear if EUA prices fall sharply, reducing their collateral value.",
        "confidence": "High",
        "confidence_note": "EEX Environmentals / EU ETS product page.",
    },

    {
        "prompt": "What are **EU ETS2** and the **EU ETS for the Maritime Sector**, and how do they differ from the original EU ETS?",
        "completion": "The **EU ETS2** extends the EU carbon pricing mechanism to **buildings and road transport** — sectors not covered by the original EU ETS (EU ETS1), which focuses on power generation, heavy industry, and aviation. ETS2 introduces a separate cap-and-trade system with its own **fuel suppliers** as regulated entities, rather than end consumers, beginning ~2027. The **EU ETS for the Maritime Sector** (EU ETS Maritime) integrates **shipping emissions** into the existing EU ETS1 framework, phased in from 2024, requiring shipowners to surrender EUAs for CO₂ emissions from voyages within the EU, EU inbound/outbound, and between EU ports. EEX lists both EU ETS2 and Maritime ETS as distinct product categories, reflecting their separate regulatory structures and developing market needs.",
        "key_facts": [
            "EU ETS2 covers buildings and road transport — not included in original EU ETS1",
            "ETS2 regulates fuel suppliers (not end consumers); targeted start ~2027",
            "EU ETS Maritime: ships must surrender EUAs for CO₂ on EU routes — phased from 2024",
            "Both are listed as separate EEX product categories alongside EU ETS1",
        ],
        "risk_level": "High",
        "risk_note": "ETS2 and Maritime ETS are new and developing markets; regulatory uncertainty and liquidity are risks.",
        "confidence": "High",
        "confidence_note": "EEX Environmentals page; EU regulatory framework for ETS2 and Maritime ETS.",
    },

    # ── GUARANTEES OF ORIGIN ───────────────────────────────────────────────

    {
        "prompt": "What is a **Guarantee of Origin (GO)** in the EU energy market, and what does each GO certify?",
        "completion": "A **Guarantee of Origin (GO)** is a standardised EU electronic certificate that certifies the renewable origin of **1 MWh of electricity** generated within one calendar month from a specific installation. GOs are issued under the EU Renewable Energy Directive by AIB (Association of Issuing Bodies) member state registries. They allow electricity consumers, retailers, and companies with sustainability commitments to prove and disclose the renewable source of their power consumption. GOs play a central role in voluntary frameworks including **RE100** (100% renewable electricity commitment), **SBTi** (Science Based Targets initiative), and **CDP** (Carbon Disclosure Project) corporate reporting. EEX operates as the **national registry operator** for French GOs (under appointment from the French Ministry of Ecology) and runs EU GO Futures across hydro, wind, solar, and general renewable categories.",
        "key_facts": [
            "1 GO = 1 MWh of electricity from a certified renewable source in a specific month",
            "Issued via AIB-member national registries under the EU Renewable Energy Directive",
            "Used in RE100, SBTi, and CDP corporate sustainability frameworks",
            "EEX is the national GO registry operator for France (appointed by French Ministry of Ecology)",
        ],
        "risk_level": "Low",
        "risk_note": "GO prices fluctuate based on renewable energy supply and corporate sustainability demand.",
        "confidence": "High",
        "confidence_note": "EEX Energy Certificates / GO Futures page.",
    },

    {
        "prompt": "What are the **contract specifications** for EEX GO Futures, including lot size, maturities, and settlement?",
        "completion": "EEX GO Futures have the following contract specifications:\n\n- **Contract volume (lot size):** 1 lot = **1,000 Guarantees of Origin** (= 1,000 MWh of certified renewable electricity)\n- **Minimum lot size:** 1 contract\n- **Pricing:** EUR per GO, to the second decimal place (€0.01 minimum price fluctuation)\n- **Tradable maturities:** **Next four January maturities** (annual structure)\n- **Last trading day:** The **antepenultimate Exchange Day in January** of the respective maturity year (i.e., the third-to-last trading day)\n- **Production period eligible:** Months March–December of the **year preceding the maturity**\n- **Settlement:** **Physically delivered** — ECC instructs EEX (as registry operator) to transfer GOs to the buyer's account on the last ECC Business Day of the maturity month\n- **Clearing:** ECC",
        "key_facts": [
            "1 lot = 1,000 GOs (= 1,000 MWh of certified renewable electricity)",
            "4 annual maturities: next four January delivery years",
            "Last trading day: antepenultimate Exchange Day in January",
            "Settlement: physically delivered via ECC and EEX registry on last ECC Business Day",
        ],
        "risk_level": "Low",
        "risk_note": "Physically settled — participants must have registry accounts for delivery.",
        "confidence": "High",
        "confidence_note": "EEX GO Futures product specifications page.",
    },

    {
        "prompt": "What are **WHS GO Futures** on EEX, and how do they differ from standard renewable GO Futures?",
        "completion": "**WHS (Wind, Hydro, Solar) GO Futures** on EEX are a category of GO futures with specific underlying requirements:\n\n- **Standard WHS GO Future:** Includes GOs from wind, hydro, and solar installations with **any earmark** (support level 0–4), meaning it covers both state-subsidised and unsubsidised renewable generation.\n- **WHS Unsupported GO Future:** Covers only GOs with **earmark 0** — installations that receive **no government support**, fully unsubsidised and market-driven.\n\nThis distinction matters because **corporate buyers increasingly prefer 'additionality'** — GOs from new, unsubsidised renewable capacity that demonstrably adds new green generation to the grid, rather than simply redistributing certificates from mature, heavily subsidised hydro capacity. The 'unsupported-only' contract addresses this demand by enabling price discovery specifically for market-driven GOs.",
        "key_facts": [
            "WHS GO Future: wind/hydro/solar GOs with any earmark (supported or not)",
            "WHS Unsupported GO Future: only installations with earmark 0 (no government support)",
            "Additionality principle: corporate buyers prefer unsubsidised, new-build renewable GOs",
            "Both WHS products are physically settled via ECC registry transfer",
        ],
        "risk_level": "Medium",
        "risk_note": "WHS Unsupported GOs typically trade at a premium due to additionality demand; liquidity may be lower.",
        "confidence": "High",
        "confidence_note": "EEX GO Futures product specifications page — WHS section.",
    },

    # ── HYDROGEN ───────────────────────────────────────────────────────────

    {
        "prompt": "What is the **HYDRIX index**, when was it launched, and what makes it significant for European hydrogen markets?",
        "completion": "**HYDRIX** is the first **market-based price index for green hydrogen** in Europe, launched by EEX in **2023**. It is published **every Wednesday at 16:00 CE(S)T** and is calculated as the **average of supply and demand price indications** submitted by established hydrogen industry participants. Historical and current data are available via the EEX Market Data Hub under Hydrogen > Indices > DE > Hydrix.\n\nHYDRIX gained major significance in **2025**, when Germany's **Federal Ministry for Economic Affairs and Climate Action (BMWK)** selected it as the **reference price for Germany's Carbon Contracts for Difference (CCfD) programme** — an important government instrument to subsidise green hydrogen production and reduce its cost gap vs. fossil hydrogen. This selection gives HYDRIX official market infrastructure status in the EU hydrogen economy.",
        "key_facts": [
            "HYDRIX launched 2023 — first market-based green hydrogen price index in Europe",
            "Published weekly every Wednesday at 16:00 CE(S)T via EEX Market Data Hub",
            "Calculated as the average of industry supply and demand price indications",
            "In 2025, German BMWK selected HYDRIX as CCfD reference price — officially endorsed",
        ],
        "risk_level": "Medium",
        "risk_note": "Immature market; HYDRIX reflects indicative prices from limited participants — liquidity risk.",
        "confidence": "High",
        "confidence_note": "EEX Hydrogen market page.",
    },

    {
        "prompt": "What is **H2EX** and how does EEX plan its **hydrogen trading platform** to evolve as the market matures?",
        "completion": "**H2EX** is EEX's dedicated hydrogen trading platform — described as a **browser-based trading system** designed to match the needs of the early and developing hydrogen market. EEX envisions a two-stage evolution of hydrogen trading:\n\n**Stage 1 (current/near-term):** **Spot and balancing products** will use EEX's existing **natural gas spot trading infrastructure**, leveraging its 24/7 trading capability and established TSO nomination workflows — adapted for hydrogen once physical infrastructure exists.\n\n**Stage 2 (as clusters develop):** **Forward products** will be launched on H2EX alongside the first functioning **hydrogen clusters** (geographic concentrations of production and demand), pending adequate regulatory and infrastructure backing. Products will cover both hydrogen and **hydrogen derivatives (Power-to-X assets)** such as ammonia and methanol.\n\nEEX also supports **H2Global / Hintco** as the trading subsidiary for the international H2Global mechanism, further embedding EEX in nascent hydrogen market infrastructure.",
        "key_facts": [
            "H2EX: browser-based platform for hydrogen forward trading",
            "Spot/balancing products to use existing EEX gas spot infrastructure",
            "Forward products to launch alongside first hydrogen clusters with regulatory backing",
            "Products will cover hydrogen and Power-to-X derivatives (ammonia, methanol)",
        ],
        "risk_level": "High",
        "risk_note": "Hydrogen market infrastructure and regulation are immature — timeline for products is uncertain.",
        "confidence": "High",
        "confidence_note": "EEX Hydrogen market page.",
    },

    {
        "prompt": "What is the **EEX Hydrogen Working Group**, and who participates?",
        "completion": "The **EEX Hydrogen Working Group** was established in **2020** and comprises over **100 companies** across **10+ countries**, covering the **full hydrogen value chain** globally — including producers, traders, industrial off-takers, infrastructure providers, and financial institutions. The working group was instrumental in developing both the **HYDRIX index** and the technical design of the **H2EX hydrogen trading platform**. It serves as a collaborative forum where EEX engages with industry stakeholders to shape hydrogen market standards, trading rules, and certificate frameworks before fully liquid trading platforms exist. EEX also issues **Letters of Intent (LOIs) and Letters of Support (LOSs)** to hydrogen cluster projects to formalise cooperation at an early development stage.",
        "key_facts": [
            "EEX Hydrogen Working Group: established 2020; 100+ companies from 10+ countries",
            "Full value chain: producers, traders, industrial off-takers, infrastructure, finance",
            "Working group led to HYDRIX index launch and H2EX platform design",
            "EEX issues LOIs/LOSs to hydrogen cluster projects to formalise early cooperation",
        ],
        "risk_level": "Low",
        "risk_note": "Working group is pre-competitive; binding commitments require separate contract agreements.",
        "confidence": "High",
        "confidence_note": "EEX Hydrogen market page.",
    },

    # ── AGRICULTURAL PRODUCTS ──────────────────────────────────────────────

    {
        "prompt": "What **agricultural commodity derivatives** does EEX list, and what is their settlement structure?",
        "completion": "EEX lists **two categories** of agricultural commodity derivatives: **Dairy Products** and **Potatoes**. All EEX agricultural contracts are **financially settled (cash-settled)** — there is no physical delivery of agricultural commodities. Clearing is provided by **European Commodity Clearing AG (ECC)**, consistent with EEX's broader derivatives infrastructure. EEX positions these products as offering participants access to commodity price risk management and price discovery for processed agricultural commodities in a regulated, CCP-cleared environment. The agricultural business is managed by EEX's Head of Agricultural Commodities, Sascha Siegel (Leipzig).",
        "key_facts": [
            "EEX agricultural products: Dairy Products futures and Potato futures",
            "All agricultural contracts are financially (cash) settled — no physical delivery",
            "Cleared by ECC (same CCP as EEX energy derivatives)",
        ],
        "risk_level": "Medium",
        "risk_note": "Agricultural commodity prices exhibit seasonality and weather-driven volatility.",
        "confidence": "High",
        "confidence_note": "EEX Agriculturals overview page.",
    },

    # ── ADVISORY ────────────────────────────────────────────────────────────

    {
        "prompt": "I am a **solar power developer** in Spain. My plant generates on weekends but I can only find Mon–Fri Peak futures. **How should I manage the weekend shape risk on EEX?**",
        "completion": "Weekend shape risk is a real and material revenue risk for solar assets operating in markets where spot prices on weekends may differ significantly from Mon–Fri Peak prices. Here is the recommended approach:\n\n**1. Use EEX Mon-Sun Peak Futures for the Spanish market.**\nEEX offers Mon-Sun Peak Futures specifically for Spain, covering 08:00–20:00 CET seven days a week. These directly match your solar generation profile and eliminate the weekend mismatch that standard Peak futures leave unhedged.\n\n**2. Stack your hedge across maturities.**\nMon-Sun Peak contracts run from daily to calendar year — use short-dated contracts for near-term production and year contracts for approved pipeline projects requiring long-term revenue certainty.\n\n**3. Consider Cal+10 year futures for PPA pricing.**\nIf you are structuring a Power Purchase Agreement with a corporate buyer, EEX Spanish year futures extend to Cal+10 (10 years). The daily settlement curve provides a transparent reference for PPA pricing and lender due diligence.\n\n**4. Pair with GO Futures if the buyer has RE100 or SBTi commitments.**\nMany corporate PPAs require bundled Guarantees of Origin. EEX Solar GO Futures (EGOS) can be overlaid to provide a complete renewable certificate hedge alongside the physical price hedge.",
        "key_facts": [
            "Mon-Sun Peak Future: 08:00–20:00 CET daily (including weekends) — specific to Spanish market",
            "Standard Mon–Fri Peak futures leave weekend solar revenues unhedged",
            "Spain futures with Cal+10 maturities support long-term PPA pricing",
            "EEX Solar GO Futures (EGOS) can be bundled with power futures for corporate RE buyers",
        ],
        "risk_level": "High",
        "risk_note": "Unhedged weekend profile risk can cause revenue shortfalls; proper product selection is critical.",
        "confidence": "High",
        "confidence_note": "EEX Renewable Energy Hedging; GO Futures; Power Futures specifications.",
    },

    {
        "prompt": "Our company has a **RE100 commitment** to source 100% renewable electricity. Should we use **EEX GO Futures** to meet it, and what should we consider?",
        "completion": "RE100 requires companies to cover **100% of electricity consumption with renewable energy**, typically documented via **Guarantees of Origin (GOs)**. EEX GO Futures are a strong tool for managing this obligation, but require careful structuring:\n\n**When EEX GO Futures work well:**\n- You need **forward price hedging** for GOs — locking in a price 1–4 years ahead rather than buying GOs at uncertain spot prices each year.\n- You have the capacity to **handle physical delivery** — GO Futures are physically settled via registry transfer (ECC instructs EEX registry to transfer 1,000 GOs per lot to your account).\n- You want **technology-specific GOs** (hydro, wind, or solar) to align with corporate narrative or consumer preferences.\n\n**Key considerations:**\n1. **Additionality:** If your RE100 commitments emphasise new renewable capacity, choose **EEX WHS Unsupported GO Futures** (earmark 0 — unsubsidised generation) over general GO Futures. Increasingly, RE100 signatories and SBTi guidance favour additionality.\n2. **Volume matching:** 1 lot = 1,000 GOs = 1,000 MWh. Ensure your contracted volume is a round multiple or hedge the remainder in the spot GO market.\n3. **January maturity structure:** GO Futures only trade 4 annual maturities (next 4 January expiries). Plan your procurement timeline accordingly.\n4. **Registry access:** You must have an account in the relevant AIB-member GO registry to receive physical delivery.",
        "key_facts": [
            "RE100 requires 100% renewable electricity coverage — GOs are the standard documentation instrument",
            "EEX GO Futures available for hydro (EGOH), wind (EGOW), solar (EGOS), renewable (EGOE)",
            "1 lot = 1,000 GOs; physically settled; 4 annual January maturities",
            "WHS Unsupported (EGOU) preferred where additionality matters (earmark 0 only)",
        ],
        "risk_level": "Medium",
        "risk_note": "GO prices and additionality standards are evolving; annual-only maturities limit intra-year hedging flexibility.",
        "confidence": "High",
        "confidence_note": "EEX GO Futures product specs; RE100, SBTi, and CDP framework requirements.",
    },

    {
        "prompt": "As an **energy trader entering the EU ETS market** for the first time, what do I need to know about **trading EUAs on EEX**?",
        "completion": "The EU ETS is the world's largest carbon market and EEX is its leading exchange. Here are the key things a new trader must understand:\n\n**1. Product structure:**\nEEX offers **EU ETS Spot** (physical EUAs with same-day registry transfer), **EUA Futures** (exchange-traded forwards), and **EUA Options** (European-style), all cleared by ECC. EU ETS2 and Maritime ETS are emerging as separate instruments.\n\n**2. Primary vs. secondary market:**\nThe primary market is EU member state **auctions** run via the M7 Auction System — different access requirements apply than for secondary market trading. Secondary market trading uses the **Eurex T7** matching engine (screen: EEX TT).\n\n**3. Capital efficiency:**\nEUAs can serve as **collateral** at ECC, reducing your total margin requirement if you also trade power or gas. Cross-margining across emissions, power, and gas can significantly reduce posted collateral.\n\n**4. Regulatory obligations:**\nYou will be subject to **REMIT** (transaction reporting and market abuse surveillance) and **MiFIR** reporting obligations. EEX provides REMIT transaction reporting services. Ensure your compliance framework is in place before trading.\n\n**5. Registry account:**\nFor spot EUA settlement, you need an **EU registry account** in the Union Registry. Same-day transfer is available, but account setup lead times can be several weeks.\n\n**6. Price drivers to monitor:**\nEUA prices are driven by: energy fuel switching thresholds (gas vs. coal), renewable capacity additions, free allocation adjustments, economic activity (industrial output), and policy signals such as the EU ETS cap trajectory and Market Stability Reserve (MSR) mechanism.",
        "key_facts": [
            "EEX offers EUA spot, futures, and options; all cleared by ECC",
            "Secondary market trading on Eurex T7 engine; primary market via M7 Auction System",
            "EUAs accepted as collateral at ECC — cross-margining with power/gas available",
            "REMIT and MiFIR compliance obligations apply; EU registry account required for spot",
        ],
        "risk_level": "High",
        "risk_note": "EU ETS prices are highly volatile and sensitive to policy changes (MSR, cap tightening, ETS2 expansion).",
        "confidence": "High",
        "confidence_note": "EEX Environmentals page; EU ETS regulatory framework.",
    },

    {
        "prompt": "A **Nordic wind farm operator** wants to hedge their revenues on EEX. How should they think about **system price risk vs. area price risk**?",
        "completion": "A Nordic wind farm is exposed to two distinct layers of price risk that require separate hedging instruments:\n\n**1. System Price Risk:**\nThe Nordic System Price is the reference for the most liquid Nordic power futures market. Hedging against system price using **EEX Nordic System Price Futures** provides baseline revenue certainty but does not account for the difference between the system price and what the generator actually receives (their zone price).\n\n**2. Area Price Risk (Zonal Basis Risk):**\nA wind farm in **SE3 (Stockholm)** receives the SE3 area price, not the System Price. During transmission constraints (e.g., limited cross-border interconnection capacity), SE3 can trade significantly above or below the System Price. This gap is the **EPAD (Electricity Price Area Differential)**. On EEX, you can replicate this by simultaneously holding:\n- A **Nordic System Price Future** (long, for baseline hedge)\n- An **SE3 Zonal Future** (to capture/neutralise the area price difference)\n\n**Recommended approach:**\n1. Hedge your baseline generation revenue with System Price Futures.\n2. Add a zonal position (Implied EPAD) in your specific bidding zone to complete the hedge.\n3. Use the **Future-to-Spot Service (FTS)** if you want your futures positions automatically converted to EPEX SPOT day-ahead bids for physical volume settlement.\n4. For long-term PPA pricing, combine with Cal+10 year futures for your zone.",
        "key_facts": [
            "Nordic wind generators receive their bidding zone area price, not the system price",
            "EPAD = price difference between bidding zone and Nordic System Price",
            "EEX Implied EPAD: long system price future + long/short zonal future simultaneously",
            "FTS (Future-to-Spot Service) available for all 12 Nordic bidding zones — week and month futures",
        ],
        "risk_level": "High",
        "risk_note": "Incomplete hedging (system price only, no zonal) leaves significant area price risk unhedged.",
        "confidence": "High",
        "confidence_note": "EEX Nordic Power Markets page; Power Futures specifications.",
    },

    {
        "prompt": "A **gas-fired power plant (CCGT)** wants to lock in its **generation margin**. Should it hedge power and gas separately, or use a **Spark Spread** on EEX?",
        "completion": "For a CCGT operator, the **Spark Spread** approach is strongly preferable to separate power and gas hedges. Here's why:\n\n**Separate hedge problem:**\nIf you buy a gas hedge on TTF and sell a power future separately, you are still exposed to **correlation risk** — the risk that gas prices and power prices move differently than expected, compressing your margin even with both positions hedged. If gas prices spike but power prices don't follow immediately (lag risk), your plant runs at a loss despite both hedges being in place.\n\n**Spark Spread solution:**\nEEX allows you to combine a **financially settled Power Future** with an **EEX Physical Gas Future** (clearing via ECC), directly trading the differential. This **locks in the generation margin** (spark spread) as a single position, eliminating the correlation and basis risk between separate hedges.\n\n**Practical steps:**\n1. Calculate your plant's **spark spread efficiency** (how many MWh of gas at your efficiency factor are needed to generate 1 MWh of electricity).\n2. Structure the combined EEX position in the correct ratio (MW power vs. MWh gas).\n3. Use time spreads and geographic spreads if your gas sourcing and power delivery are from different hubs/zones.\n4. Clear everything through ECC for cross-margining benefits and counterparty risk elimination.",
        "key_facts": [
            "Spark spread = power price minus gas cost (adjusted for plant efficiency)",
            "Separate hedges leave correlation risk; spark spread eliminates it as a single instrument",
            "EEX enables spark spread via combined power futures + physical gas futures, cleared by ECC",
            "Cross-margining through ECC reduces total margin vs. two separate products on different venues",
        ],
        "risk_level": "High",
        "risk_note": "Incorrectly sized spark spread hedge (wrong efficiency factor ratio) can create residual net long/short exposure.",
        "confidence": "High",
        "confidence_note": "EEX Physical Gas Derivatives page — Spark Spreads; ECC cross-margining documentation.",
    },

    {
        "prompt": "Our company wants to **invest in a green hydrogen project** in Germany and needs a price reference for offtake contracts. **What does EEX's HYDRIX offer, and is it sufficient?**",
        "completion": "HYDRIX is currently the **only market-based green hydrogen price index in Europe** and has already achieved a significant milestone: **Germany's BMWK selected it as the reference price for its Carbon Contracts for Difference (CCfD) programme** in 2025. This gives HYDRIX regulatory endorsement for German hydrogen subsidy auctions.\n\n**What HYDRIX offers your project:**\n- A weekly published, transparent average of supply/demand indications from hydrogen industry participants\n- Available as historical and current data from the EEX Market Data Hub\n- A benchmark that is increasingly standardised in German government offtake and subsidy frameworks\n\n**Limitations to consider:**\n1. **Liquidity:** HYDRIX is based on **indications** (not binding executed trades), meaning it reflects sentiment rather than deep two-way market liquidity. It is not yet a fully traded index like TTF for gas.\n2. **Forward curve:** Unlike gas (TTF futures) or power (EEX futures), there is no equivalent liquid hydrogen futures curve for multi-year price discovery. This limits your ability to lock in prices for a long-term offtake agreement.\n3. **Coverage:** HYDRIX currently focuses on German green hydrogen. Cross-border hydrogen pricing is still nascent.\n\n**Recommendation:** Use HYDRIX as your **floor price reference and government CCfD anchor**, but supplement it with bilateral price negotiations for private offtake — and engage with EEX's Hydrogen Working Group to track the development of H2EX forward products, which will eventually provide exchange-traded forward curve liquidity.",
        "key_facts": [
            "HYDRIX: weekly index, first market-based green hydrogen price reference in Europe",
            "BMWK selected HYDRIX as Germany's CCfD reference price in 2025",
            "Based on industry price indications, not yet executed bilateral trades",
            "No traded forward curve yet for hydrogen — H2EX will address this as market matures",
        ],
        "risk_level": "High",
        "risk_note": "HYDRIX reflects a very young market; index is indicative only — should not be sole basis for long-term contracts.",
        "confidence": "High",
        "confidence_note": "EEX Hydrogen page; BMWK CCfD announcement 2025.",
    },

    {
        "prompt": "Our trading desk wants to use **EEX Power Options** to hedge tail risks in our German power portfolio. **What should we know about their structure before we start?**",
        "completion": "EEX German Power Options are well-suited for tail risk hedging but have specific structural features that must be understood:\n\n**1. European-style exercise:**\nOptions can only be exercised **at expiry** — not before. This limits their use to hedging specific delivery periods rather than providing American-style flexibility to exit early. Plan your hedge so expiry aligns with the delivery period you are protecting.\n\n**2. Futures-style premium payment:**\nThe option premium is **not paid upfront**. Instead, it is collected via **daily variation margin** throughout the option's life. This is positive for cash flow but means your margin account fluctuates daily with option value — you must ensure your clearing account has sufficient liquidity buffer.\n\n**3. Exercise payoff is a futures position:**\nWhen you exercise (or are auto-exercised at expiry), you receive a **futures position** in the underlying German power future — not a cash payment. You will then need to manage, roll, or close this futures position.\n\n**4. Short-dated options structure:**\nGerman short-dated options offer **up to 12 simultaneous expiry dates** (4 per year) for the next 3–4 delivery years, providing good granularity for structuring hedges around specific quarterly delivery periods.\n\n**5. Maturity range:**\nGerman options extend to **24 months, 6 quarters, and 4 years** — the widest in EEX's power options suite.\n\n**Recommendation:** Use bought puts on year futures to define a **floor for power revenues** (for generators) or bought calls to cap procurement costs (for consumers). Combine with futures positions to create structured collars if your budget for option premium margin is constrained.",
        "key_facts": [
            "EEX Power Options are European-style — exercise at expiry only",
            "Futures-style premium: daily variation margin, not upfront payment",
            "Exercise delivers a futures position, not cash — subsequent management required",
            "German options maturity range: 24 months + 6 quarters + 4 years (widest on EEX)",
        ],
        "risk_level": "High",
        "risk_note": "Options require daily liquidity management (variation margin); exercise delivers futures — book must absorb positions.",
        "confidence": "High",
        "confidence_note": "EEX Power Options product specifications page.",
    },

]


def make_completion(plain: str, key_facts: list,
                    risk_level: str, risk_note: str,
                    confidence: str, confidence_note: str) -> str:
    facts_block = "\n".join(f"• {f}" for f in key_facts)
    return (
        f"ANSWER: {plain}\n\n"
        f"KEY FACTS:\n{facts_block}\n\n"
        f"RISK LEVEL: {risk_level}\n"
        f"→ {risk_note}\n\n"
        f"CONFIDENCE: {confidence}\n"
        f"→ {confidence_note}"
    )


lines = [
    json.dumps(
        {
            "prompt": e["prompt"],
            "completion": make_completion(
                e["completion"],
                e["key_facts"],
                e["risk_level"],
                e["risk_note"],
                e["confidence"],
                e["confidence_note"],
            ),
        },
        ensure_ascii=False,
    )
    for e in entries
]

with open(STRUCTURED_PATH, "a", encoding="utf-8") as f:
    f.write("\n" + "\n".join(lines) + "\n")

count_added = len(lines)
total = sum(1 for line in open(STRUCTURED_PATH, encoding="utf-8") if line.strip())
print(f"[+] Appended {count_added} entries → energy_data_structured.jsonl")
print(f"    Total lines now: {total}")
