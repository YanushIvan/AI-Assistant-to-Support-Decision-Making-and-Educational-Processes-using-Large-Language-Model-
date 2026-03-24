"""
Appends new IEA / Eurostat / ACER energy QA pairs to both dataset files.
Sources:
  - https://www.iea.org/topics/the-middle-east-and-global-energy-markets
  - https://ec.europa.eu/eurostat/web/energy
  - https://www.acer.europa.eu/
"""
import json, pathlib

BASE = pathlib.Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# New QA pairs  (prompt, completion_plain, key_facts, risk_level, confidence)
# ---------------------------------------------------------------------------
entries = [
    # ── MEDIUM ──────────────────────────────────────────────────────────────
    {
        "prompt": "What **share of world seaborne oil trade** transited the Strait of Hormuz in 2025, and what was the volume in absolute terms?",
        "completion": "In 2025, approximately **25% of world seaborne oil trade** transited the Strait of Hormuz, corresponding to a pre-conflict flow of around **20 million barrels per day (mb/d)** of crude oil and refined products. This makes the Strait the world's most critical maritime chokepoint for energy trade.",
        "key_facts": [
            "~25% of world seaborne oil trade transited the Strait of Hormuz in 2025",
            "Pre-conflict flow was ~20 mb/d of crude oil and refined products",
            "The Strait connects the Persian Gulf with the Gulf of Oman and Arabian Sea",
        ],
        "risk_level": "High",
        "risk_note": "High geopolitical exposure; any disruption has immediate global price impact.",
        "confidence": "High",
        "confidence_note": "Based on IEA 2026 official market analysis.",
    },
    {
        "prompt": "What is the total **alternative bypass pipeline capacity** if the Strait of Hormuz is fully closed, and which countries provide it?",
        "completion": "If the Strait of Hormuz is fully closed, the only alternative bypass capacity available is **3.5–5.5 mb/d**, provided by pipelines in **Saudi Arabia** and the **UAE**. This is drastically insufficient to replace the pre-war flow of ~20 mb/d, meaning a sustained closure causes a severe and largely irreplaceable short-term supply shock to global oil markets.",
        "key_facts": [
            "Alternative bypass capacity: only 3.5–5.5 mb/d (Saudi Arabia + UAE pipelines)",
            "Pre-conflict Strait flow: ~20 mb/d — bypasses cover at most ~27% of that volume",
            "No other credible alternative transit routes exist for Gulf seaborne oil",
        ],
        "risk_level": "High",
        "risk_note": "Insufficient bypass capacity makes Strait closure an extreme systemic risk.",
        "confidence": "High",
        "confidence_note": "Sourced from IEA 2026 Middle East energy market analysis.",
    },
    {
        "prompt": "What **share of global LNG trade** passed through the Strait of Hormuz in 2025, and which exporting countries were most exposed?",
        "completion": "Over **110 billion cubic metres (bcm)** of LNG transited the Strait of Hormuz in 2025, representing nearly **one-fifth of global LNG trade**. **Qatar** routed approximately **93%** of its LNG exports through the Strait, and the **UAE** routed approximately **96%**. Around **80–90%** of these volumes were destined for **Asian** markets (Japan, South Korea, China, India), with only just over **10%** flowing to Europe.",
        "key_facts": [
            ">110 bcm of LNG transited the Strait in 2025",
            "Qatar: ~93% of LNG exports via the Strait; UAE: ~96%",
            "Combined = ~20% of global LNG trade",
            "~80–90% of Strait LNG destined for Asia; ~10% to Europe",
        ],
        "risk_level": "High",
        "risk_note": "Qatar and UAE are almost entirely dependent on this single transit route.",
        "confidence": "High",
        "confidence_note": "IEA 2026 official data on Strait LNG flows.",
    },
    {
        "prompt": "What **emergency stock release** did the IEA authorise in March 2026, and how does it compare to all previous IEA collective actions?",
        "completion": "On **11 March 2026**, IEA member countries unanimously agreed to release **400 million barrels** of strategic petroleum reserves — the **largest-ever collective IEA stock release**. This was the **sixth** such action in the IEA's history; previous releases occurred in **1991** (Gulf War), **2005** (Hurricane Katrina), **2011** (Libyan civil war), and **twice in 2022** (Russia-Ukraine war).",
        "key_facts": [
            "400 million barrels released — largest in IEA history",
            "Sixth collective action since IEA founding in 1974",
            "Previous actions: 1991, 2005, 2011, and twice in 2022",
            "All 32 IEA member countries voted unanimously",
        ],
        "risk_level": "High",
        "risk_note": "Scale of release reflects the severity of the 2026 supply disruption.",
        "confidence": "High",
        "confidence_note": "IEA official announcement, March 2026.",
    },
    {
        "prompt": "How many **days of net oil imports** must each IEA member country hold as **emergency stocks**, and how large were global inventories at the time of the 2026 crisis?",
        "completion": "Each of the IEA's **32 member countries** must hold emergency stocks equal to at least **90 days** of net oil imports. At the time of the 2026 Middle East crisis, global observed inventories exceeded **8.2 billion barrels** — the highest since February 2021. Government emergency stocks in advanced economies totalled **1.25 billion barrels**, with a further **600 million barrels** held by industry under government obligation.",
        "key_facts": [
            "IEA 90-day net import coverage requirement for all 32 members",
            "Global observed inventories: >8.2 billion barrels (highest since Feb 2021)",
            "Government emergency stocks: 1.25 billion barrels",
            "Industry stocks under obligation: 600 million barrels",
        ],
        "risk_level": "Medium",
        "risk_note": "Buffer is large but could be depleted by a prolonged Strait closure.",
        "confidence": "High",
        "confidence_note": "IEA 2026 Middle East market analysis, official figures.",
    },
    {
        "prompt": "What happened to **global LNG supply** as a result of the 2026 Middle East conflict, and what was the role of the **Ras Laffan** facility?",
        "completion": "The 2026 Middle East conflict reduced **global LNG supply by roughly 20%** due to the Strait of Hormuz closure. A key driver was the shutdown of **Ras Laffan** — Qatar's massive LNG export complex and the **world's largest LNG facility** — following an attack on **2 March 2026**. Since Qatar routes ~93% of its LNG through the Strait, the facility's closure had an immediate global supply impact. Dutch TTF (European gas benchmark) rose **over 55%** since hostilities began.",
        "key_facts": [
            "Global LNG supply reduced by ~20% due to Strait closure",
            "Ras Laffan (Qatar) — world's largest LNG facility — shut after 2 March 2026 attack",
            "Qatar routes ~93% of LNG via the Strait",
            "Dutch TTF rose >55% since conflict began",
        ],
        "risk_level": "High",
        "risk_note": "Ras Laffan shutdown is a systemic shock to global LNG supply.",
        "confidence": "High",
        "confidence_note": "IEA 2026 Middle East energy market report.",
    },
    {
        "prompt": "What role does **ACER** play in the European energy market, and which sectors fall under its regulatory oversight?",
        "completion": "**ACER (Agency for the Cooperation of Energy Regulators)**, headquartered in Ljubljana, Slovenia, oversees integration of European electricity and gas markets and coordinates national regulators. Its key responsibilities include: monitoring **cross-border electricity trading** and congestion; overseeing compliance with the **70% cross-zonal capacity target**; administering the **REMIT** market transparency framework; assessing LNG prices; and monitoring decarbonisation of gas networks and the emerging **hydrogen market**. A 2026 flagship report concluded that expanding EU energy market integration is key for global competitiveness and decarbonisation.",
        "key_facts": [
            "ACER = Agency for the Cooperation of Energy Regulators, based in Ljubljana, Slovenia",
            "Covers electricity markets (capacity allocation, bidding zones, balancing, REMIT) and gas/hydrogen",
            "Monitors the 70% cross-zonal electricity capacity (CZC) target",
            "2026 report: market integration is key for EU competitiveness and decarbonisation",
        ],
        "risk_level": "Low",
        "risk_note": "Regulatory information; risk applies to non-compliant market participants.",
        "confidence": "High",
        "confidence_note": "ACER official website and 2026 market integration report.",
    },
    {
        "prompt": "What is the EU's **70% cross-zonal electricity capacity (CZC) target**, and why does it matter for market integration?",
        "completion": "The EU requires that at least **70% of electricity transmission capacity** on interconnectors between bidding zones be made available for **cross-border trade**. This target, actively monitored by ACER, prevents national grid operators from ring-fencing capacity for domestic use. Meeting the 70% CZC target enables optimal dispatch of cheap renewable electricity across borders, lowers consumer prices, and supports EU decarbonisation goals by allowing wind in the North Sea or solar in Iberia to be consumed where needed most.",
        "key_facts": [
            "70% of interconnector capacity must be available for cross-border electricity trade under EU rules",
            "ACER monitors compliance through cross-zonal trade and congestion dashboards",
            "Non-compliance fragments the internal electricity market and raises consumer prices",
            "Meeting the target helps allocate renewable electricity across the EU more efficiently",
        ],
        "risk_level": "Medium",
        "risk_note": "TSOs failing to meet the target face regulatory and reputational consequences.",
        "confidence": "High",
        "confidence_note": "ACER official monitoring framework and Electricity Regulation (EU) 2019/943.",
    },
    {
        "prompt": "What is the **REMIT framework** that ACER oversees, and what does it cover?",
        "completion": "**REMIT** (Regulation on Wholesale Energy Market Integrity and Transparency) is the EU legal framework prohibiting **market abuse** — including insider trading and market manipulation — in wholesale electricity and gas markets. ACER coordinates national regulatory authorities to monitor suspicious trading activity and conduct **cross-border investigations**. The Q4 2025 REMIT Quarterly report detailed new procedures for cross-border enforcement cooperation, reflecting the growing complexity of integrated EU energy markets.",
        "key_facts": [
            "REMIT prohibits insider trading and market manipulation in EU wholesale energy markets",
            "ACER coordinates national regulators for cross-border market surveillance",
            "Q4 2025 report introduced new cross-border investigation procedures",
        ],
        "risk_level": "Medium",
        "risk_note": "Non-compliant trading entities face regulatory investigation and fines.",
        "confidence": "High",
        "confidence_note": "ACER REMIT framework documentation and Q4 2025 quarterly report.",
    },
    {
        "prompt": "What challenges are slowing **hydrogen market development** in Europe, according to ACER's November 2025 monitoring report?",
        "completion": "According to ACER's November 2025 hydrogen monitoring report, European hydrogen market development is hampered by two main factors: **high production and infrastructure costs** and **slow transposition of EU hydrogen regulations** by member states. These delays create regulatory uncertainty that discourages private investment. Despite the certification of Europe's **first hydrogen transmission network operator** in March 2026 — supported by ACER — the overall pace of market development remains significantly behind EU decarbonisation targets.",
        "key_facts": [
            "High hydrogen production and infrastructure costs are a key barrier",
            "Slow national transposition of EU hydrogen regulations creates investment uncertainty",
            "First EU hydrogen transmission network operator certified in March 2026 (ACER-supported)",
        ],
        "risk_level": "High",
        "risk_note": "Regulatory delays risk missing EU 2030 hydrogen deployment targets.",
        "confidence": "High",
        "confidence_note": "ACER November 2025 hydrogen monitoring report.",
    },
    {
        "prompt": "What does **Eurostat** report on European energy, and what change did **nuclear power production** see in the EU in 2024?",
        "completion": "Eurostat publishes comprehensive EU energy statistics covering supply, transformation, and consumption across all fuel types, including renewables share, energy dependency ratios, household and industrial electricity and gas prices, and energy efficiency indicators. For **2024**, Eurostat reported a **+4.8% increase in nuclear power production** across EU member states, as several nuclear plants returned from extended maintenance. These statistics are consolidated annually in the *Energy in Europe* publication (2026 edition released 10 March 2026).",
        "key_facts": [
            "Eurostat tracks EU energy supply, dependency, renewables share, and prices",
            "Nuclear power production increased +4.8% in the EU in 2024",
            "Energy in Europe 2026 edition released 10 March 2026",
        ],
        "risk_level": "Low",
        "risk_note": "Statistical data; risk relates to policy implications of nuclear production changes.",
        "confidence": "High",
        "confidence_note": "Eurostat Energy Overview (2026).",
    },
    {
        "prompt": "What **share of global urea (fertiliser) trade** passes through the Strait of Hormuz, and what other commodities beyond energy are exposed to a Strait disruption?",
        "completion": "Over **30% of global urea trade** passes through the Strait of Hormuz. Other major commodities exposed include **ammonia and phosphate** (~20% of global trade), **aluminium** (~8% of global supply, ~5 million tonnes/year from Gulf smelters in Bahrain, Qatar, Saudi Arabia, UAE), and **sulphur** (~50% of seaborne trade). Sulphur is a critical input for petroleum refining and minerals processing (copper, nickel, zinc). A Strait closure therefore cascades far beyond energy into food security, industrial materials, and agricultural supply chains.",
        "key_facts": [
            ">30% of global urea trade via the Strait of Hormuz",
            "~20% of global ammonia and phosphate trade via the Strait",
            "~8% of global aluminium supply (Gulf smelters in Bahrain, Qatar, Saudi Arabia, UAE)",
            "~50% of global seaborne sulphur trade via the Strait",
        ],
        "risk_level": "High",
        "risk_note": "Strait disruptions impact food security and industrial materials beyond energy.",
        "confidence": "High",
        "confidence_note": "IEA 2026 Middle East market analysis, commodity exposure data.",
    },
    # ── HIGH DIFFICULTY ──────────────────────────────────────────────────────
    {
        "prompt": "Why does a **Strait of Hormuz closure disproportionately impact Asia** compared to Europe for LNG supply, and what does this mean for global gas price dynamics?",
        "completion": "Approximately **80–90% of Strait of Hormuz LNG volumes** are destined for **Asian markets** (Japan, South Korea, China, India), while only around **10%** flows to Europe. Asian buyers often rely on long-term Qatari contracts as the backbone of their supply, with limited alternatives in the Pacific Basin. Europe has greater access to **Atlantic Basin LNG** (US, Norway, West Africa) and pipeline gas, insulating it partially. The result is extreme **Asia-Europe price divergence** (JKM vs TTF), which incentivises spot LNG cargo diversion from Asia to Europe, further worsening Asia's supply position. Energy policy analysts recommend that Asian LNG importers accelerate **domestic LNG storage expansion** and diversify toward non-Strait-dependent exporters such as **Australia** and the **US**.",
        "key_facts": [
            "~80–90% of Strait LNG flows are destined for Asia; ~10% to Europe",
            "Asian buyers have fewer alternative supply sources than European buyers",
            "Price divergence between JKM (Asia) and TTF (Europe) incentivises cargo diversion toward Europe",
            "Recommended response: Asian storage expansion and supply diversification to Australia/US LNG",
        ],
        "risk_level": "High",
        "risk_note": "Asia faces acute supply shortfall with limited spot market alternatives.",
        "confidence": "High",
        "confidence_note": "IEA 2026 Middle East market analysis.",
    },
    {
        "prompt": "Analyse the **combined effect** of Gulf oil production cuts, refinery shutdowns, and Strait of Hormuz closure on global **middle distillates markets** (diesel, jet fuel).",
        "completion": "The combination of the Strait of Hormuz closure and **>3 mb/d of Gulf refining capacity shutdowns** creates compounding stress on global **middle distillates** (diesel, jet fuel) markets, which the IEA had already described as \"relatively tight\" pre-conflict. Gulf exporters shipped **3.3 mb/d of refined products** and **1.5 mb/d of LPG** through the Strait; losing these flows simultaneously removes supply at two value-chain points. Non-Gulf refiners (Europe, Northeast Asia, US) face both elevated Brent crude costs (up >40%) and the need to replace refined product imports. The tightness is especially acute for **aviation, shipping, and trucking**, and cascades into **agricultural supply chains** through fertiliser disruptions (>30% of global urea trade via Strait). Non-Gulf refiners have little spare diesel capacity to compensate for a sustained Gulf outage.",
        "key_facts": [
            ">3 mb/d of Gulf refining capacity shut due to attacks and lack of export outlets",
            "Gulf exported 3.3 mb/d refined products + 1.5 mb/d LPG via Strait pre-war",
            "Middle distillates already 'relatively tight' pre-conflict per IEA",
            "Brent crude futures rose >40%; cascading impact on fertiliser-dependent agriculture",
        ],
        "risk_level": "High",
        "risk_note": "Prolonged Gulf refinery shutdown would be extremely difficult to compensate globally.",
        "confidence": "High",
        "confidence_note": "IEA 2026 Middle East energy market report.",
    },
    {
        "prompt": "Explain how the **IEA's 90-day emergency stock rule** interacts with its **collective action mechanism**, and why the 2026 stock release was historically exceptional.",
        "completion": "The IEA's **90-day net import coverage rule** requires all 32 member countries to maintain strategic petroleum reserves (SPRs), creating a distributed buffer of over **1.25 billion barrels** in government stocks. This reserve is the foundation for **collective action**: when the Governing Board determines a severe disruption warrants intervention, members can vote to release agreed volumes simultaneously, preventing hoarding and maximising market stabilisation. The **400 million barrel release in March 2026** was unprecedented in two respects: (1) it was the **largest-ever single release** — roughly a third of all identified government emergency stocks — and (2) it was triggered by what the IEA described as **\"the largest supply disruption in the history of the global oil market\"**, with Gulf countries cutting at least **10 mb/d** and alternative bypass capacity (3.5–5.5 mb/d) wholly unable to compensate.",
        "key_facts": [
            "90-day rule creates >1.25 billion barrels of distributed government emergency stocks",
            "400 million barrels released in March 2026 — largest in IEA history",
            "Gulf production cuts of ≥10 mb/d triggered the release — largest-ever supply disruption",
            "Alternative bypass capacity (3.5–5.5 mb/d) far insufficient to offset the loss",
        ],
        "risk_level": "High",
        "risk_note": "Even the largest-ever stock release cannot fully compensate for 10 mb/d sustained loss.",
        "confidence": "High",
        "confidence_note": "IEA official announcement and 2026 market analysis.",
    },
    {
        "prompt": "Explain the importance of **bidding zone design** in EU electricity markets and how ACER's monitoring of the **70% CZC target** relates to market efficiency and decarbonisation.",
        "completion": "**Bidding zones** are geographic areas within which electricity prices are uniform, unconstrained by internal transmission limits. Their design determines how efficiently renewable electricity flows across the EU: poorly configured zones trap cheap wind or solar power behind congested borders, forcing consumers elsewhere to pay higher prices and compelling generators to curtail low-carbon output. ACER's enforcement of the **70% cross-zonal capacity target** ensures that TSOs make interconnector capacity genuinely available for cross-border trade, rather than reserving it domestically. Improving CZC compliance allows the EU to exploit its geographically diverse renewable base — Nordic hydro, Iberian solar, North Sea wind — more efficiently, lowers average consumer prices, reduces costly balancing needs, and is identified in ACER's 2026 flagship report as a **key pillar of EU global competitiveness**.",
        "key_facts": [
            "Bidding zones define areas of uniform electricity pricing; poor design wastes renewable capacity",
            "70% CZC target: TSOs must offer ≥70% of interconnector capacity for cross-border trade",
            "ACER monitors compliance and publishes annual market integration reports",
            "Higher CZC rates lower consumer prices, reduce curtailment, and enable decarbonisation",
        ],
        "risk_level": "Medium",
        "risk_note": "TSOs below 70% face regulatory action; market fragmentation raises costs for consumers.",
        "confidence": "High",
        "confidence_note": "ACER 2026 market integration report and Electricity Regulation (EU) 2019/943.",
    },
    # ── ADVISORY ────────────────────────────────────────────────────────────
    {
        "prompt": "Our country imports **over 60% of its LNG from Qatar**, all routed through the Strait of Hormuz. What **strategic steps** should our energy ministry take to reduce this supply vulnerability?",
        "completion": "Given Qatar routes ~93% of its LNG through the Strait of Hormuz, a 60%+ dependency exposes your country to catastrophic supply disruption during any regional conflict. Recommended strategic steps:\n\n**1. Diversify supply sources immediately.**\nEnter long-term contracts with Atlantic Basin suppliers (US, Norway, West Africa, Trinidad) whose export routes bypass the Strait entirely. Even partial diversification materially reduces concentration risk.\n\n**2. Build strategic LNG storage.**\nIncrease domestic LNG regasification and storage capacity to buffer at least 60–90 days of import demand, mirroring the IEA's 90-day oil stock logic.\n\n**3. Invest in alternative gas infrastructure.**\nDevelop pipeline interconnections with neighbouring countries and consider investments in non-gas power generation (nuclear, renewables) to reduce absolute LNG dependency.\n\n**4. Participate in international emergency mechanisms.**\nCoordinate with other LNG-importing nations on emergency stock sharing and joint procurement to leverage market power during crises.\n\n**5. Stress-test your energy security plans.**\nConduct scenario analyses assuming 3-, 6-, and 12-month Strait closures, and pre-negotiate emergency supply contracts with alternative suppliers that can be activated rapidly without long lead times.",
        "key_facts": [
            "Qatar routes ~93% of LNG exports via the Strait of Hormuz",
            ">60% single-corridor import dependency is extreme concentration risk",
            "Alternative suppliers: US (Sabine Pass, Corpus Christi), Norway, West Africa",
            "IEA 90-day stock logic can be applied to LNG strategic reserves",
        ],
        "risk_level": "High",
        "risk_note": "Without diversification, a Strait closure threatens immediate energy emergency.",
        "confidence": "High",
        "confidence_note": "Based on IEA 2026 analysis and energy security frameworks.",
    },
    {
        "prompt": "As an **energy portfolio manager** at a European utility, how should you restructure your **gas procurement strategy** to hedge against a sustained Middle East supply disruption?",
        "completion": "The 2026 Middle East crisis shows that even partial Strait of Hormuz disruption drives Dutch TTF up >55%. A European utility should adopt the following adjustments:\n\n**1. Reduce Strait-exposed volume.**\nAudit all LNG contracts for Gulf origin (Qatar, UAE). Prioritise renegotiating or replacing them with US LNG, Norwegian pipeline gas, or West African LNG before market premiums widen further.\n\n**2. Use financial hedging.**\nIncrease TTF forward contract and options coverage. Consider JKM-TTF basis swaps to monetise Asia-Europe price divergence created by supply shocks.\n\n**3. Diversify counterparties and routes.**\nAvoid supplier concentration: cap any single origin or transit route at no more than 25–30% of total gas procurement.\n\n**4. Activate demand flexibility.**\nNegotiate interruptible or fuel-switching contracts with large industrial consumers so they can reduce or shift gas demand during price spikes, lowering your spot market exposure.\n\n**5. Monitor IEA and EU emergency mechanisms.**\nIEA stock releases and EU emergency gas regulations dampen and shorten price spikes — understanding the trigger conditions helps calibrate your forward buying and inventory build strategy.",
        "key_facts": [
            "Dutch TTF rose >55% following 2026 Strait of Hormuz disruption",
            "Recommended single-origin cap: ≤25–30% of total gas procurement",
            "JKM-TTF basis swaps allow monetisation of Asia-Europe price divergence",
            "Demand flexibility through interruptible contracts reduces spot market exposure",
        ],
        "risk_level": "High",
        "risk_note": "Without diversification and hedging, utilities face unmanageable price and supply risk.",
        "confidence": "High",
        "confidence_note": "IEA 2026 analysis and energy risk management best practice.",
    },
    {
        "prompt": "My company imports **Qatari LNG via long-term SPA**. Strait of Hormuz tensions are escalating rapidly. **What concrete actions should we take now?**",
        "completion": "Given Qatar's ~93% Strait exposure (and the Ras Laffan shutdown precedent of 2 March 2026), you should act across three horizons simultaneously:\n\n**Immediate (0–2 weeks):**\n- Review your SPA force majeure (FM) clauses. Most Qatari contracts include FM for infrastructure attacks or government export orders. Understand your legal position if deliveries are suspended.\n- Contact alternative spot LNG suppliers (US traders, Nigerian LNG, Norwegian Equinor) to explore emergency cargo availability and indicative pricing before the market fully reprices.\n\n**Short-term (1–3 months):**\n- Secure alternative gas supply: Norwegian pipeline gas, Algerian routes, or US Atlantic LNG.\n- Fill terminal storage to maximum capacity immediately while supply still flows.\n- Notify downstream customers of potential supply constraints and pre-activate any demand reduction agreements.\n\n**Strategic (3–12 months):**\n- Negotiate a supplementary LNG supply agreement with an Atlantic Basin exporter to permanently reduce Strait-routed volume to below 30% of your portfolio.\n- Engage your national energy regulator on emergency supply obligations and available government stock mechanisms.\n- Hedge via TTF call options to cap cost exposure if you must replace Qatari cargoes at spot prices during disruption.",
        "key_facts": [
            "Ras Laffan (world's largest LNG facility) shut after 2 March 2026 attack",
            "Force majeure clauses in Qatari SPAs typically cover government export orders and infrastructure attacks",
            "Target: reduce Strait-routed LNG to <30% of total portfolio",
            "TTF call options can cap cost exposure for replacement spot cargoes",
        ],
        "risk_level": "High",
        "risk_note": "Inaction during escalation leaves the company exposed to full supply loss and spot price spike.",
        "confidence": "High",
        "confidence_note": "IEA 2026 Middle East analysis and LNG contract risk management best practice.",
    },
    {
        "prompt": "An EU member state consistently **fails to meet the 70% cross-zonal electricity capacity (CZC) target**. What should the national TSO do to achieve compliance, and what are the consequences of continued non-compliance?",
        "completion": "Persistent failure to meet the **70% CZC target** means the TSO is artificially restricting cross-border trade, fragmenting the EU internal electricity market. ACER monitors and reports this publicly.\n\n**Steps to achieve compliance:**\n\n1. **Conduct a network adequacy study** to identify constrained corridors and determine whether limits reflect genuine physical constraints or administrative conservatism.\n2. **Invest in grid infrastructure** — upgrade internal transmission lines and interconnectors bottlenecking cross-zonal flows. EU financing (CEF Energy, Innovation Fund) may be available.\n3. **Adopt Flow-Based Market Coupling (FBMC)** — this methodology demonstrates higher available capacity without compromising operational security, enabling the 70% target to be met on more hours.\n4. **Coordinate with neighbouring TSOs** through ENTSO-E Regional Coordination Centres to align operating security parameters and reduce joint curtailment conservatism.\n\n**Consequences of non-compliance:**\n- ACER reports deficits in annual monitoring publications, creating political and reputational pressure.\n- The European Commission can initiate infringement proceedings against the member state.\n- Persistent restriction of cross-border flows undermines EU market integration and may trigger financial penalties under Electricity Regulation (EU) 2019/943.",
        "key_facts": [
            "70% CZC target requires TSOs to make ≥70% of interconnector capacity available for trade",
            "ACER monitors and publicly reports compliance annually",
            "Flow-Based Market Coupling (FBMC) enables higher demonstrated available capacity",
            "Continued non-compliance can lead to EC infringement proceedings",
        ],
        "risk_level": "Medium",
        "risk_note": "Legal and financial consequences for TSOs; higher consumer costs from market fragmentation.",
        "confidence": "High",
        "confidence_note": "ACER monitoring framework and Electricity Regulation (EU) 2019/943.",
    },
    {
        "prompt": "Our industrial company is a **large electricity consumer** in Europe. ACER's 2025 retail monitoring found that **flexible contracts can lower consumer bills**. Should we switch to a flexible tariff?",
        "completion": "ACER's November 2025 finding that flexibility-focused contracts can unlock significant consumer savings is highly relevant for large industrial buyers. Here is a structured assessment:\n\n**When flexibility contracts clearly make sense:**\n- You have **shiftable or interruptible loads** — processes that can run at night or during high renewable generation periods (electrolysis, pumping, charging, heating).\n- You have **on-site storage** (batteries, thermal) allowing decoupling of consumption from production scheduling.\n- Your energy bill is large enough that even a 10–15% cost reduction is material.\n\n**Key risks to evaluate before switching:**\n1. **Price volatility exposure** — fully indexed spot-price contracts expose you to extreme spikes (TTF +55% in 2026). Ensure financial hedges or physical flexibility can manage this.\n2. **Operational feasibility** — assess which processes can genuinely shift without hidden production costs.\n3. **Contract terms** — review balancing responsibility, imbalance penalties, and metering requirements.\n4. **National market specifics** — ACER's findings are EU-wide; available flexibility products vary by member state and bidding zone.\n\n**Recommendation:** For large consumers with real demand flexibility, a **partial flexibility contract** (70–80% baseload hedged, 20–30% on day-ahead or spot) balances savings capture with price risk management.",
        "key_facts": [
            "ACER November 2025 retail monitoring: flexible contracts can unlock consumer savings",
            "Suitable for consumers with shiftable loads or on-site storage",
            "Fully indexed spot contracts carry extreme price spike risk (e.g., TTF +55% in 2026 crisis)",
            "Recommended approach: hybrid model (hedged baseload + partial spot/indexed)",
        ],
        "risk_level": "Medium",
        "risk_note": "Flexibility savings are real but require proper risk management to avoid price spike exposure.",
        "confidence": "High",
        "confidence_note": "ACER 2025 Retail Energy Market Monitoring Report.",
    },
    {
        "prompt": "Our country plans to **invest in hydrogen transmission infrastructure**. ACER's 2025 monitoring says hydrogen is \"hampered by high costs and slow transposition.\" **What should policymakers do?**",
        "completion": "ACER's November 2025 analysis identifies two interconnected barriers: **economic** (high costs) and **regulatory** (slow EU rule transposition). Policy response should address both:\n\n**Reducing costs:**\n1. **Prioritise repurposing existing gas pipelines** for hydrogen where technically feasible — dramatically cheaper than new-build infrastructure and reducible with ACER-supervised TSO cost benchmarking.\n2. **Support green hydrogen Contracts for Difference (CfD)** or Hydrogen Auctions to close the cost gap between green hydrogen and fossil alternatives, creating durable demand.\n3. **Cluster hydrogen off-takers geographically** to anchor demand near early infrastructure, reducing stranded-asset risk.\n\n**Accelerating regulation:**\n1. **Immediately transpose the EU Hydrogen and Decarbonised Gas Market Regulation** — transposition delays paralyse private investment.\n2. **Certify hydrogen network operators** following the precedent of Europe's first hydrogen TSO certification (March 2026, ACER-supported).\n3. **Engage ACER early** in designing national hydrogen tariff structures to ensure EU compatibility.\n\n**Overall strategic advice:** Early movers gain competitive advantage in attracting hydrogen corridor investment. Regulatory clarity — not market maturity — is the primary signal investors need.",
        "key_facts": [
            "ACER Nov 2025: hydrogen hampered by high costs and slow national regulatory transposition",
            "First EU hydrogen transmission network operator certified March 2026 (ACER-supported)",
            "Repurposing gas pipelines dramatically reduces hydrogen infrastructure costs",
            "Hydrogen CfD/auction mechanisms lower production cost gap vs fossil fuels",
        ],
        "risk_level": "High",
        "risk_note": "Delayed action risks missing EU 2030 hydrogen targets and loss of investment to faster-moving countries.",
        "confidence": "High",
        "confidence_note": "ACER 2025 hydrogen monitoring report and 2026 certification news.",
    },
    {
        "prompt": "A **commodity-dependent developing country** relies on **Gulf fertiliser imports** (urea, ammonia) for its agriculture. How should it plan for disruptions caused by **Strait of Hormuz instability**?",
        "completion": "The Strait of Hormuz carries **>30% of global urea trade** and **~20% of global ammonia and phosphate trade**, making it a critical chokepoint for global food security. For a developing country reliant on these imports:\n\n**Immediate risk mitigation:**\n1. **Build strategic fertiliser reserves** — maintain a minimum 3–6 month buffer stock timed to pre-planting seasons. This requires investment in warehousing and storage infrastructure.\n2. **Diversify fertiliser suppliers** — establish relationships with non-Gulf sources: Moroccan phosphate producers (OCP Group), Eastern European or North African urea exporters, or non-Strait-routed suppliers. Prioritise routes that bypass the Strait entirely.\n\n**Medium-term structural measures:**\n3. **Invest in domestic fertiliser production** — where natural gas or renewable energy permits, develop local ammonia or urea capacity to reduce import dependency.\n4. **Promote precision agriculture** — reducing fertiliser waste through optimised application lowers total import volumes and vulnerability.\n5. **Join regional procurement consortia** — coordinate with neighbouring countries to achieve bulk purchasing power and shared strategic reserves.\n\n**Policy framing:** Treat fertiliser supply security as a **national security issue** equivalent to energy security — apply the same diversification, strategic reserve, and emergency protocol logic that IEA applies to oil.",
        "key_facts": [
            ">30% of global urea trade and ~20% of ammonia/phosphate trade via the Strait",
            "A 3–6 month strategic fertiliser reserve is recommended as minimum buffer",
            "Non-Gulf suppliers: OCP Group (Morocco), East European/North African urea exporters",
            "Domestic ammonia/urea production reduces dependency where energy inputs allow",
        ],
        "risk_level": "High",
        "risk_note": "Without reserves and supplier diversification, a Strait closure threatens food security.",
        "confidence": "High",
        "confidence_note": "IEA 2026 Middle East analysis, commodity exposure data.",
    },
    {
        "prompt": "An **energy trader in the EU** asks: how should we ensure compliance with **REMIT** and avoid regulatory scrutiny from ACER during periods of market volatility?",
        "completion": "ACER's REMIT surveillance intensifies during periods of high market volatility, as sudden price spikes or unusual trading volumes can trigger algorithmic alerts. To ensure compliance:\n\n**1. Maintain complete and accurate trade reporting.**\nAll wholesale energy contracts and orders must be reported to ACER via Registered Reporting Mechanisms (RRMs) in near-real-time. Gaps or delays in reporting are a primary trigger for investigation.\n\n**2. Document the rationale for all trading decisions.**\nDuring volatility, trading strategies that look aggressive in hindsight must be defensible. Maintain timestamped records of market analysis, internal risk approvals, and price references used for each significant trade.\n\n**3. Avoid trading on any non-public information.**\nInsider information in energy markets includes unannounced production outages, undisclosed storage data, or unpublished regulatory decisions. Cancel or hold any trade if non-public material information is received.\n\n**4. Monitor cross-border dimension.**\nACER's Q4 2025 REMIT Quarterly introduced new cross-border investigation procedures — a single suspicious pattern spanning two member states can now be jointly investigated. Ensure your compliance programme covers all jurisdictions where you trade.\n\n**5. Conduct periodic internal REMIT audits.**\nA pre-crisis compliance review calibrates your monitoring systems, updates watchlists for positions near reportable thresholds, and ensures staff training is current.\n\nIn summary, REMIT compliance is fundamentally about **transparency and contemporaneous documentation** — if you can explain every trade with documented, legitimate commercial rationale, you are well-positioned.",
        "key_facts": [
            "All wholesale energy trades must be reported to ACER via RRMs in near-real-time",
            "ACER Q4 2025: new cross-border REMIT investigation procedures introduced",
            "Insider trading in energy includes unannounced outage data and undisclosed storage figures",
            "Documentation of trading rationale is the primary defence in any REMIT investigation",
        ],
        "risk_level": "High",
        "risk_note": "REMIT violations carry significant fines and reputational damage; cross-border cases are coordinated by ACER.",
        "confidence": "High",
        "confidence_note": "ACER REMIT framework and Q4 2025 REMIT Quarterly report.",
    },
]

# ---------------------------------------------------------------------------
# Write to energy_data_structured.jsonl (structured format)
# ---------------------------------------------------------------------------
def make_structured_completion(e: dict) -> str:
    facts_block = "\n".join(f"• {fact}" for fact in e["key_facts"])
    return (
        f"ANSWER: {e['completion']}\n\n"
        f"KEY FACTS:\n{facts_block}\n\n"
        f"RISK LEVEL: {e['risk_level']}\n"
        f"→ {e['risk_note']}\n\n"
        f"CONFIDENCE: {e['confidence']}\n"
        f"→ {e['confidence_note']}"
    )

structured_path = BASE / "energy_data_structured.jsonl"
structured_lines = [
    json.dumps(
        {"prompt": e["prompt"], "completion": make_structured_completion(e)},
        ensure_ascii=False,
    )
    for e in entries
]
with open(structured_path, "a", encoding="utf-8") as f:
    f.write("\n" + "\n".join(structured_lines) + "\n")
print(f"[+] Appended {len(structured_lines)} entries to {structured_path.name}")

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
struct_count = sum(1 for line in open(structured_path, encoding="utf-8") if line.strip())
print(f"\nTotal lines — structured: {struct_count}")
