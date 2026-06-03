"""Generate branded placeholder screenshots for the README."""
import math
import os
from PIL import Image, ImageDraw

# ── Palette ───────────────────────────────────────────────────────────────────
BG     = (14,  17,  23)
CARD   = (21,  27,  46)
BORDER = (30,  42,  58)
CYAN   = (0,  212, 255)
WHITE  = (232, 232, 232)
DIM    = (102, 102, 102)
GREEN  = (0,  200, 100)
RED    = (255,  68,  68)
GOLD   = (255, 215,   0)
PURPLE = (180, 100, 255)
ORANGE = (255, 165,   0)
GREY   = (150, 150, 150)

W, H = 1200, 700

os.makedirs("assets/screenshots", exist_ok=True)


def base_canvas(active_tab: int, subtitle: str):
    img  = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # ── sidebar ────────────────────────────────────────────
    draw.rectangle([0, 0, 230, H], fill=(11, 13, 19))
    draw.rectangle([230, 0, 232, H], fill=BORDER)
    draw.rectangle([14, 14, 50, 50], fill=CYAN)        # logo box
    draw.text((58, 20), "AlphaForge", fill=CYAN)
    draw.text((14, 58), "CONFIGURATION", fill=DIM)
    draw.rectangle([14, 75, 216, 76], fill=BORDER)

    labels = ["Asset Selection", "Date Range", "Capital / Costs",
              "Strategy", "Parameters", "Advanced"]
    for i, lbl in enumerate(labels):
        y = 84 + i * 70
        draw.rectangle([14, y, 216, y + 58], fill=(18, 23, 35), outline=BORDER)
        draw.text((22, y + 7), lbl, fill=WHITE)
        draw.rectangle([22, y + 27, 206, y + 44], fill=(25, 32, 50), outline=BORDER)

    # ── tab bar ────────────────────────────────────────────
    tab_labels = ["🔬 Research Lab", "📊 Portfolio Builder",
                  "⚔️ Comparison", "🧮 Risk Analytics",
                  "📋 Tear Sheet", "💾 Saved Runs"]
    tab_w = (W - 232) // len(tab_labels)
    draw.rectangle([232, 0, W, 46], fill=(12, 14, 20))
    for i, tlbl in enumerate(tab_labels):
        x0 = 232 + i * tab_w
        if i == active_tab:
            draw.rectangle([x0 + 2, 4, x0 + tab_w - 2, 42], fill=CYAN)
            draw.text((x0 + 8, 14), tlbl, fill=(14, 17, 23))
        else:
            draw.text((x0 + 8, 14), tlbl, fill=DIM)

    # ── subtitle bar ───────────────────────────────────────
    draw.text((240, 54), subtitle, fill=DIM)
    draw.rectangle([232, 70, W, 71], fill=BORDER)

    return img, draw


# ══════════════════════════════════════════════════════════════════════════════
# Screenshot 1 — Research Lab
# ══════════════════════════════════════════════════════════════════════════════
img1, d1 = base_canvas(0, "AAPL  -  Moving Average Crossover  -  2020-01-01 to 2024-12-31")

# metric cards
metrics = [
    ("CAGR",       "+18.4%",  GREEN),
    ("Sharpe",     "1.42",    CYAN),
    ("Max DD",     "-14.2%",  RED),
    ("Win Rate",   "58.1%",   CYAN),
    ("Sortino",    "1.87",    CYAN),
    ("Calmar",     "1.29",    GREEN),
]
cw = (W - 240) // 6
for i, (lbl, val, col) in enumerate(metrics):
    x0 = 238 + i * cw
    d1.rectangle([x0, 78, x0 + cw - 6, 138], fill=CARD, outline=BORDER)
    d1.text((x0 + 8, 86), lbl, fill=DIM)
    d1.text((x0 + 8, 106), val, fill=col)

# equity curve
d1.rectangle([238, 146, W - 8, 400], fill=CARD, outline=BORDER)
d1.text((250, 154), "Portfolio Value  ($100,000 → $231,450)", fill=DIM)
pts = []
for i in range(0, 940, 3):
    x = 246 + i
    y = int(390 - 180 * (i / 940) ** 0.72
            - 14 * math.sin(i / 38)
            - 7  * math.sin(i / 15))
    pts.append((x, y))
for i in range(len(pts) - 1):
    d1.line([pts[i], pts[i + 1]], fill=CYAN, width=2)
# benchmark line
pts_b = []
for i in range(0, 940, 3):
    x = 246 + i
    y = int(390 - 130 * (i / 940) ** 0.65 - 9 * math.sin(i / 45))
    pts_b.append((x, y))
for i in range(len(pts_b) - 1):
    d1.line([pts_b[i], pts_b[i + 1]], fill=GREY, width=1)
d1.text((W - 120, 154), "— Strategy", fill=CYAN)
d1.text((W - 120, 170), "— SPY B&H",  fill=GREY)

# drawdown strip
d1.rectangle([238, 408, W - 8, 490], fill=CARD, outline=BORDER)
d1.text((250, 416), "Drawdown", fill=DIM)
for i in range(0, 940, 3):
    x = 246 + i
    depth = int(24 * abs(math.sin(i / 58))) + int(8 * abs(math.sin(i / 21)))
    if depth > 4:
        d1.rectangle([x, 484 - depth, x + 3, 484], fill=(200, 50, 50))

# trade log
d1.rectangle([238, 498, W - 8, 692], fill=CARD, outline=BORDER)
d1.text((250, 506), "Trade Log", fill=DIM)
trade_rows = [
    ("2021-03-15", "BUY",  "127.43", "+$1,240"),
    ("2021-07-22", "SELL", "148.12", "+$2,105"),
    ("2022-01-10", "BUY",  "171.83", "-$834"),
    ("2022-09-30", "SELL", "150.43", "+$421"),
]
for ri, (dt, side, px, pnl) in enumerate(trade_rows):
    y = 524 + ri * 36
    d1.rectangle([248, y, W - 16, y + 30], fill=(18, 23, 35), outline=BORDER)
    d1.text((256, y + 7), dt,   fill=DIM)
    d1.text((400, y + 7), side, fill=GREEN if side == "BUY" else RED)
    d1.text((500, y + 7), px,   fill=WHITE)
    d1.text((620, y + 7), pnl,  fill=GREEN if pnl.startswith("+") else RED)

img1.save("assets/screenshots/research_lab.png")
print("research_lab.png done")


# ══════════════════════════════════════════════════════════════════════════════
# Screenshot 2 — Portfolio Builder
# ══════════════════════════════════════════════════════════════════════════════
img2, d2 = base_canvas(1, "Multi-asset portfolio  -  AAPL 30%  MSFT 25%  GOOGL 25%  NVDA 20%")

# donut chart
cx, cy, r_outer, r_inner = 400, 340, 120, 65
slices = [
    (0.30, CYAN,          "AAPL  30%"),
    (0.25, (0, 150, 200), "MSFT  25%"),
    (0.25, (0,  80, 130), "GOOGL 25%"),
    (0.20, (100,180,230), "NVDA  20%"),
]
angle = -math.pi / 2
for frac, col, lbl in slices:
    end_a = angle + 2 * math.pi * frac
    poly = [(cx, cy)]
    for step in range(61):
        a = angle + (end_a - angle) * step / 60
        poly.append((cx + r_outer * math.cos(a), cy + r_outer * math.sin(a)))
    d2.polygon(poly, fill=col)
    angle = end_a
d2.ellipse([cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner], fill=CARD)
d2.text((cx - 34, cy - 8), "Portfolio", fill=DIM)

# legend
for i, (_, col, lbl) in enumerate(slices):
    d2.rectangle([560, 250 + i * 40, 578, 268 + i * 40], fill=col)
    d2.text((590, 252 + i * 40), lbl, fill=WHITE)

# portfolio metrics
port_metrics = [
    ("Portfolio Sharpe",      "1.31",  CYAN),
    ("Portfolio CAGR",        "+21.4%", GREEN),
    ("Diversification Ratio", "1.18",  CYAN),
    ("HHI Concentration",     "0.267", GOLD),
    ("Avg Pairwise Corr.",    "0.61",  WHITE),
    ("Portfolio Vol (Ann.)",  "18.7%", RED),
]
for i, (lbl, val, col) in enumerate(port_metrics):
    x0 = 760 + (i % 2) * 205
    y0 = 230 + (i // 2) * 80
    d2.rectangle([x0, y0, x0 + 195, y0 + 68], fill=CARD, outline=BORDER)
    d2.text((x0 + 8, y0 + 8),  lbl, fill=DIM)
    d2.text((x0 + 8, y0 + 32), val, fill=col)

# correlation heatmap
d2.rectangle([238, 480, 700, 692], fill=CARD, outline=BORDER)
d2.text((250, 488), "Correlation Matrix", fill=DIM)
tickers4 = ["AAPL", "MSFT", "GOOGL", "NVDA"]
corr4 = [[1.00, 0.73, 0.65, 0.71],
         [0.73, 1.00, 0.68, 0.69],
         [0.65, 0.68, 1.00, 0.62],
         [0.71, 0.69, 0.62, 1.00]]
cell = 44
for ri in range(4):
    for ci in range(4):
        v   = corr4[ri][ci]
        col = (0, 180, 80) if ri == ci else (int(v * 220), int((1 - v) * 80), int(v * 60))
        x0  = 310 + ci * cell
        y0  = 508 + ri * cell
        d2.rectangle([x0, y0, x0 + cell - 2, y0 + cell - 2], fill=col, outline=BORDER)
        d2.text((x0 + 6, y0 + 13), f"{v:.2f}", fill=WHITE)
    d2.text((258, 508 + ri * cell + 13), tickers4[ri], fill=DIM)
for ci, t in enumerate(tickers4):
    d2.text((310 + ci * cell + 8, 496), t, fill=DIM)

# equity curve
d2.rectangle([710, 480, W - 8, 692], fill=CARD, outline=BORDER)
d2.text((722, 488), "Portfolio vs. SPY", fill=DIM)
for si, (scol, base) in enumerate([(CYAN, 0.9), (GREY, 0.72)]):
    pts = []
    for i in range(0, 470, 3):
        x = 718 + i
        y = int(680 - 145 * base * (i / 470) ** 0.75 - 10 * math.sin(i / 30 + si))
        pts.append((x, y))
    for i in range(len(pts) - 1):
        d2.line([pts[i], pts[i + 1]], fill=scol, width=2 if si == 0 else 1)

img2.save("assets/screenshots/portfolio_builder.png")
print("portfolio_builder.png done")


# ══════════════════════════════════════════════════════════════════════════════
# Screenshot 3 — Strategy Comparison
# ══════════════════════════════════════════════════════════════════════════════
img3, d3 = base_canvas(2, "6 strategies ranked  -  AAPL  -  2020-01-01 to 2024-12-31")

# ranked table header
cols  = ["Rank", "Strategy",         "Sharpe", "CAGR",    "Max DD",  "Calmar", "Win Rate"]
cws   = [52,     230,                 80,       80,        80,        80,       80]
rows3 = [
    ["#1", "MA Crossover",    "1.42", "+18.4%", "-14.2%", "1.29", "58.1%"],
    ["#2", "Momentum",         "1.31", "+16.1%", "-15.8%", "1.02", "54.3%"],
    ["#3", "RSI Mean Rev.",    "1.18", "+14.7%", "-13.6%", "1.08", "56.9%"],
    ["#4", "Bollinger Bands",  "1.07", "+13.2%", "-16.1%", "0.82", "52.7%"],
    ["#5", "Mean Reversion",   "0.94", "+11.8%", "-18.4%", "0.64", "51.2%"],
    ["#6", "Buy & Hold",       "0.87", "+10.3%", "-20.1%", "0.51", "—"],
]
x0h = 240
d3.rectangle([240, 78, W - 8, 104], fill=(18, 24, 40), outline=BORDER)
for c, cw in zip(cols, cws):
    d3.text((x0h + 6, 84), c, fill=CYAN)
    x0h += cw

for ri, row3 in enumerate(rows3):
    y = 106 + ri * 38
    bg = CARD if ri % 2 == 0 else (16, 21, 34)
    if ri == 0:
        bg = (14, 28, 14)
    d3.rectangle([240, y, W - 8, y + 36], fill=bg, outline=BORDER)
    x = 240
    for j, (val, cw) in enumerate(zip(row3, cws)):
        if j == 0 and ri == 0:
            col = GOLD
        elif "+" in val:
            col = GREEN
        elif "-" in val and "%" in val:
            col = RED
        elif val == "—":
            col = DIM
        else:
            col = WHITE
        d3.text((x + 6, y + 10), val, fill=col)
        x += cw

# overlaid equity curves
d3.rectangle([240, 342, W - 8, 548], fill=CARD, outline=BORDER)
d3.text((252, 350), "Equity Curves — All Strategies", fill=DIM)
strat_colors = [CYAN, GREEN, ORANGE, PURPLE, GOLD, GREY]
strat_labels = ["MA Crossover", "Momentum", "RSI MR", "Bollinger", "Mean Rev.", "Buy & Hold"]
strat_gains  = [0.90, 0.75, 0.65, 0.55, 0.44, 0.38]
for si, (scol, sgain) in enumerate(zip(strat_colors, strat_gains)):
    pts = []
    for i in range(0, 928, 4):
        x = 248 + i
        y = int(540 - 155 * sgain * (i / 928) ** 0.78
                - 10 * math.sin(i / 40 + si * 0.8)
                - 5  * math.sin(i / 18 + si))
        pts.append((x, y))
    for i in range(len(pts) - 1):
        d3.line([pts[i], pts[i + 1]], fill=scol, width=2 if si == 0 else 1)
for si, (scol, slbl) in enumerate(zip(strat_colors, strat_labels)):
    d3.rectangle([256 + si * 155, 358, 272 + si * 155, 370], fill=scol)
    d3.text((278 + si * 155, 356), slbl, fill=scol)

# callout box
d3.rectangle([240, 556, W - 8, 692], fill=(12, 22, 12), outline=(0, 110, 55))
d3.text((256, 568), "🏆  Best Strategy: MA Crossover  —  Sharpe 1.42  ·  CAGR +18.4%  ·  Max DD -14.2%",
        fill=GREEN)
d3.text((256, 600), "Forward Test: Train 70% / Test 30%  ·  Degradation Ratio: 0.91  ·  No overfitting detected",
        fill=DIM)
d3.text((256, 630), "Factor exposures:  Market β 0.82  ·  α (ann.) +6.1%  ·  R² 0.71",
        fill=DIM)
d3.text((256, 660), "Stress: 2008 GFC est. -18.4%  ·  COVID est. -12.1%  ·  2022 Rate Hike est. -9.8%",
        fill=DIM)

img3.save("assets/screenshots/strategy_comparison.png")
print("strategy_comparison.png done")

print("\nAll screenshots saved to assets/screenshots/")
