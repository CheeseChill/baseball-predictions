# Baseball Predictions — 6-Month Feature Roadmap

## Month 1: Daily Experience Improvements

- **Game-day push notifications** — notify when a game has edge > 8% within 2 hours of first pitch.
- **Bet slip builder** — user selects picks from Today page → generates a formatted text summary to paste into DraftKings.
- **Confidence badge tooltip** — hover over ✅/➡/⛔ badge to show the three model outputs that drove the signal.
- **Time-zone aware schedule** — auto-convert game times to the user's local time zone via a cookie or URL param.

## Month 2: Data Enrichment

- **ESPN game notes integration** — pull weather summary, lineup card, and starting pitcher confirmed/doubtful status from ESPN API and surface on the Today page.
- **Injury report widget** — prominent IL callout for key position players (top-5 fWAR players on roster).
- **Last 5 head-to-head mini table** — H2H results for the exact SP matchup on the Matchup Analysis page.

## Month 3: Analytics Pages

- **Season Trends page** — team-level rolling 10-game win%, run-scoring, and pitcher ERA plotted by week.
- **Bet type ROI drilldown** — separate Performance tab rows for Moneyline / Run Line / Totals per team.
- **Park factor visualiser** — interactive bar chart of all 30 park factors vs. league average for overs/unders.

## Month 4: User Personalisation

- **Watchlist** — users mark 3–5 teams as favourites; Today page leads with those games.
- **Bankroll tracker** — enter starting bankroll and auto-calculate Kelly units in dollars.
- **Bet history upload** — paste CSV of past bets to see P&L, ROI, and model accuracy vs. your actual results.

## Month 5: Model Transparency

- **Per-game SHAP waterfall** — expandable panel on each game card showing the top drivers.
- **Model accuracy history** — chart of 30-day rolling accuracy for each model, visible in the Models tab.
- **Edge distribution histogram** — how often does the model find >3%, >6%, >10% edge? Helps users calibrate expectations.

## Month 6: Automation & Scale

- **Pre-game email digest** — daily 7 AM ET email listing today's high-confidence picks with game context.
- **Live line movement alerts** — detect when DraftKings line moves > 10 cents from opening; highlight on dashboard.
- **Multi-season backtester** — configure date range, bet type, edge threshold → show simulated P&L curve.
