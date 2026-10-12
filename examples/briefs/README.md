# Example briefs

Five briefs from the gateway model trials, written by Qwen3.8-27B BF16 with the observer guidance on (repeat 1). Each
one is the gateway's Markdown output after the policy checks, with the test-set id as the heading. The full JSON for these runs, with evidence, planner
turns and trace, is in `experiments/gateway-model-trials/results/runs_v2.jsonl.gz`.

| Brief | Truth | Data | What it shows |
|---|---|---|---|
| [FL-A_plot1_burn_thermal](FL-A_plot1_burn_thermal.md) | fire | RGB + thermal | Large fire front; the writeup and thermal agree |
| [SY-2_5](SY-2_5.md) | fire | RGB + thermal | Sycan Marsh fire; thermal backs the call |
| [FL-E_plot2_takeoff_smoke_thermal](FL-E_plot2_takeoff_smoke_thermal.md) | no fire | RGB + thermal | Takeoff over smoke per the sequence notes; no pixel at 80 C and the writeup sees no smoke: correctly no fire |
| [FL-D_plot1_preburn_rgb](FL-D_plot1_preburn_rgb.md) | no fire | RGB only | Pre-burn forest: correctly no fire |
| [FL-G_plot2_smallfire_rgb](FL-G_plot2_smallfire_rgb.md) | fire | RGB only | The hard case. The writeup says no fire and takes the glow for lens flare. The measurement rule sees flames on 2 of 3 image checks and flags it, but on RGB-only evidence the rule is advice, so it appears as a limitation and does not override the writeup |
