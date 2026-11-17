// Qwen planner tool calls on the escalated sequences.
//
// Pulled verbatim from experiments/decision-cascade/results/runs_v3.jsonl.gz
// (label "cascade-d3"): the calls from run.planner.turns[].tool_calls, the
// answers from run.evidence.planner_probes.
//
// Only sequences that fall under d3's 0.80 confidence bar escalate and reach
// Qwen, so only those two have tool calls. Everything else stops at d3.
//
// Important: these answers feed the written brief (where the fire is, smoke vs
// flames). They do not make the fire / no-fire call; that comes from the
// measured evidence (detector, CLIP, thermal).
window.TOOL_CALLS = {
  "SY-2_5": {
    rounds: 2, finished: true,
    calls: [
      { tool: "decide", frame: "f05", probes: [
        { q: "flames", a: "no", p: 0.88 },
        { q: "smoke", a: "yes", p: 0.98 },
        { q: "region", a: "bottom", p: 0.65 } ] },
      { tool: "decide", frame: "f06", probes: [
        { q: "flames", a: "no", p: 0.98 },
        { q: "smoke", a: "no", p: 0.92 },
        { q: "region", a: "none", p: 1.00 } ] },
      { tool: "decide", frame: "f02", probes: [
        { q: "flames", a: "no", p: 0.84 },
        { q: "smoke", a: "yes", p: 0.99 },
        { q: "region", a: "center", p: 0.62 } ] },
      { tool: "finish" }
    ]
  },
  "FL-G_plot2_smallfire_rgb": {
    rounds: 2, finished: true,
    calls: [
      { tool: "decide", frame: "f08", probes: [
        { q: "flames", a: "yes", p: 1.00 },
        { q: "smoke", a: "no", p: 0.71 },
        { q: "region", a: "right", p: 0.70 } ] },
      { tool: "decide", frame: "f07", probes: [
        { q: "flames", a: "yes", p: 0.99 },
        { q: "smoke", a: "no", p: 0.71 },
        { q: "region", a: "none", p: 0.54 } ] },
      { tool: "decide", frame: "f06", probes: [
        { q: "flames", a: "yes", p: 0.96 },
        { q: "smoke", a: "no", p: 0.76 },
        { q: "region", a: "none", p: 0.75 } ] },
      { tool: "finish" }
    ]
  }
};
