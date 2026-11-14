/* Replay console logic. Reads window.DEMO (see data.js and export_demo.py). The
   page never calls a model: it replays the saved pipeline output for each
   sequence. Per sequence it shows three real things:
     1. the cascade path   - what d3 answered, and whether the cascade accepted
                              that call or escalated to Qwen3.8-27B;
     2. per-frame measurements - detector p(fire), CLIP p(fire), thermal hot
                              area, replayed on a scrubber and a chart;
     3. the recon brief    - the structured Qwen3.8-27B writeup.
   Spread direction is not reported (indeterminate on this data). */
(function () {
  "use strict";

  var DEMO = window.DEMO || { index: { sequences: [] }, sequences: {} };
  var state = { seqId: null, i: 0, tp: 0, playing: false, raf: null, lastTs: null,
                flowStep: -1, flowSteps: [], anims: {},
                measureShown: false, d3Shown: false, briefShown: false };
  var STEP_MS = 900;
  var toggles = { mask: true, score: true, clip: true, thermal: true };
  var $ = function (sel) { return document.querySelector(sel); };

  // ---- Theme -------------------------------------------------------------
  function initTheme() {
    var saved = null;
    try { saved = localStorage.getItem("recon-theme"); } catch (e) {}
    if (saved === "dark" || saved === "light") document.documentElement.setAttribute("data-theme", saved);
    $("#theme").addEventListener("click", function () {
      var cur = document.documentElement.getAttribute("data-theme");
      var isDark = cur ? cur === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
      var next = isDark ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("recon-theme", next); } catch (e) {}
      drawChart();
    });
  }

  // ---- Picker ------------------------------------------------------------
  function renderPicker() {
    var list = $("#seq-list");
    list.innerHTML = "";
    DEMO.index.sequences.forEach(function (s) {
      var li = document.createElement("li");
      var b = document.createElement("button");
      b.className = "seq-item";
      b.setAttribute("data-id", s.id);
      b.setAttribute("aria-current", String(s.id === state.seqId));
      var srcTag = s.has_thermal
        ? '<span class="tag">thermal</span>'
        : '<span class="tag">RGB only</span>';
      var resTag = s.correct
        ? '<span class="tag ok">correct</span>'
        : '<span class="tag miss">miss</span>';
      b.innerHTML =
        '<span class="name">' + esc(s.display_name) + "</span>" +
        '<span class="meta">' + srcTag + resTag + "</span>";
      b.addEventListener("click", function () { selectSeq(s.id); });
      li.appendChild(b);
      list.appendChild(li);
    });
  }

  function selectSeq(id) {
    state.seqId = id;
    state.i = 0;
    stop();
    var seq = DEMO.sequences[id];
    document.querySelectorAll(".seq-item").forEach(function (el) {
      el.setAttribute("aria-current", String(el.getAttribute("data-id") === id));
    });
    initFlow(seq);
    var scrub = $("#scrub");
    scrub.max = String(seq.frames.length - 1);
    scrub.value = "0";
    render();
  }

  // ---- Decision pipeline (interactive) -----------------------------------
  // Same five nodes for every sequence. d3 answers first; a gate checks its
  // confidence against the accept threshold. Confident -> the Qwen second
  // opinion is X'd out and bypassed; not confident -> it escalates to Qwen.
  // "Begin analysis" then "Next step" reveal the path one node at a time.
  var NODES = ["measure", "d3", "gate", "qwen", "final"];

  function initFlow(seq) {
    var escalate = seq.cascade.escalated;
    // The node sequence is identical except the Qwen node is only a real step
    // when the sequence escalates; otherwise it is reached but drawn bypassed.
    state.flowSteps = escalate
      ? ["measure", "d3", "gate", "qwen", "final"]
      : ["measure", "d3", "gate", "final"];
    state.flowStep = -1;
    state.measureShown = state.d3Shown = state.briefShown = false;
    cancelAnims();
    // Nothing is revealed until you step onto its node.
    ["#m-det", "#m-clip", "#m-therm", "#d3-fire", "#d3-smoke", "#d3-none", "#d3-unclear"]
      .forEach(function (s) { $(s).textContent = ""; });
    ["fire", "smoke_only", "none", "unclear"].forEach(function (k) {
      document.getElementById("d3row-" + k).classList.remove("win");
    });
    $("#nd-final").innerHTML = esc(seq.cascade.final_status) + (seq.cascade.correct ? " &check;" : " &times;");
    $("#brief-body").innerHTML =
      '<p class="brief-wait">Step through to the <b>Final brief</b> to see the result.</p>';
    renderFlow(seq);
  }

  // ---- reveal animations -------------------------------------------------
  // Count a readout up from zero to its real value when its node is revealed.
  function countUp(sel, target, decimals, suffix) {
    var el = $(sel);
    cancelAnim(sel);
    if (target == null) { el.textContent = "n/a"; return; }
    var t0 = null, dur = 1300;
    function frame(ts) {
      if (t0 == null) t0 = ts;
      var k = Math.min(1, (ts - t0) / dur);
      var e = 1 - Math.pow(1 - k, 3);
      el.textContent = (target * e).toFixed(decimals) + suffix;
      if (k < 1) state.anims[sel] = requestAnimationFrame(frame);
      else { el.textContent = target.toFixed(decimals) + suffix; delete state.anims[sel]; }
    }
    state.anims[sel] = requestAnimationFrame(frame);
  }
  function cancelAnim(sel) { if (state.anims[sel]) { cancelAnimationFrame(state.anims[sel]); delete state.anims[sel]; } }
  function cancelAnims() { Object.keys(state.anims).forEach(cancelAnim); }

  function revealMeasure(seq) {
    var cur = seq.frames[state.i];
    countUp("#m-det", cur.detector_p_fire * 100, 0, "%");
    countUp("#m-clip", cur.clip_p_fire * 100, 0, "%");
    countUp("#m-therm", cur.thermal_hot_percent, 2, "%");
  }
  function revealD3(seq) {
    var sp = seq.cascade.status_probabilities || {};
    countUp("#d3-fire", (sp.fire || 0) * 100, 0, "%");
    countUp("#d3-smoke", (sp.smoke_only || 0) * 100, 0, "%");
    countUp("#d3-none", (sp.none || 0) * 100, 0, "%");
    countUp("#d3-unclear", (sp.unclear || 0) * 100, 0, "%");
    var keys = ["fire", "smoke_only", "none", "unclear"];
    var win = keys.reduce(function (a, b) { return (sp[b] || 0) > (sp[a] || 0) ? b : a; }, "fire");
    keys.forEach(function (k) { document.getElementById("d3row-" + k).classList.toggle("win", k === win); });
  }
  function revealBrief(seq) {
    renderBrief(seq);
    var body = $("#brief-body");
    body.classList.remove("revealing"); void body.offsetWidth; body.classList.add("revealing");
    var sum = document.getElementById("brief-summary");
    if (sum) typeWriter("#brief-summary", sum.textContent);
  }
  function typeWriter(sel, full) {
    var el = $(sel); if (!el) return;
    cancelAnim(sel); el.textContent = ""; el.classList.remove("done");
    var t0 = null, cps = 48;
    function frame(ts) {
      if (t0 == null) t0 = ts;
      var n = Math.min(full.length, Math.floor((ts - t0) / 1000 * cps));
      el.textContent = full.slice(0, n);
      if (n < full.length) state.anims[sel] = requestAnimationFrame(frame);
      else { delete state.anims[sel]; el.classList.add("done"); }
    }
    state.anims[sel] = requestAnimationFrame(frame);
  }

  function flowReached(name) {
    // Has the walkthrough revealed this node yet?
    var upto = state.flowSteps.slice(0, state.flowStep + 1);
    return upto.indexOf(name) !== -1;
  }

  function renderFlow(seq) {
    var c = seq.cascade, bypass = !c.escalated;
    var dia = $("#flow-diagram");
    dia.setAttribute("data-branch", bypass ? "bypass" : "escalate");

    // Light nodes that have been reached.
    NODES.forEach(function (n) {
      $("#n-" + n).classList.toggle("lit", flowReached(n));
    });
    // Once the gate is reached, show its verdict and the Qwen node's fate.
    var gateReached = flowReached("gate");
    $("#n-qwen").classList.toggle("bypassed", bypass && gateReached);
    if (bypass && gateReached) $("#n-qwen").classList.remove("lit");
    $("#nd-gate").innerHTML = gateReached ? (bypass ? "yes" : "no &rarr; escalate") : "&middot;";

    // Connectors. Only the thermal reading feeds d3; detector and CLIP stay as
    // evidence, so the lit line runs from Measure (thermal) into d3.
    setLine("ln-therm-d3", flowReached("d3"));
    setLine("ln-d3-gate", flowReached("gate"));
    setLine("ln-gate-final", (bypass && flowReached("final")));
    setLine("ln-gate-qwen", (!bypass && flowReached("qwen")));
    setLine("ln-qwen-final", (!bypass && flowReached("final")));

    // Evidence path: DINOv2 + CLIP feed Qwen, which writes the brief. It lights
    // when Qwen actually runs (the escalated path); on a bypass Qwen is skipped.
    var qwenReached = flowReached("qwen");
    setLine("ln-ev-qwen", qwenReached);
    dia.classList.toggle("ev-on", qwenReached);
    dia.classList.toggle("brief-on", flowReached("final"));
    $("#to-brief").innerHTML = bypass ? "d3&rsquo;s call &rarr;" : "Qwen&rsquo;s brief &rarr;";

    // Buttons.
    var started = state.flowStep >= 0;
    var done = state.flowStep >= state.flowSteps.length - 1;
    $("#flow-begin").hidden = started;
    $("#flow-next").hidden = !started || done;
    $("#flow-reset").hidden = !started;

    $("#flow-caption").innerHTML = flowCaption(seq);
  }

  function setLine(id, on) {
    var el = document.getElementById(id);
    el.classList.toggle("on", !!on);
  }

  function flowCaption(seq) {
    var c = seq.cascade, p = Math.round(c.decider_top_p * 100), acc = Math.round((c.accept_p || 0.8) * 100);
    if (state.flowStep < 0) return "Press <b>Begin analysis</b> to step through how this sequence is decided.";
    var cur = state.flowSteps[state.flowStep];
    if (cur === "measure") return "Three readings per frame: <b>DINOv2</b> (detector), <b>CLIP</b>, and the " +
      "deterministic 80&deg;C thermal hot-area. Thermal feeds d3's call; DINOv2 and CLIP feed the written brief.";
    if (cur === "d3") return "<b>d3</b> looks at all the frames itself and calls it <b>" + esc(c.decider_status) +
      "</b>, " + p + "% sure. Only the thermal reading feeds in; DINOv2 and CLIP stay as evidence.";
    if (cur === "gate") return c.escalated
      ? p + "% sure is under d3's " + acc + "% confidence bar, so it escalates to <b>Qwen 3.8</b> for a closer look."
      : p + "% sure clears d3's " + acc + "% confidence bar, so the <b>Qwen 3.8</b> second opinion is skipped.";
    if (cur === "qwen") return "<b>Qwen 3.8-27B</b> takes a deeper look, checking every frame, then writes the brief.";
    if (cur === "final") {
      var verdict = c.correct
        ? "matches the ground truth (" + esc(seq.truth) + ")"
        : "<b>misses</b> the ground truth (" + esc(seq.truth) + ")";
      return "Final call: <b>" + esc(c.final_status) + "</b>. It " + verdict + ".";
    }
    return "";
  }

  // Advance to the current step, revealing (and animating) whatever that node shows.
  function stepTo(seq) {
    var cur = state.flowSteps[state.flowStep];
    if (cur === "measure" && !state.measureShown) { state.measureShown = true; revealMeasure(seq); }
    if (cur === "d3" && !state.d3Shown) { state.d3Shown = true; revealD3(seq); }
    if (cur === "final" && !state.briefShown) { state.briefShown = true; revealBrief(seq); }
    renderFlow(seq);
  }
  function flowBegin() { if (state.flowStep < 0) { state.flowStep = 0; stepTo(DEMO.sequences[state.seqId]); } }
  function flowNext() {
    if (state.flowStep < state.flowSteps.length - 1) { state.flowStep++; stepTo(DEMO.sequences[state.seqId]); }
  }
  function flowReset() { initFlow(DEMO.sequences[state.seqId]); }

  function statusClass(s) {
    if (s === "fire") return "s-fire";
    if (s === "no fire or smoke") return "s-none";
    if (s === "smoke only") return "s-smoke";
    return "s-ind";
  }

  // ---- Transport ---------------------------------------------------------
  function play() {
    var seq = DEMO.sequences[state.seqId];
    if (state.tp >= 1) { state.i = 0; state.tp = 0; renderDiscrete(seq); }
    state.playing = true; $("#play").textContent = "Pause"; state.lastTs = null;
    state.raf = requestAnimationFrame(tick);
  }
  function stop() {
    state.playing = false;
    if (state.raf) { cancelAnimationFrame(state.raf); state.raf = null; }
    $("#play").textContent = "Play";
  }
  function tick(ts) {
    if (!state.playing) return;
    var seq = DEMO.sequences[state.seqId], n = seq.frames.length;
    if (state.lastTs == null) state.lastTs = ts;
    var dt = ts - state.lastTs; state.lastTs = ts;
    state.tp += dt / ((n - 1) * STEP_MS);
    if (state.tp > 1) state.tp = 1;
    var newI = Math.min(n - 1, Math.floor(state.tp * (n - 1) + 1e-6));
    if (newI !== state.i) { state.i = newI; $("#scrub").value = String(newI); renderDiscrete(seq); }
    drawDynamic(seq);
    if (state.tp >= 1) { stop(); return; }
    state.raf = requestAnimationFrame(tick);
  }

  function render() {
    var seq = DEMO.sequences[state.seqId];
    var n = seq.frames.length;
    state.tp = n > 1 ? state.i / (n - 1) : 1;
    renderDiscrete(seq);
    drawDynamic(seq);
  }

  function renderDiscrete(seq) {
    var cur = seq.frames[state.i];
    $("#stamp-l").textContent = seq.source.flight;
    $("#stamp-r").innerHTML = "T+" + fmt(cur.t_sec) + "<br>" + cur.clock;
    $("#counter").textContent = "Frame " + (state.i + 1) + " / " + seq.frames.length;
    updateMeasure(seq);
  }

  // Live readouts in the Measure node, tracking the current frame. Stays blank
  // until the walkthrough has revealed the Measure node.
  function updateMeasure(seq) {
    if (!state.measureShown) return;
    var cur = seq.frames[state.i];
    var pct = function (v) { return v == null ? "n/a" : Math.round(v * 100) + "%"; };
    if (!state.anims["#m-det"]) $("#m-det").textContent = pct(cur.detector_p_fire);
    if (!state.anims["#m-clip"]) $("#m-clip").textContent = pct(cur.clip_p_fire);
    if (!state.anims["#m-therm"]) $("#m-therm").textContent =
      cur.thermal_hot_percent == null ? "n/a" : cur.thermal_hot_percent.toFixed(2) + "%";
  }

  function drawDynamic(seq) { drawFrame(seq, state.i); drawChart(); }

  function sizeCanvas(c) {
    var dpr = window.devicePixelRatio || 1;
    var r = c.getBoundingClientRect();
    c.width = Math.max(1, Math.round(r.width * dpr));
    c.height = Math.max(1, Math.round(r.height * dpr));
    var ctx = c.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: r.width, h: r.height };
  }

  // Placeholder scene drawn from the numbers (no real drone stills bundled). The
  // thermal hot area is drawn as an orange blob at the brief's location word,
  // sized to the measured hot-area share. The score chip is the real detector
  // p(fire) for the frame.
  function drawFrame(seq, i) {
    var c = $("#frame");
    var s = sizeCanvas(c), ctx = s.ctx, W = s.w, H = s.h;
    var cur = seq.frames[i];

    // No real drone stills are bundled, so the frame is a thermal-style field
    // (dark IR gradient, faint grid, vignette, HUD corner ticks) with the real
    // measured hot area drawn on top. It is an honest stand-in, not a photo.
    var g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, "#14171c"); g.addColorStop(1, "#0a0c10");
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = "rgba(130,150,160,0.06)"; ctx.lineWidth = 1;
    for (var gx = 1; gx < 8; gx++) { ctx.beginPath(); ctx.moveTo(gx / 8 * W, 0); ctx.lineTo(gx / 8 * W, H); ctx.stroke(); }
    for (var gy = 1; gy < 5; gy++) { ctx.beginPath(); ctx.moveTo(0, gy / 5 * H); ctx.lineTo(W, gy / 5 * H); ctx.stroke(); }
    var vg = ctx.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.2, W / 2, H / 2, Math.max(W, H) * 0.7);
    vg.addColorStop(0, "rgba(0,0,0,0)"); vg.addColorStop(1, "rgba(0,0,0,0.5)");
    ctx.fillStyle = vg; ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = "rgba(180,200,210,0.22)"; ctx.lineWidth = 2;
    var tk = 16;
    [[14, 14, 1, 1], [W - 14, 14, -1, 1], [14, H - 14, 1, -1], [W - 14, H - 14, -1, -1]].forEach(function (c) {
      ctx.beginPath(); ctx.moveTo(c[0], c[1] + c[3] * tk); ctx.lineTo(c[0], c[1]); ctx.lineTo(c[0] + c[2] * tk, c[1]); ctx.stroke();
    });

    var hasHot = cur.radius_frac != null && cur.radius_frac > 0 && cur.centroid;
    if (hasHot) {
      var cx = cur.centroid[0] * W, cy = cur.centroid[1] * H;
      var rad = cur.radius_frac * Math.min(W, H) * 1.6;
      if (toggles.mask) {
        ctx.fillStyle = "rgba(255, 70, 0, 0.28)";
        ctx.strokeStyle = "rgba(255, 120, 40, 0.9)";
        ctx.lineWidth = 2; ctx.setLineDash([5, 4]);
        ctx.beginPath(); ctx.arc(cx, cy, rad, 0, 7); ctx.fill(); ctx.stroke();
        ctx.setLineDash([]);
      }
      var fg = ctx.createRadialGradient(cx, cy, 1, cx, cy, rad * 1.1);
      fg.addColorStop(0, "rgba(255,236,150,0.95)");
      fg.addColorStop(0.4, "rgba(255,150,40,0.85)");
      fg.addColorStop(1, "rgba(200,40,0,0)");
      ctx.fillStyle = fg;
      ctx.beginPath(); ctx.arc(cx, cy, rad * 1.1, 0, 7); ctx.fill();
    } else if (!seq.source.has_thermal) {
      ctx.fillStyle = "rgba(255,255,255,0.5)";
      ctx.font = "600 12px IBM Plex Mono, monospace"; ctx.textAlign = "center";
      ctx.fillText("RGB only - no thermal hot-area measurement", W / 2, 22);
    }

    // Measurement chips, stacked bottom-left, one per enabled toggle.
    var chips = [];
    if (toggles.score && cur.detector_p_fire != null) chips.push({ t: "DINOv2   " + cur.detector_p_fire.toFixed(2), c: "#ffd27a" });
    if (toggles.clip && cur.clip_p_fire != null) chips.push({ t: "CLIP   " + cur.clip_p_fire.toFixed(2), c: "#9ccbf0" });
    if (toggles.thermal) chips.push({
      t: cur.thermal_hot_percent != null
        ? "Thermal   " + cur.thermal_hot_percent.toFixed(2) + "%" + (cur.thermal_max_c != null ? "   " + Math.round(cur.thermal_max_c) + "C" : "")
        : "Thermal   n/a",
      c: "#f0a06a"
    });
    ctx.font = "600 13px IBM Plex Mono, monospace"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
    var chY = H - 22;
    for (var ci = chips.length - 1; ci >= 0; ci--) {
      var cw = ctx.measureText(chips[ci].t).width + 16;
      ctx.fillStyle = "rgba(0,0,0,0.55)";
      roundRect(ctx, 12, chY - 11, cw, 22, 5); ctx.fill();
      ctx.fillStyle = chips[ci].c;
      ctx.fillText(chips[ci].t, 20, chY);
      chY -= 28;
    }
  }

  // ---- Chart -------------------------------------------------------------
  function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

  function chartSeries(seq) {
    var cDet = cssVar("--series-score") || "#e0a93b";
    var cClip = cssVar("--series-clip") || "#6aa9e0";
    var cHot = cssVar("--series-area") || "#e2622f";
    var out = [
      { key: "detector_p_fire", label: "DINOv2 p(fire)", color: cDet, scale: 100 },
      { key: "clip_p_fire", label: "CLIP p(fire)", color: cClip, scale: 100 },
    ];
    if (seq.source.has_thermal) {
      out.push({ key: "radius_frac", label: "Thermal hot-area radius", color: cHot, scale: 100 });
    }
    return out;
  }

  function drawChart() {
    var seq = DEMO.sequences[state.seqId];
    if (!seq) return;
    var c = $("#chart");
    var s = sizeCanvas(c), ctx = s.ctx, W = s.w, H = s.h;
    var f = seq.frames, n = f.length;
    var padL = 34, padR = 10, padT = 10, padB = 22;
    var plotW = W - padL - padR, plotH = H - padT - padB;
    var x = function (idx) { return padL + (n === 1 ? 0 : (idx / (n - 1)) * plotW); };
    var y = function (pct) { return padT + (1 - pct / 100) * plotH; };
    var grid = cssVar("--grid"), muted = cssVar("--muted"), fg = cssVar("--fg");
    var series = chartSeries(seq);

    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = grid; ctx.lineWidth = 1;
    ctx.font = "10px IBM Plex Mono, monospace"; ctx.fillStyle = muted; ctx.textBaseline = "middle";
    [0, 25, 50, 75, 100].forEach(function (v) {
      var yy = y(v);
      ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(W - padR, yy); ctx.stroke();
      ctx.textAlign = "right"; ctx.fillText(v + "%", padL - 6, yy);
    });

    series.forEach(function (ser) {
      ctx.strokeStyle = ser.color; ctx.lineWidth = 2; ctx.beginPath();
      var started = false;
      f.forEach(function (fr, idx) {
        var raw = fr[ser.key];
        if (raw == null) return;
        var xx = x(idx), yy = y(raw * ser.scale);
        if (!started) { ctx.moveTo(xx, yy); started = true; } else ctx.lineTo(xx, yy);
      });
      ctx.stroke();
    });

    var mx = x(state.i);
    ctx.strokeStyle = fg; ctx.globalAlpha = 0.5; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(mx, padT); ctx.lineTo(mx, H - padB); ctx.stroke();
    ctx.setLineDash([]); ctx.globalAlpha = 1;
    series.forEach(function (ser) {
      var raw = f[state.i][ser.key];
      if (raw == null) return;
      ctx.fillStyle = ser.color; ctx.beginPath(); ctx.arc(mx, y(raw * ser.scale), 4, 0, 7); ctx.fill();
    });

    $("#chart-legend").innerHTML = series.map(function (ser) {
      return '<span><i style="background:' + ser.color + '"></i>' + esc(ser.label) + "</span>";
    }).join("");

    c._x = x; c._n = n; c._padL = padL; c._padR = padR;
  }

  function initChartHover() {
    var box = $("#chartbox"), tip = $("#chart-tip"), c = $("#chart");
    function idxAt(clientX) {
      var r = box.getBoundingClientRect();
      var frac = (clientX - r.left - c._padL) / (r.width - c._padL - c._padR);
      return Math.max(0, Math.min(c._n - 1, Math.round(frac * (c._n - 1))));
    }
    box.addEventListener("mousemove", function (e) {
      var seq = DEMO.sequences[state.seqId];
      if (!seq || !c._n) return;
      var fr = seq.frames[idxAt(e.clientX)];
      tip.hidden = false;
      tip.style.left = c._x(fr.index) + "px"; tip.style.top = "6px";
      var hot = fr.thermal_hot_percent != null ? " · hot " + fr.thermal_hot_percent.toFixed(2) + "%" : "";
      tip.innerHTML = fr.clock + " · dino " + fr.detector_p_fire.toFixed(2) +
        " · clip " + fr.clip_p_fire.toFixed(2) + hot;
    });
    box.addEventListener("mouseleave", function () { tip.hidden = true; });
    box.addEventListener("click", function (e) {
      var seq = DEMO.sequences[state.seqId];
      if (!seq || !c._n) return;
      stop(); state.i = idxAt(e.clientX); $("#scrub").value = String(state.i); render();
    });
  }

  // ---- Brief -------------------------------------------------------------
  function renderBrief(seq) {
    // Faithful to the cascade: the fast path produces no LLM brief, so d3-decided
    // sequences show d3's quick result, not a borrowed Qwen writeup.
    if (!seq.cascade.escalated) { renderQuickBrief(seq); return; }
    var b = seq.brief, c = seq.cascade;
    var prov = "Written by " + esc(c.llm_label) + " after d3 escalated this sequence.";

    var html = "";
    html += '<div class="verdict">' +
      '<span class="presence ' + statusClass(b.presence) + '">' + esc(b.presence) + "</span>" +
      '<span class="pill ' + b.presence_confidence + '">' + esc(b.presence_confidence) + " confidence</span></div>";
    html += '<p class="prov mono">' + prov + "</p>";

    html += field("Fire / smoke", esc(b.presence_evidence));
    var ext = esc(b.extent) + (b.hot_area_percent != null
      ? ' <span class="measure mono">(' + b.hot_area_percent.toFixed(2) + "% of frame above 80 C)</span>" : "");
    html += field("Extent", ext);
    if (b.location_in_frame) html += field("Location in frame", esc(b.location_in_frame));
    html += field("Trend", "<b>" + esc(b.trend) + "</b> <span class=\"measure\">" + esc(b.trend_basis) + "</span>");
    html += field("Overall confidence", '<span class="pill ' + b.overall_confidence + '">' + esc(b.overall_confidence) + "</span>");

    html += listBlock("Observations", b.observations);
    html += listBlock("Inferences", b.inferences);
    html += listBlock("Limitations", b.limitations);
    if (b.summary) html += '<div class="summary"><div class="k">Summary</div><p id="brief-summary">' + esc(b.summary) + "</p></div>";

    $("#brief-body").innerHTML = html;
  }

  // d3-decided sequences: the fast path made the call and stopped, no Qwen brief.
  function renderQuickBrief(seq) {
    var c = seq.cascade;
    var conf = Math.round(c.decider_top_p * 100);
    var bar = Math.round((c.accept_p || 0.8) * 100);
    var secs = c.decider_seconds != null ? c.decider_seconds : c.total_seconds;
    var html = "";
    html += '<div class="verdict">' +
      '<span class="presence ' + statusClass(c.final_status) + '">' + esc(c.final_status) + "</span>" +
      '<span class="pill ' + (conf >= bar ? "high" : "medium") + '">d3 &middot; ' + conf + "%</span></div>";
    html += '<p class="prov mono">Resolved by d3. No Qwen brief for this sequence.</p>';
    html += field("Decision", "d3 called it <b>" + esc(c.decider_status) + "</b> at " + conf +
      "%, clearing the " + bar + "% confidence bar" + (secs != null ? " in " + secs + "s" : "") + ".");
    var gt = c.correct
      ? esc(seq.truth) + ' <span class="measure">&middot; d3 was right</span>'
      : '<b class="miss-note">' + esc(seq.truth) + "</b> &middot; d3 missed (it said " + esc(c.final_status) + ")";
    html += field("Ground truth", gt);
    html += '<div class="summary"><div class="k">Fast path</div><p id="brief-summary">' +
      "d3 answered the fire call straight from the frames and the cascade stopped, so Qwen 3.8 never ran and no written brief exists for this sequence. That is the cascade working as intended: the small fast model clears the easy cases, and the 27B model is only spent when d3 is unsure." +
      "</p></div>";
    $("#brief-body").innerHTML = html;
  }

  function field(k, vHtml) {
    return '<div class="field"><div class="k">' + esc(k) + '</div><div class="v">' + vHtml + "</div></div>";
  }
  function listBlock(title, items) {
    if (!items || !items.length) return "";
    return '<div class="field"><div class="k">' + esc(title) + "</div><ul class=\"brief-list\">" +
      items.map(function (x) { return "<li>" + esc(x) + "</li>"; }).join("") + "</ul></div>";
  }

  // ---- helpers -----------------------------------------------------------
  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath(); ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }
  function fmt(sec) { sec = Math.round(sec); var m = Math.floor(sec / 60), s = sec % 60; return m + ":" + (s < 10 ? "0" : "") + s; }
  function esc(str) { return String(str).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }

  // ---- wire up -----------------------------------------------------------
  function init() {
    initTheme();
    renderPicker();
    $("#flow-begin").addEventListener("click", flowBegin);
    $("#flow-next").addEventListener("click", flowNext);
    $("#flow-reset").addEventListener("click", flowReset);
    $("#play").addEventListener("click", function () { state.playing ? stop() : play(); });
    $("#scrub").addEventListener("input", function (e) { stop(); state.i = Number(e.target.value); render(); });
    ["mask", "score", "clip", "thermal"].forEach(function (key) {
      $("#tg-" + key).addEventListener("change", function (e) { toggles[key] = e.target.checked; drawFrame(DEMO.sequences[state.seqId], state.i); });
    });
    initChartHover();
    var ro = new ResizeObserver(function () { if (state.seqId) { drawFrame(DEMO.sequences[state.seqId], state.i); drawChart(); } });
    ro.observe($("#frame")); ro.observe($("#chart"));
    if (DEMO.index.sequences.length) selectSeq(DEMO.index.sequences[0].id);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
