/* Replay console logic. Reads window.DEMO (see data.js). The page never calls a
   model: it replays per-sequence JSON the pipeline exported. Per-frame radius,
   score, and centroid are shared measurements; each model contributes its own
   brief (trend, direction, confidence), and switching models swaps that brief
   and the thermal-agreement badge. Extent is the measured radius at the current
   frame, so it fills in as the replay plays. */
(function () {
  "use strict";

  var DEMO = window.DEMO || { index: { sequences: [] }, sequences: {} };
  var state = { seqId: null, modelId: null, i: 0, tp: 0, playing: false, raf: null, lastTs: null };
  var STEP_MS = 950;  // dwell per real (thermal) frame
  var toggles = { mask: true, score: true, track: true, heading: true };
  var $ = function (sel) { return document.querySelector(sel); };

  // Frame-relative direction phrases to unit vectors (x right, y down).
  var DIR_VECTORS = {
    "toward the right of frame": [1, 0],
    "toward upper right of frame": [0.707, -0.707],
    "toward top of frame": [0, -1],
    "toward upper left of frame": [-0.707, -0.707],
    "toward the left of frame": [-1, 0],
    "toward lower left of frame": [-0.707, 0.707],
    "toward bottom of frame": [0, 1],
    "toward lower right of frame": [0.707, 0.707]
  };

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

  // ---- Picker + model bar ------------------------------------------------
  function renderPicker() {
    var list = $("#seq-list");
    list.innerHTML = "";
    DEMO.index.sequences.forEach(function (s) {
      var li = document.createElement("li");
      var b = document.createElement("button");
      b.className = "seq-item";
      b.setAttribute("data-id", s.id);
      b.setAttribute("aria-current", String(s.id === state.seqId));
      b.innerHTML =
        '<span class="name">' + esc(s.display_name) + "</span>" +
        '<span class="meta"><span class="tag">' + esc(s.case.replace(/_/g, " ")) +
        "</span><span class=\"tag\">" + s.models.length + " models</span></span>";
      b.addEventListener("click", function () { selectSeq(s.id); });
      li.appendChild(b);
      list.appendChild(li);
    });
  }

  function renderModelBar(seq) {
    var seg = $("#model-seg");
    seg.innerHTML = "";
    Object.keys(seq.models).forEach(function (mid) {
      var m = seq.models[mid];
      var b = document.createElement("button");
      b.type = "button";
      b.className = "seg-btn";
      b.setAttribute("data-mid", mid);
      b.setAttribute("aria-pressed", String(mid === state.modelId));
      b.innerHTML = esc(m.label) + '<span class="fam">' + esc(m.family) + "</span>";
      b.addEventListener("click", function () { pickModel(seq, mid); });
      seg.appendChild(b);
    });
  }

  function pickModel(seq, mid) {
    state.modelId = mid;
    document.querySelectorAll(".seg-btn").forEach(function (el) {
      el.setAttribute("aria-pressed", String(el.getAttribute("data-mid") === mid));
    });
    updateBrief(seq, state.i);
    drawFrame(seq, state.i);
  }

  function selectSeq(id) {
    state.seqId = id;
    state.i = 0;
    stop();
    var seq = DEMO.sequences[id];
    state.modelId = seq.default_model || Object.keys(seq.models)[0];
    document.querySelectorAll(".seq-item").forEach(function (el) {
      el.setAttribute("aria-current", String(el.getAttribute("data-id") === id));
    });
    renderModelBar(seq);
    var scrub = $("#scrub");
    scrub.max = String(seq.frames.length - 1);
    scrub.value = "0";
    render();
  }

  // ---- Transport ---------------------------------------------------------
  // The thermal truth is sampled ~1 min apart, so it snaps frame to frame
  // (state.i). The model path is interpolated, so it glides on a continuous
  // clock (state.tp). Playback advances tp smoothly and steps i at boundaries.
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

  // ---- Brief -------------------------------------------------------------
  function updateBrief(seq, i) {
    var single = $("#brief-single"), compare = $("#brief-compare"), badge = $("#badge");
    if (state.modelId === "__all__") {
      single.hidden = true; compare.hidden = false; badge.hidden = true;
      renderCompare(seq);
      return;
    }
    single.hidden = false; compare.hidden = true; badge.hidden = false;

    var trx = seq.thermal_reference;
    $("#ref-line").innerHTML = '<span class="lbl">Thermal reference</span>' +
      (trx.fire_present
        ? "<b>" + esc(trx.trend) + "</b> &middot; <b>" + esc(dirLabel(trx.direction, trx.angle_deg)) + "</b>"
        : "<b>no fire</b>");

    var m = seq.models[state.modelId];
    var mb = m.brief;
    var f = seq.frames, cur = f[i];
    var isFire = seq.case !== "no_fire";

    setField("presence", mb.presence, true);

    // Extent showcases the model's final size (its claim) next to the thermal
    // truth at the current frame, so the gap between them is visible. Trend,
    // direction, and confidence are the model's call for the whole sequence.
    var ext;
    if (!isFire) {
      ext = "Not applicable";
    } else if (m.reference) {
      ext = Math.round(cur.radius_frac * 100) + "% of frame (thermal, now)";
    } else {
      var ms = m.size_estimate_frac;
      var truthNow = Math.round(cur.radius_frac * 100);
      ext = (ms != null ? "~" + Math.round(ms * 100) + "% model final" : "model: not estimated") +
            "  ·  " + truthNow + "% thermal now";
    }
    setField("extent", ext, true);
    setField("direction", dirLabel(mb.direction.value, mb.direction.angle_deg), true);
    setField("trend", mb.trend, true);

    $("#f-confidence").classList.remove("pending");
    $("#f-confidence .v").innerHTML = '<span class="pill ' + mb.confidence.split(" ")[0] + '">' + esc(mb.confidence) + "</span>";

    $("#f-limits").classList.remove("pending");
    $("#f-limits .v").innerHTML = '<ul class="limits">' + mb.limitations.map(function (x) {
      return "<li>" + esc(x) + "</li>";
    }).join("") + "</ul>";

    // Thermal badge: this model vs the thermal reference.
    var badge = $("#badge");
    if (m.reference) {
      badge.className = "thermal-badge agree";
      badge.innerHTML = '<span class="icon">=</span><span><span class="t-title">Thermal reference</span><br>' +
        '<span class="t-note">The deterministic thermal baseline. It defines the truth, so it is not scored against itself.</span></span>';
    } else {
      var tr = seq.thermal_reference, ag = m.agreement;
      var agrees = ag.presence && ag.trend && ag.direction;
      badge.className = "thermal-badge " + (agrees ? "agree" : "disagree");
      var note;
      if (!tr.available) {
        note = "No thermal camera on this flight. RGB-only brief.";
      } else if (agrees) {
        note = "Thermal reference confirms this model's presence, trend, and direction.";
      } else {
        var d = [];
        if (!ag.trend) d.push("trend (thermal: " + tr.trend + ")");
        if (!ag.direction) d.push("direction (thermal: " + tr.direction + ")");
        if (!ag.presence) d.push("presence");
        note = "Disagrees with thermal on " + d.join(" and ") + ".";
      }
      badge.innerHTML =
        '<span class="icon">' + (agrees ? "=" : "!") + "</span>" +
        '<span><span class="t-title">' + (agrees ? "Thermal agrees" : "Thermal disagrees") +
        '</span><br><span class="t-note">' + esc(note) + "</span></span>";
    }
  }

  function setField(key, value, revealed) {
    var el = $("#f-" + key);
    el.classList.toggle("pending", !revealed);
    el.querySelector(".v").textContent = revealed ? value : "pending";
  }

  function renderCompare(seq) {
    var tr = seq.thermal_reference;
    var ref = tr.fire_present
      ? "<b>" + esc(tr.trend) + "</b> &middot; <b>" + esc(dirCompact(tr.direction, tr.angle_deg)) + "</b>"
      : "<b>no fire</b>";
    var rows = Object.keys(seq.models).map(function (mid) {
      var m = seq.models[mid], ag = m.agreement, b = m.brief;
      var hits = (ag.presence ? 1 : 0) + (ag.trend ? 1 : 0) + (ag.direction ? 1 : 0);
      var ok = hits === 3;
      var matchHtml = m.reference
        ? '<span class="match ref">reference</span>'
        : '<span class="match ' + (ok ? "ok" : "bad") + '">' + hits + "/3</span>";
      return '<li class="cmp-row">' +
        '<div class="cmp-top"><span class="cmp-name">' + esc(m.label) +
          "</span>" + matchHtml + "</div>" +
        '<div class="cmp-fam">' + esc(m.family) + "</div>" +
        '<div class="cmp-vals">' + esc(b.trend) + ' <span class="mut">&middot;</span> ' +
          esc(dirCompact(b.direction.value, b.direction.angle_deg)) + ' <span class="mut">&middot;</span> ' + esc(b.confidence) + "</div>" +
        "</li>";
    }).join("");
    $("#brief-compare").innerHTML =
      '<div class="cmp-ref"><span class="lbl">Thermal reference (the yardstick)</span>' + ref + "</div>" +
      '<ul class="cmp-list">' + rows + "</ul>";
  }

  // ---- Frame canvas (placeholder render from the numbers) ----------------
  function render() {
    var seq = DEMO.sequences[state.seqId];
    var n = seq.frames.length;
    state.tp = n > 1 ? state.i / (n - 1) : 1;  // snap model progress to frame when not playing
    renderDiscrete(seq);
    drawDynamic(seq);
  }

  // Discrete, per-frame updates (truth-rate): stamps, counter, brief.
  function renderDiscrete(seq) {
    var cur = seq.frames[state.i];
    $("#stamp-l").textContent = seq.source.flight;
    var r = cur.fire ? "  r " + Math.round(cur.radius_frac * 100) + "%" : "";
    $("#stamp-r").innerHTML = "T+" + fmt(cur.t_sec) + "<br>" + cur.clock + r;
    $("#counter").textContent = "Frame " + (state.i + 1) + " / " + seq.frames.length;
    updateBrief(seq, state.i);
  }

  // Canvas redraw (runs every animation frame during playback).
  function drawDynamic(seq) {
    drawFrame(seq, state.i);
    drawChart();
  }

  function sizeCanvas(c) {
    var dpr = window.devicePixelRatio || 1;
    var r = c.getBoundingClientRect();
    c.width = Math.max(1, Math.round(r.width * dpr));
    c.height = Math.max(1, Math.round(r.height * dpr));
    var ctx = c.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: r.width, h: r.height };
  }

  function drawFrame(seq, i) {
    var c = $("#frame");
    var s = sizeCanvas(c), ctx = s.ctx, W = s.w, H = s.h;
    var f = seq.frames, cur = f[i];

    var g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, "#2b2f25"); g.addColorStop(1, "#191b15");
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    ctx.globalAlpha = 0.5;
    for (var k = 0; k < 40; k++) {
      var rx = Math.abs((Math.sin(k * 12.9898 + seq.sequence_id.length) * 43758.5453) % 1);
      var ry = Math.abs((Math.sin(k * 78.233 + seq.sequence_id.length) * 43758.5453) % 1);
      ctx.fillStyle = k % 2 ? "#333a2b" : "#22271c";
      ctx.beginPath(); ctx.arc(rx * W, ry * H, 6 + rx * 16, 0, 7); ctx.fill();
    }
    ctx.globalAlpha = 1;

    var fire = cur.fire && cur.radius_frac > 0 && cur.centroid;
    if (fire) {
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

      if (toggles.track) {
        ctx.strokeStyle = "rgba(120,220,255,0.95)"; ctx.lineWidth = 2;
        ctx.beginPath();
        for (var j = 0; j <= i; j++) {
          if (!f[j].centroid) continue;
          var px = f[j].centroid[0] * W, py = f[j].centroid[1] * H;
          if (j === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.stroke();
        ctx.fillStyle = "rgba(120,220,255,1)";
        ctx.beginPath(); ctx.arc(cx, cy, 4, 0, 7); ctx.fill();
      }
    }

    if (seq.case === "smoke_obscured" && i >= 3) {
      var amt = Math.min(0.72, 0.18 * (i - 2));
      var sg = ctx.createLinearGradient(0, 0, W, H);
      sg.addColorStop(0, "rgba(180,178,172," + amt + ")");
      sg.addColorStop(1, "rgba(120,120,120," + (amt * 0.6).toFixed(2) + ")");
      ctx.fillStyle = sg; ctx.fillRect(0, 0, W, H);
    }

    drawHeading(ctx, seq, W, H);

    if (toggles.score) {
      var label = cur.detection_score.toFixed(2);
      ctx.font = "600 13px IBM Plex Mono, monospace";
      var tw = ctx.measureText("score " + label).width + 16;
      ctx.fillStyle = "rgba(0,0,0,0.55)";
      roundRect(ctx, 12, H - 34, tw, 22, 5); ctx.fill();
      ctx.fillStyle = cur.fire ? "#ffd27a" : "#9fe8c0";
      ctx.textBaseline = "middle"; ctx.fillText("score " + label, 20, H - 22);
    }
  }

  // The selected model's claim, drawn in white at the fire's start point: a
  // dashed circle for its predicted size and an arrow for its claimed heading.
  // Contrast the circle with the orange truth blob and the arrow with the
  // weaving cyan truth path.
  function drawHeading(ctx, seq, W, H) {
    if (!toggles.heading || state.modelId === "__all__") return;
    var m = seq.models[state.modelId];
    if (!m || m.reference) return;  // the thermal baseline IS the truth; no estimate to draw
    var start = seq.frames[0].centroid;
    if (!start) return;
    var p = state.tp;                              // continuous, so the model glides
    var sx = start[0] * W, sy = start[1] * H;
    var mind = Math.min(W, H);

    var ang = m.brief.direction.angle_deg;
    var vec = ang != null
      ? [Math.sin(ang * Math.PI / 180), -Math.cos(ang * Math.PI / 180)]
      : DIR_VECTORS[m.brief.direction.value];
    var len = 0.3 * mind;
    var tx = vec ? sx + vec[0] * len : sx, ty = vec ? sy + vec[1] * len : sy;

    ctx.save();
    ctx.shadowColor = "rgba(0,0,0,0.75)"; ctx.shadowBlur = 4;
    ctx.strokeStyle = "#ffffff"; ctx.fillStyle = "#ffffff"; ctx.lineCap = "round";

    // Full predicted heading arrow (the path the blob travels).
    if (vec) {
      ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(tx, ty); ctx.stroke();
      var a = Math.atan2(ty - sy, tx - sx), hl = 13;
      ctx.beginPath();
      ctx.moveTo(tx, ty);
      ctx.lineTo(tx - hl * Math.cos(a - 0.42), ty - hl * Math.sin(a - 0.42));
      ctx.lineTo(tx - hl * Math.cos(a + 0.42), ty - hl * Math.sin(a + 0.42));
      ctx.closePath(); ctx.fill();
      ctx.font = "600 11px IBM Plex Mono, monospace"; ctx.textBaseline = "middle";
      ctx.textAlign = vec[0] >= 0 ? "left" : "right";
      ctx.fillText("model claim", tx + (vec[0] >= 0 ? 8 : -8), ty - 11);
    }

    // Predicted blob: glides start -> tip and grows 0 -> final size, with a
    // dot at its center riding along the heading line.
    if (m.size_estimate_frac != null) {
      var bx = sx + (tx - sx) * p, by = sy + (ty - sy) * p;
      var br = m.size_estimate_frac * mind * 1.6 * p;
      ctx.lineWidth = 2; ctx.setLineDash([6, 5]);
      ctx.beginPath(); ctx.arc(bx, by, br, 0, 7); ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath(); ctx.arc(bx, by, 4, 0, 7); ctx.fill();
    }

    ctx.beginPath(); ctx.arc(sx, sy, 3, 0, 7); ctx.fill();
    ctx.restore();
  }

  // ---- Chart: fire radius % and detection score % over the sequence ------
  function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

  function drawChart() {
    var seq = DEMO.sequences[state.seqId];
    if (!seq) return;
    var c = $("#chart");
    var s = sizeCanvas(c), ctx = s.ctx, W = s.w, H = s.h;
    var f = seq.frames;
    var padL = 34, padR = 66, padT = 10, padB = 22;
    var plotW = W - padL - padR, plotH = H - padT - padB;
    var n = f.length;
    var x = function (idx) { return padL + (n === 1 ? 0 : (idx / (n - 1)) * plotW); };
    var y = function (pct) { return padT + (1 - pct / 100) * plotH; };

    var grid = cssVar("--grid"), muted = cssVar("--muted"), fg = cssVar("--fg");
    var cRad = cssVar("--series-area") || "#d97706", cScore = cssVar("--series-score") || "#2596d4";

    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = grid; ctx.lineWidth = 1;
    ctx.font = "10px IBM Plex Mono, monospace"; ctx.fillStyle = muted; ctx.textBaseline = "middle";
    [0, 25, 50, 75, 100].forEach(function (v) {
      var yy = y(v);
      ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(W - padR, yy); ctx.stroke();
      ctx.textAlign = "right"; ctx.fillText(v + "%", padL - 6, yy);
    });

    // Soft fill under the fire-size line so it reads even when low.
    ctx.beginPath();
    f.forEach(function (fr, idx) { var xx = x(idx), yy = y(fr.radius_frac * 100); if (idx === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy); });
    ctx.lineTo(x(n - 1), y(0)); ctx.lineTo(x(0), y(0)); ctx.closePath();
    ctx.globalAlpha = 0.14; ctx.fillStyle = cRad; ctx.fill(); ctx.globalAlpha = 1;

    function line(key, color) {
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
      f.forEach(function (fr, idx) {
        var pct = key === "radius" ? fr.radius_frac * 100 : fr.detection_score * 100;
        var xx = x(idx), yy = y(pct);
        if (idx === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
      });
      ctx.stroke();
      var last = f[n - 1];
      var lp = key === "radius" ? last.radius_frac * 100 : last.detection_score * 100;
      ctx.fillStyle = color; ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillText(key === "radius" ? "Size" : "Conf", W - padR + 6, y(lp));
    }
    line("score", cScore);
    line("radius", cRad);

    var mx = x(state.i);
    ctx.strokeStyle = fg; ctx.globalAlpha = 0.5; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(mx, padT); ctx.lineTo(mx, H - padB); ctx.stroke();
    ctx.setLineDash([]); ctx.globalAlpha = 1;
    [["radius", cRad], ["score", cScore]].forEach(function (p) {
      var fr = f[state.i];
      var pct = p[0] === "radius" ? fr.radius_frac * 100 : fr.detection_score * 100;
      ctx.fillStyle = p[1]; ctx.beginPath(); ctx.arc(mx, y(pct), 4, 0, 7); ctx.fill();
    });

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
      tip.innerHTML = fr.clock + " · r " + (fr.radius_frac * 100).toFixed(0) + "% · score " + fr.detection_score.toFixed(2);
    });
    box.addEventListener("mouseleave", function () { tip.hidden = true; });
    box.addEventListener("click", function (e) {
      var seq = DEMO.sequences[state.seqId];
      if (!seq || !c._n) return;
      stop(); state.i = idxAt(e.clientX); $("#scrub").value = String(state.i); render();
    });
  }

  // ---- helpers -----------------------------------------------------------
  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath(); ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }
  function fmt(sec) { var m = Math.floor(sec / 60), s = sec % 60; return m + ":" + (s < 10 ? "0" : "") + s; }
  function shortDir(v) { return String(v).replace(/^toward /, "").replace(/ of frame$/, ""); }
  function dirLabel(value, angle) { return angle != null ? angle + "° frame-relative (" + shortDir(value) + ")" : shortDir(value); }
  function dirCompact(value, angle) { return angle != null ? angle + "° (" + shortDir(value) + ")" : shortDir(value); }
  function esc(str) { return String(str).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }

  // ---- wire up -----------------------------------------------------------
  function init() {
    initTheme();
    renderPicker();
    $("#play").addEventListener("click", function () { state.playing ? stop() : play(); });
    $("#scrub").addEventListener("input", function (e) { stop(); state.i = Number(e.target.value); render(); });
    ["mask", "score", "track", "heading"].forEach(function (key) {
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
