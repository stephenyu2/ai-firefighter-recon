(function () {
  var tip = document.getElementById("tip");
  if (!tip) return;
  function fill(el) {
    var parts = (el.getAttribute("data-tip") || "").split("|");
    tip.replaceChildren();
    var s = document.createElement("strong");
    s.textContent = parts[0];
    tip.appendChild(s);
    for (var i = 1; i < parts.length; i++) {
      var d = document.createElement("div");
      d.textContent = parts[i];
      tip.appendChild(d);
    }
    tip.hidden = false;
  }
  function place(x, y) {
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var nx = x + 14, ny = y + 14;
    if (nx + w > window.innerWidth - 8) nx = x - w - 14;
    if (ny + h > window.innerHeight - 8) ny = y - h - 14;
    tip.style.left = Math.max(8, nx) + "px";
    tip.style.top = Math.max(8, ny) + "px";
  }
  document.querySelectorAll("[data-tip]").forEach(function (el) {
    el.addEventListener("pointermove", function (e) { fill(el); place(e.clientX, e.clientY); });
    el.addEventListener("pointerleave", function () { tip.hidden = true; });
    el.addEventListener("focus", function () {
      fill(el);
      var r = el.getBoundingClientRect();
      place(r.left + r.width / 2, r.top);
    });
    el.addEventListener("blur", function () { tip.hidden = true; });
  });
})();
