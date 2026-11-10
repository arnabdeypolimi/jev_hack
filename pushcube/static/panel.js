// Jev's "mind": what it decided (System 1) and the futures it simulated when unsure (System 2).
const DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
const DISTS = ["short", "medium", "long"];
const SVGNS = "http://www.w3.org/2000/svg";

const STATUS = {
  idle: ["Waiting for a round", ""],
  deciding: ["Jev is deciding…", "reading the coordinates"],
  thinking_harder: ["Not sure — simulating options", "System 2: three futures, same physics"],
  acting: ["Pushing", ""],
  judge_offline: ["Judge offline", "falling back to a geometric rule"],
  done: ["Round over for Jev", ""],
};

const pct = (p) => `${Math.round((p ?? 0) * 100)}%`;

function wedgePath(i, r0, r1) {
  const a0 = ((i * 45 - 22.5 - 90) * Math.PI) / 180;
  const a1 = ((i * 45 + 22.5 - 90) * Math.PI) / 180;
  const p = (r, a) => `${(r * Math.cos(a)).toFixed(2)} ${(r * Math.sin(a)).toFixed(2)}`;
  return `M ${p(r0, a0)} L ${p(r1, a0)} A ${r1} ${r1} 0 0 1 ${p(r1, a1)} L ${p(r0, a1)} A ${r0} ${r0} 0 0 0 ${p(r0, a0)} Z`;
}

function buildCompass(host) {
  const svg = document.createElementNS(SVGNS, "svg");
  svg.setAttribute("viewBox", "-120 -120 240 240");
  svg.classList.add("compass");
  const ring = document.createElementNS(SVGNS, "circle");
  ring.setAttribute("r", "92");
  ring.setAttribute("class", "compass-ring");
  svg.appendChild(ring);
  const wedges = DIRS.map((d, i) => {
    const path = document.createElementNS(SVGNS, "path");
    path.setAttribute("class", "wedge");
    path.setAttribute("d", wedgePath(i, 20, 30));
    svg.appendChild(path);
    const a = ((i * 45 - 90) * Math.PI) / 180;
    const label = document.createElementNS(SVGNS, "text");
    label.setAttribute("x", (108 * Math.cos(a)).toFixed(1));
    label.setAttribute("y", (108 * Math.sin(a) + 5).toFixed(1));
    label.setAttribute("class", "compass-label");
    label.textContent = d;
    svg.appendChild(label);
    return { d, i, path, label };
  });
  const hub = document.createElementNS(SVGNS, "rect");
  Object.entries({ x: -12, y: -12, width: 24, height: 24, rx: 4, class: "compass-hub" }).forEach(([k, v]) => hub.setAttribute(k, v));
  svg.appendChild(hub);
  host.appendChild(svg);
  return {
    update(probs, chosen) {
      for (const w of wedges) {
        const p = probs?.[w.d] ?? 0;
        w.path.setAttribute("d", wedgePath(w.i, 20, 30 + 62 * Math.sqrt(p)));
        w.path.classList.toggle("chosen", w.d === chosen);
        w.label.classList.toggle("chosen", w.d === chosen);
      }
    },
  };
}

export function drawMiniBoard(canvas, { goal, start, after, size = 512, cubeSize = 60 }) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = canvas.clientWidth || 120;
  canvas.width = canvas.height = Math.round(w * dpr);
  const g = canvas.getContext("2d");
  const s = canvas.width / size;
  g.fillStyle = "#f7f3ea";
  g.fillRect(0, 0, canvas.width, canvas.height);
  g.strokeStyle = "rgba(31,36,51,0.5)";
  g.lineWidth = 2 * dpr;
  g.strokeRect(1, 1, canvas.width - 2, canvas.height - 2);
  const box = ([x, y]) => [(x - cubeSize / 2) * s, (y - cubeSize / 2) * s, cubeSize * s, cubeSize * s];
  if (goal) {
    g.fillStyle = "rgba(47,143,91,0.18)";
    g.fillRect(...box(goal));
    g.strokeStyle = "#2f8f5b";
    g.strokeRect(...box(goal));
  }
  if (start) {
    g.setLineDash([4 * dpr, 4 * dpr]);
    g.strokeStyle = "rgba(31,36,51,0.55)";
    g.strokeRect(...box(start));
    g.setLineDash([]);
  }
  if (after) {
    g.fillStyle = "#f0a23b";
    g.fillRect(...box(after));
    g.strokeStyle = "#8a5513";
    g.strokeRect(...box(after));
    if (start) {
      g.strokeStyle = "#1f2433";
      g.beginPath();
      g.moveTo(start[0] * s, start[1] * s);
      g.lineTo(after[0] * s, after[1] * s);
      g.stroke();
    }
  }
}

export function createPanel(root) {
  const el = (sel) => root.querySelector(sel);
  const compass = buildCompass(el(".compass-host"));
  const distRows = Object.fromEntries(
    DISTS.map((d) => {
      const row = document.createElement("div");
      row.className = "dist-row";
      row.innerHTML = `<span class="dist-name">${d}</span><span class="bar"><i></i></span><span class="dist-p">–</span>`;
      el(".distances").appendChild(row);
      return [d, row];
    }),
  );
  let ctx = { goal: null, cube: null };

  function setStatus(status, option) {
    const [title, sub] = STATUS[status] ?? [status, ""];
    el(".status-title").textContent = status === "acting" && option ? `Pushing ${option.replace("-", " · ")}` : title;
    el(".status-sub").textContent = sub;
    root.dataset.status = status;
  }

  function reset() {
    compass.update({}, null);
    for (const row of Object.values(distRows)) {
      row.querySelector("i").style.transform = "scaleX(0)";
      row.querySelector(".dist-p").textContent = "–";
      row.classList.remove("chosen");
    }
    el(".confidence-value").textContent = "–";
    el(".confidence-bar i").style.transform = "scaleX(0)";
    el(".source").textContent = "";
    el(".futures").replaceChildren();
    root.classList.remove("thinking");
    setStatus("idle");
  }

  return {
    setContext(goal, cube) { ctx = { goal, cube }; },
    setStatus,
    reset,
    showDecision(msg) {
      const [dir, dist] = msg.option.split("-");
      compass.update(msg.direction_probs, msg.source === "jev" ? dir : null);
      for (const [d, row] of Object.entries(distRows)) {
        const p = msg.distance_probs?.[d] ?? 0;
        row.querySelector("i").style.transform = `scaleX(${p})`;
        row.querySelector(".dist-p").textContent = msg.source === "jev" ? pct(p) : "–";
        row.classList.toggle("chosen", d === dist && msg.source === "jev");
      }
      el(".confidence-value").textContent = msg.source === "jev" ? pct(msg.confidence) : "–";
      el(".confidence-bar i").style.transform = `scaleX(${msg.source === "jev" ? msg.confidence : 0})`;
      el(".source").textContent = msg.source === "jev" ? `Jev · ${Math.round(msg.latency_ms)} ms` : "geometric fallback";
      el(".source").classList.toggle("offline", msg.source !== "jev");
      el(".futures").replaceChildren();
      root.classList.remove("thinking");
    },
    showImagination(msg) {
      root.classList.add("thinking");
      const host = el(".futures");
      host.replaceChildren();
      for (const o of msg.outcomes) {
        const card = document.createElement("figure");
        card.className = "future";
        card.dataset.option = o.option;
        card.innerHTML = `<canvas></canvas><figcaption><b>${o.option.replace("-", " · ")}</b><span>overlap ${pct(o.overlap_after)}</span><span class="pick-p"></span></figcaption>`;
        host.appendChild(card);
        drawMiniBoard(card.querySelector("canvas"), { goal: ctx.goal, start: ctx.cube, after: o.cube_after });
      }
    },
    showPick(msg) {
      for (const card of root.querySelectorAll(".future")) {
        const p = msg.probs?.[card.dataset.option];
        card.querySelector(".pick-p").textContent = p != null ? `Jev ${pct(p)}` : "";
        card.classList.toggle("picked", card.dataset.option === msg.option);
      }
    },
  };
}
