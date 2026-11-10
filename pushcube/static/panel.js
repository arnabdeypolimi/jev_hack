// Jev's "mind": what it decided (System 1) and the futures it simulated when unsure (System 2).
const DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
const DISTS = ["short", "medium", "long"];
const AIMS = ["left", "center", "right"];
const AIM_LABEL = { left: "left ↻", center: "centre", right: "right ↺" };
const SVGNS = "http://www.w3.org/2000/svg";

const STATUS = {
  idle: ["Waiting for a round", ""],
  deciding: ["Jev is deciding…", "reading the coordinates"],
  thinking_harder: ["Not sure — simulating options", "System 2: three futures, same physics"],
  acting: ["Pushing", ""],
  judge_offline: ["Judge offline", "falling back to a simulated best move"],
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

function tracePose(g, polygons, [x, y, a = 0], s) {
  const c = Math.cos(a), sn = Math.sin(a);
  g.beginPath();
  for (const part of polygons) {
    part.forEach(([px, py], i) => {
      const wx = (x + px * c - py * sn) * s, wy = (y + px * sn + py * c) * s;
      i ? g.lineTo(wx, wy) : g.moveTo(wx, wy);
    });
    g.closePath();
  }
}

export function drawMiniBoard(canvas, { polygons, goal, start, after, size = 512 }) {
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
  if (!polygons) return;
  if (goal) {
    tracePose(g, polygons, goal, s);
    g.fillStyle = "rgba(47,143,91,0.18)";
    g.fill();
    g.strokeStyle = "#2f8f5b";
    g.stroke();
  }
  if (start) {
    tracePose(g, polygons, start, s);
    g.setLineDash([4 * dpr, 4 * dpr]);
    g.strokeStyle = "rgba(31,36,51,0.55)";
    g.stroke();
    g.setLineDash([]);
  }
  if (after) {
    tracePose(g, polygons, after, s);
    g.fillStyle = "#f0a23b";
    g.fill();
    g.strokeStyle = "#8a5513";
    g.stroke();
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
  const barRows = (names, host, label = (n) => n) => Object.fromEntries(
    names.map((d) => {
      const row = document.createElement("div");
      row.className = "dist-row";
      row.innerHTML = `<span class="dist-name">${label(d)}</span><span class="bar"><i></i></span><span class="dist-p">–</span>`;
      host.appendChild(row);
      return [d, row];
    }),
  );
  const distRows = barRows(DISTS, el(".distances"));
  const aimRows = barRows(AIMS, el(".aims"), (a) => AIM_LABEL[a]);
  let ctx = { goal: null, pose: null, polygons: null };
  let jevStatus = "unknown";
  const FALLBACK_WHY = { missing: "no Jev key", rejected: "Jev key rejected", unreachable: "TypeSafe unreachable" };

  function fillBars(rows, probs, chosen, jev) {
    for (const [name, row] of Object.entries(rows)) {
      const p = probs?.[name] ?? 0;
      row.querySelector("i").style.transform = `scaleX(${jev ? p : 0})`;
      row.querySelector(".dist-p").textContent = jev ? pct(p) : "–";
      row.classList.toggle("chosen", jev && name === chosen);
    }
  }

  function setStatus(status, option) {
    const [title, sub] = STATUS[status] ?? [status, ""];
    el(".status-title").textContent = status === "acting" && option ? `Pushing ${option.replaceAll("-", " · ")}` : title;
    el(".status-sub").textContent = sub;
    root.dataset.status = status;
  }

  function reset() {
    compass.update({}, null);
    fillBars(distRows, {}, null, false);
    fillBars(aimRows, {}, null, false);
    el(".confidence-value").textContent = "–";
    el(".confidence-bar i").style.transform = "scaleX(0)";
    el(".source").textContent = "";
    el(".futures").replaceChildren();
    root.classList.remove("thinking");
    setStatus("idle");
  }

  return {
    setContext(goal, pose, polygons) { ctx = { goal, pose, polygons }; },
    setJevStatus(status) { jevStatus = status; },
    setStatus,
    reset,
    showDecision(msg) {
      const [dir, dist, aim = "center"] = msg.option.split("-");
      const jev = msg.source === "jev";
      compass.update(msg.direction_probs, jev ? dir : null);
      fillBars(distRows, msg.distance_probs, dist, jev);
      fillBars(aimRows, msg.aim_probs, aim, jev);
      el(".confidence-value").textContent = msg.source === "jev" ? pct(msg.confidence) : "–";
      el(".confidence-bar i").style.transform = `scaleX(${msg.source === "jev" ? msg.confidence : 0})`;
      el(".source").textContent = msg.source === "jev" ? `Jev · ${Math.round(msg.latency_ms)} ms` : `fallback: simulated best move${FALLBACK_WHY[jevStatus] ? ` (${FALLBACK_WHY[jevStatus]})` : ""}`;
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
        card.innerHTML = `<canvas></canvas><figcaption><b>${o.option.replaceAll("-", " · ")}</b><span>overlap ${pct(o.overlap_after)}</span><span class="pick-p"></span></figcaption>`;
        host.appendChild(card);
        drawMiniBoard(card.querySelector("canvas"), {
          polygons: ctx.polygons, goal: ctx.goal, start: ctx.pose, after: [...o.cube_after, o.angle_after ?? 0],
        });
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
