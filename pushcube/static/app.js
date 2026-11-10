import { createBoard3D } from "./board3d.js";
import { createPanel } from "./panel.js";

const $ = (sel, root = document) => root.querySelector(sel);
const sides = { human: $(".side.you"), ai: $(".side.jev") };
const panel = createPanel($(".mind"));

const app = {
  ws: null,
  boards: {},
  round: null,
  aiPose: null,
  shape: null,
  timerEnds: null,
  view: "tilt",
};

function send(msg) {
  if (app.ws?.readyState === WebSocket.OPEN) app.ws.send(JSON.stringify(msg));
}

function ensureBoards(shape) {
  if (app.boards.human) return;
  app.boards.human = createBoard3D($("#board-human"), shape, { accent: 0x2f5fa8 });
  app.boards.ai = createBoard3D($("#board-ai"), shape, { accent: 0xd9472b });
}

function applyState(msg) {
  const board = app.boards[msg.board];
  if (!board) return;
  board.setState(msg.pusher, msg.cube, msg.angle ?? 0);
  $(".pushes", sides[msg.board]).textContent = msg.pushes;
  $(".overlap", sides[msg.board]).textContent = `${Math.round(msg.overlap * 100)}%`;
  if (msg.board === "ai") app.aiPose = [...msg.cube, msg.angle ?? 0];
}

function applyRound(r) {
  const fresh = !app.round || app.round.id !== r.id;
  app.round = r;
  const newShape = r.shape && r.shape !== app.shape;
  if (newShape) app.shape = r.shape;
  for (const b of Object.values(app.boards)) {
    if (newShape) b.setBlock(r.polygons, r.outline);
    b.setGoal(r.goal);
    if (fresh) b.snap();
  }
  if (fresh) {
    panel.reset();
    app.boards.ai?.setGhosts([]);
  }
  app.timerEnds = r.phase === "playing" ? (fresh || !app.timerEnds ? performance.now() + r.round_seconds * 1000 : app.timerEnds) : null;
  if (r.phase !== "playing") $(".timer").textContent = r.phase === "finished" ? "0" : "–";
  document.body.dataset.phase = r.phase;
}

const handlers = {
  snapshot(m) {
    const first = !app.boards.human;
    ensureBoards(m.shape);
    document.documentElement.style.setProperty("--tau", `${(m.tau ?? 0.5) * 100}%`);
    applyRound(m.round);
    applyState(m.boards.human);
    applyState(m.boards.ai);
    app.replay = m.replay;
    showJevStatus(m.jev_status);
    if (m.last_decision) panel.showDecision(m.last_decision);
    if (first) {
      bindBoard("human");
      bindBoard("ai");
      setView(app.view);
    }
    showResults(m.round);
    syncControls();
  },
  round(r) {
    applyRound(r);
    showResults(r);
    syncControls();
  },
  state: applyState,
  ai_status(m) {
    panel.setStatus(m.status, m.option);
    if (m.status === "acting" || m.status === "deciding") app.boards.ai?.setGhosts([]);
  },
  decision(m) {
    if (m.jev_status) showJevStatus(m.jev_status);
    panel.setContext(app.round?.goal, app.aiPose, app.round?.polygons);
    panel.showDecision(m);
  },
  imagination(m) {
    panel.setContext(app.round?.goal, app.aiPose, app.round?.polygons);
    panel.showImagination(m);
    app.boards.ai?.setGhosts(m.outcomes.map((o) => ({ xy: o.cube_after, angle: o.angle_after ?? 0, weight: 0.3 })));
  },
  pick(m) {
    panel.showPick(m);
    app.boards.ai?.setGhosts([]);
  },
};

const JEV_STATUS_TEXT = {
  missing: "No Jev key in .env — Jev's board plays a fallback",
  rejected: "TypeSafe rejected the Jev key (401) — Jev's board plays a fallback",
  unreachable: "Can't reach TypeSafe — Jev's board plays a fallback",
};

function showJevStatus(status) {
  const badges = [];
  if (app.replay) badges.push(`<span class="badge">replay</span>`);
  if (JEV_STATUS_TEXT[status]) badges.push(`<span class="badge warn">${JEV_STATUS_TEXT[status]}</span>`);
  $(".badges").innerHTML = badges.join("");
  panel.setJevStatus(status);
}

function connect(delay = 500) {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  app.ws = ws;
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    handlers[msg.type]?.(msg);
  };
  ws.onopen = () => { delay = 500; };
  ws.onclose = () => setTimeout(() => connect(Math.min(delay * 2, 5000)), delay);
}

function tickTimer() {
  if (app.timerEnds) {
    const left = Math.max(0, Math.ceil((app.timerEnds - performance.now()) / 1000));
    $(".timer").textContent = left;
    $(".timer").classList.toggle("low", left <= 10);
  }
  requestAnimationFrame(tickTimer);
}

// ---- challenge picker -----------------------------------------------------------------------
async function loadChallenges() {
  const list = await (await fetch("/challenges")).json();
  const nav = $(".challenges");
  const tiers = {};
  for (const c of list) (tiers[c.tier] ??= []).push(c);
  nav.replaceChildren(...Object.entries(tiers).map(([tier, items]) => {
    const group = document.createElement("div");
    group.className = "tier";
    group.innerHTML = `<span class="tier-name">${tier}</span>`;
    for (const c of items) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = c.name;
      b.dataset.id = c.id;
      b.addEventListener("click", () => send({ type: "select", challenge: c.id }));
      group.appendChild(b);
    }
    return group;
  }));
  syncControls();
}

function syncControls() {
  const playing = app.round?.phase === "playing";
  for (const b of document.querySelectorAll(".challenges button")) {
    b.setAttribute("aria-pressed", String(b.dataset.id === app.round?.challenge));
    b.disabled = playing;
  }
  $(".start").textContent = playing ? "Restart" : "Start round";
  $(".stop").hidden = !playing;
}

$(".stop").addEventListener("click", () => send({ type: "stop" }));

$(".start").addEventListener("click", () => {
  const r = app.round;
  if (r?.challenge) send({ type: "start", challenge: r.challenge });
  else send({ type: "start", goal: r?.goal, start: r?.start });
});

// ---- results ---------------------------------------------------------------------------------
function showResults(r) {
  const banner = $(".banner");
  if (r.phase !== "finished" || !r.results) {
    banner.hidden = true;
    return;
  }
  const line = (who, s) => `<p><b>${who}</b> · ${s.pushes} pushes · ${Math.round(s.best_overlap * 100)}% ${s.reached ? "✓" : ""}</p>`;
  const title = { human: `<span class="winner-you">You win</span>`, ai: `<span class="winner-jev">Jev wins</span>`, tie: "It's a tie" }[r.winner];
  banner.innerHTML = `<h2>${title}</h2>${line("You", r.results.human)}${line("Jev", r.results.ai)}`;
  banner.hidden = false;
}

// ---- human pusher and goal dragging -------------------------------------------------------------
const pointer = { down: false, lastSent: 0, dragging: null };

function sendMouse(point, down, force = false) {
  const now = performance.now();
  if (!force && now - pointer.lastSent < 33) return;
  pointer.lastSent = now;
  if (point) send({ type: "mouse", x: point[0], y: point[1], down });
  else send({ type: "mouse", x: 0, y: 0, down: false });
}

function nearGoal(point) {
  const g = app.round?.goal;
  return g && point && Math.hypot(point[0] - g[0], point[1] - g[1]) <= 55;
}

function bindBoard(name) {
  const board = app.boards[name];
  const canvas = board.canvas;
  canvas.addEventListener("pointerdown", (e) => {
    const p = board.pickBoardPoint(e);
    if (app.round?.phase === "playing") {
      if (name !== "human" || !p) return;
      pointer.down = true;
      canvas.setPointerCapture(e.pointerId);
      sendMouse(p, true, true);
    } else if (nearGoal(p)) {
      pointer.dragging = { board, offset: [app.round.goal[0] - p[0], app.round.goal[1] - p[1]] };
      canvas.setPointerCapture(e.pointerId);
    }
  });
  canvas.addEventListener("pointermove", (e) => {
    const p = board.pickBoardPoint(e);
    if (pointer.dragging?.board === board && p) {
      const g = [p[0] + pointer.dragging.offset[0], p[1] + pointer.dragging.offset[1], app.round.goal[2] ?? 0];
      for (const b of Object.values(app.boards)) b.setGoal(g);
      pointer.dragging.goal = g;
    } else if (name === "human" && pointer.down) {
      sendMouse(p, Boolean(p));
    }
    canvas.style.cursor = app.round?.phase !== "playing" && nearGoal(p) ? "grab" : "";
  });
  const release = () => {
    if (pointer.dragging?.board === board) {
      if (pointer.dragging.goal) send({ type: "set_goal", goal: pointer.dragging.goal });
      pointer.dragging = null;
    }
    if (name === "human" && pointer.down) {
      pointer.down = false;
      sendMouse(null, false, true);
    }
  };
  // Scroll over the goal during setup to rotate it in 15° steps.
  canvas.addEventListener("wheel", (e) => {
    if (app.round?.phase === "playing" || !nearGoal(board.pickBoardPoint(e))) return;
    e.preventDefault();
    const [x, y, a = 0] = app.round.goal;
    const next = [x, y, a + Math.sign(e.deltaY) * (Math.PI / 12)];
    for (const b of Object.values(app.boards)) b.setGoal(next);
    send({ type: "set_goal", goal: next });
  }, { passive: false });
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);
}

// Both boards always share one camera view so they can be compared side by side.
function setView(view) {
  app.view = view;
  for (const b of Object.values(app.boards)) b.setView(view);
}

window.addEventListener("keydown", (e) => {
  if (e.key.toLowerCase() !== "v" || e.target.closest("input, textarea")) return;
  setView(app.view === "top" ? "tilt" : "top");
});

connect();
loadChallenges();
requestAnimationFrame(tickTimer);
