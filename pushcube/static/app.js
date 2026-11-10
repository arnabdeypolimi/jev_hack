import { createBoard3D } from "./board3d.js";
import { createPanel } from "./panel.js";

const $ = (sel, root = document) => root.querySelector(sel);
const sides = { human: $(".side.you"), ai: $(".side.jev") };
const panel = createPanel($(".mind"));

const app = {
  ws: null,
  boards: {},
  round: null,
  aiCube: null,
  timerEnds: null,
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
  board.setState(msg.pusher, msg.cube);
  $(".pushes", sides[msg.board]).textContent = msg.pushes;
  $(".overlap", sides[msg.board]).textContent = `${Math.round(msg.overlap * 100)}%`;
  if (msg.board === "ai") app.aiCube = msg.cube;
}

function applyRound(r) {
  const fresh = !app.round || app.round.id !== r.id;
  app.round = r;
  for (const b of Object.values(app.boards)) {
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
    ensureBoards(m.shape);
    document.documentElement.style.setProperty("--tau", `${(m.tau ?? 0.5) * 100}%`);
    applyRound(m.round);
    applyState(m.boards.human);
    applyState(m.boards.ai);
    const badges = [];
    if (m.replay) badges.push(`<span class="badge">replay</span>`);
    if (!m.jev_available) badges.push(`<span class="badge warn">judge offline — no Jev key</span>`);
    $(".badges").innerHTML = badges.join("");
    if (m.last_decision) panel.showDecision(m.last_decision);
  },
  round: applyRound,
  state: applyState,
  ai_status(m) {
    panel.setStatus(m.status, m.option);
    if (m.status === "acting" || m.status === "deciding") app.boards.ai?.setGhosts([]);
  },
  decision(m) {
    panel.setContext(app.round?.goal, app.aiCube);
    panel.showDecision(m);
  },
  imagination(m) {
    panel.setContext(app.round?.goal, app.aiCube);
    panel.showImagination(m);
    app.boards.ai?.setGhosts(m.outcomes.map((o) => ({ xy: o.cube_after, weight: 0.3 })));
  },
  pick(m) {
    panel.showPick(m);
    app.boards.ai?.setGhosts([]);
  },
};

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

$(".start").addEventListener("click", () => send({ type: "start", challenge: app.round?.challenge || "easy-east" }));
window.addEventListener("keydown", (e) => {
  if (e.key.toLowerCase() !== "v" || e.target.closest("input, textarea")) return;
  for (const b of Object.values(app.boards)) b.setView(b.view === "top" ? "tilt" : "top");
});

export { app, send, handlers };
connect();
requestAnimationFrame(tickTimer);
