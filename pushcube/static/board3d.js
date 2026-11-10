// Three.js view of one 2D board. Physics lives on the server; this only draws its state.
import * as THREE from "three";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";

const PAPER = 0xf4efe4;
const INK = 0x1f2433;
const GOAL = 0x2f8f5b;
const CUBE = 0xf0a23b;

export function simToWorld([x, y], half) {
  return { x: x - half, z: y - half };
}

export function worldToSim({ x, z }, half) {
  return [x + half, z + half];
}

function gridTexture(size) {
  const c = document.createElement("canvas");
  c.width = c.height = 1024;
  const g = c.getContext("2d");
  g.fillStyle = "#f7f3ea";
  g.fillRect(0, 0, 1024, 1024);
  const step = 1024 / (size / 16);
  for (let i = 0, n = 0; i <= 1024; i += step, n++) {
    g.strokeStyle = n % 4 === 0 ? "rgba(31,36,51,0.16)" : "rgba(31,36,51,0.06)";
    g.lineWidth = n % 4 === 0 ? 2 : 1;
    g.beginPath(); g.moveTo(i, 0); g.lineTo(i, 1024); g.stroke();
    g.beginPath(); g.moveTo(0, i); g.lineTo(1024, i); g.stroke();
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

function squareFrame(size, thickness, height, color, opacity = 1) {
  const group = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({ color, transparent: opacity < 1, opacity, roughness: 0.6 });
  const h = size / 2;
  for (const [w, d, x, z] of [
    [size + thickness, thickness, 0, -h], [size + thickness, thickness, 0, h],
    [thickness, size + thickness, -h, 0], [thickness, size + thickness, h, 0],
  ]) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, height, d), mat);
    m.position.set(x, height / 2, z);
    group.add(m);
  }
  return group;
}

export function createBoard3D(container, shape, { accent = 0xd9472b } = {}) {
  const N = shape.board_size;
  const half = N / 2;
  const [wMin, wMax] = shape.walls;

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(PAPER);

  scene.add(new THREE.HemisphereLight(0xfff8ec, 0xc9bfae, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 2.2);
  sun.position.set(-180, 520, 260);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -320, right: 320, top: 320, bottom: -320, near: 50, far: 1200 });
  scene.add(sun);

  const table = new THREE.Mesh(
    new THREE.PlaneGeometry(N, N),
    new THREE.MeshStandardMaterial({ map: gridTexture(N), roughness: 0.95 }),
  );
  table.rotation.x = -Math.PI / 2;
  table.receiveShadow = true;
  scene.add(table);

  // Low rails whose inner faces sit exactly on the simulated walls.
  const railMat = new THREE.MeshStandardMaterial({ color: INK, roughness: 0.5 });
  const t = 10, railH = 16;
  const inner = { lo: wMin - half, hi: wMax - half };
  const span = inner.hi - inner.lo + 2 * t;
  for (const [w, d, x, z] of [
    [span, t, (inner.lo + inner.hi) / 2, inner.lo - t / 2], [span, t, (inner.lo + inner.hi) / 2, inner.hi + t / 2],
    [t, span, inner.lo - t / 2, (inner.lo + inner.hi) / 2], [t, span, inner.hi + t / 2, (inner.lo + inner.hi) / 2],
  ]) {
    const rail = new THREE.Mesh(new THREE.BoxGeometry(w, railH, d), railMat);
    rail.position.set(x, railH / 2, z);
    rail.castShadow = rail.receiveShadow = true;
    scene.add(rail);
  }

  const S = shape.cube_size;
  const goal = new THREE.Group();
  const goalFill = new THREE.Mesh(
    new THREE.PlaneGeometry(S, S),
    new THREE.MeshBasicMaterial({ color: GOAL, transparent: true, opacity: 0.16, depthWrite: false }),
  );
  goalFill.rotation.x = -Math.PI / 2;
  goalFill.position.y = 0.3;
  goal.add(goalFill, squareFrame(S, 3, 1.2, GOAL));
  scene.add(goal);

  const cube = new THREE.Mesh(
    new RoundedBoxGeometry(S, S, S, 4, 5),
    new THREE.MeshStandardMaterial({ color: CUBE, roughness: 0.45 }),
  );
  cube.castShadow = cube.receiveShadow = true;
  cube.position.y = S / 2;
  scene.add(cube);

  const R = shape.pusher_radius;
  const pusher = new THREE.Group();
  const puck = new THREE.Mesh(new THREE.CylinderGeometry(R, R, 26, 40), new THREE.MeshStandardMaterial({ color: INK, roughness: 0.35 }));
  puck.position.y = 13;
  puck.castShadow = true;
  const cap = new THREE.Mesh(new THREE.CylinderGeometry(R * 0.55, R * 0.55, 2, 32), new THREE.MeshStandardMaterial({ color: accent }));
  cap.position.y = 27;
  pusher.add(puck, cap);
  scene.add(pusher);

  const ghosts = new THREE.Group();
  scene.add(ghosts);

  const tilt = new THREE.PerspectiveCamera(34, 1, 10, 3000);
  tilt.position.set(0, 800, 650);
  tilt.lookAt(0, 0, 62);
  const view = half + 16; // include the rails in the top-down view
  const top = new THREE.OrthographicCamera(-view, view, view, -view, 10, 2000);
  top.position.set(0, 900, 0);
  top.up.set(0, 0, -1);
  top.lookAt(0, 0, 0);
  let camera = tilt;

  const target = { cube: cube.position.clone(), pusher: pusher.position.clone() };
  let snapNext = true;

  function place(obj, key, xy, y) {
    const { x, z } = simToWorld(xy, half);
    target[key].set(x, y, z);
    if (snapNext) obj.position.copy(target[key]);
  }

  function resize() {
    const w = container.clientWidth, h = container.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    tilt.aspect = w / h;
    tilt.updateProjectionMatrix();
    const a = w / h;
    Object.assign(top, a >= 1 ? { left: -view * a, right: view * a, top: view, bottom: -view } : { left: -view, right: view, top: view / a, bottom: -view / a });
    top.updateProjectionMatrix();
  }
  const ro = new ResizeObserver(resize);
  ro.observe(container);
  resize();

  let last = performance.now(), raf = 0;
  function frame(now) {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    const k = 1 - Math.exp(-dt * 18);
    cube.position.lerp(target.cube, k);
    pusher.position.lerp(target.pusher, k);
    renderer.render(scene, camera);
    raf = requestAnimationFrame(frame);
  }
  raf = requestAnimationFrame(frame);

  const raycaster = new THREE.Raycaster();
  const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);

  return {
    setState(pusherXY, cubeXY) {
      place(cube, "cube", cubeXY, S / 2);
      place(pusher, "pusher", pusherXY, 0);
      snapNext = false;
    },
    snap() { snapNext = true; },
    setGoal(goalXY) {
      const { x, z } = simToWorld(goalXY, half);
      goal.position.set(x, 0, z);
    },
    setView(view) { camera = view === "top" ? top : tilt; },
    get view() { return camera === top ? "top" : "tilt"; },
    setGhosts(list) {
      ghosts.clear();
      for (const g of list) {
        const m = new THREE.Mesh(
          new RoundedBoxGeometry(S, S, S, 3, 5),
          new THREE.MeshStandardMaterial({ color: accent, transparent: true, opacity: 0.18 + 0.5 * (g.weight ?? 0.3), depthWrite: false }),
        );
        const { x, z } = simToWorld(g.xy, half);
        m.position.set(x, S / 2, z);
        ghosts.add(m);
      }
    },
    pickBoardPoint(ev) {
      const r = renderer.domElement.getBoundingClientRect();
      const ndc = new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
      raycaster.setFromCamera(ndc, camera);
      const hit = new THREE.Vector3();
      if (!raycaster.ray.intersectPlane(plane, hit)) return null;
      const [sx, sy] = worldToSim({ x: hit.x, z: hit.z }, half);
      return sx < 0 || sy < 0 || sx > N || sy > N ? null : [sx, sy];
    },
    canvas: renderer.domElement,
    dispose() { cancelAnimationFrame(raf); ro.disconnect(); renderer.dispose(); },
  };
}
