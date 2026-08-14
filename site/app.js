"use strict";

const CLASS_COLORS = ["#4fe1d2", "#ffb35c", "#8f78ff", "#ff6f91", "#7dc7ff", "#c5e86c", "#ff8a5b", "#c17ee4", "#70d19f", "#f3df77"];
const METRICS = {
  nc1: "nc1_variability",
  nc2: "nc2_simplex_deviation",
  nc3: "nc3_self_duality_deviation",
  nc4: "nc4_disagreement",
};

const state = { data: null, condition: "clean", index: 0, playing: false, timer: null, previous: null, transitionStart: 0 };
const canvas = document.getElementById("geometry-canvas");
const context = canvas.getContext("2d");
const slider = document.getElementById("epoch-slider");
const playButton = document.getElementById("play-button");
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

function trajectory() { return state.data.display_states[state.condition]; }
function current() { return trajectory()[state.index]; }

function fittedBounds(states) {
  const values = states.flatMap((entry) => entry.points.concat(entry.centres));
  const xs = values.map((point) => point.x);
  const ys = values.map((point) => point.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const padX = Math.max((maxX - minX) * 0.09, 0.5), padY = Math.max((maxY - minY) * 0.09, 0.5);
  return { minX: minX - padX, maxX: maxX + padX, minY: minY - padY, maxY: maxY + padY };
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  draw(1);
}

function mix(a, b, t) { return a + (b - a) * t; }
function ease(t) { return 1 - Math.pow(1 - t, 3); }

function draw(progress = 1) {
  if (!state.data) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = canvas.width / ratio, height = canvas.height / ratio;
  context.clearRect(0, 0, width, height);
  const wheelWidth = width > 700 ? Math.min(250, width * 0.3) : Math.min(155, width * 0.36);
  const plot = { left: 18, top: 24, right: width - wheelWidth - 20, bottom: height - 32 };
  const bounds = fittedBounds(trajectory());
  const x = (value) => plot.left + (value - bounds.minX) / (bounds.maxX - bounds.minX) * (plot.right - plot.left);
  const y = (value) => plot.bottom - (value - bounds.minY) / (bounds.maxY - bounds.minY) * (plot.bottom - plot.top);
  const now = current();
  const before = state.previous || now;
  const t = ease(progress);

  const gradient = context.createRadialGradient((plot.left + plot.right) / 2, (plot.top + plot.bottom) / 2, 10, (plot.left + plot.right) / 2, (plot.top + plot.bottom) / 2, Math.max(plot.right - plot.left, plot.bottom - plot.top) * .75);
  gradient.addColorStop(0, "rgba(115,87,255,.10)");
  gradient.addColorStop(1, "rgba(9,11,21,0)");
  context.fillStyle = gradient;
  context.fillRect(plot.left, plot.top, plot.right - plot.left, plot.bottom - plot.top);

  context.strokeStyle = "rgba(255,255,255,.06)";
  context.lineWidth = 1;
  for (let i = 1; i < 5; i += 1) {
    const xx = mix(plot.left, plot.right, i / 5), yy = mix(plot.top, plot.bottom, i / 5);
    context.beginPath(); context.moveTo(xx, plot.top); context.lineTo(xx, plot.bottom); context.stroke();
    context.beginPath(); context.moveTo(plot.left, yy); context.lineTo(plot.right, yy); context.stroke();
  }

  const showTrain = document.getElementById("show-train").checked;
  const showTest = document.getElementById("show-test").checked;
  now.points.forEach((point, index) => {
    if ((point.split === "train" && !showTrain) || (point.split === "test" && !showTest)) return;
    const old = before.points[index] || point;
    const px = x(mix(old.x, point.x, t)), py = y(mix(old.y, point.y, t));
    const color = CLASS_COLORS[point.true_label];
    context.save();
    if (point.prediction !== point.true_label) {
      context.beginPath(); context.arc(px, py, point.split === "train" ? 5 : 7, 0, Math.PI * 2);
      context.strokeStyle = "#ff5973"; context.lineWidth = 1.5; context.stroke();
    }
    if (point.split === "train") {
      context.beginPath(); context.arc(px, py, 2.2, 0, Math.PI * 2); context.fillStyle = color + "99"; context.fill();
    } else {
      context.translate(px, py); context.rotate(Math.PI / 4); context.strokeStyle = color; context.lineWidth = 1.2; context.strokeRect(-2.8, -2.8, 5.6, 5.6);
    }
    context.restore();
  });

  now.centres.forEach((centre, index) => {
    const old = before.centres[index] || centre;
    const px = x(mix(old.x, centre.x, t)), py = y(mix(old.y, centre.y, t));
    context.beginPath(); context.arc(px, py, 9, 0, Math.PI * 2); context.fillStyle = "#090b15"; context.fill();
    context.strokeStyle = CLASS_COLORS[centre.class]; context.lineWidth = 2; context.stroke();
    context.fillStyle = "#fff"; context.font = "700 9px ui-monospace, monospace"; context.textAlign = "center"; context.textBaseline = "middle"; context.fillText(String(centre.class), px, py + .5);
  });
  drawAngleWheel(now.metrics.neural_collapse.class_angle_gram, width - wheelWidth / 2 - 8, Math.max(112, height * .22), wheelWidth * .34);
}

function drawAngleWheel(gram, cx, cy, radius) {
  context.save();
  context.fillStyle = "rgba(17,22,42,.88)"; context.strokeStyle = "rgba(255,255,255,.12)"; context.lineWidth = 1;
  context.beginPath(); context.arc(cx, cy, radius + 28, 0, Math.PI * 2); context.fill(); context.stroke();
  const points = Array.from({ length: 10 }, (_, i) => ({ x: cx + Math.cos(-Math.PI / 2 + i * Math.PI / 5) * radius, y: cy + Math.sin(-Math.PI / 2 + i * Math.PI / 5) * radius }));
  for (let i = 0; i < 10; i += 1) for (let j = i + 1; j < 10; j += 1) {
    const deviation = Math.min(Math.abs(gram[i][j] + 1 / 9), 1);
    context.strokeStyle = `rgba(${Math.round(mix(190,255,deviation))},${Math.round(mix(230,89,deviation))},${Math.round(mix(225,115,deviation))},${mix(.55,.2,deviation)})`;
    context.lineWidth = mix(1.1, .45, deviation);
    context.beginPath(); context.moveTo(points[i].x, points[i].y); context.lineTo(points[j].x, points[j].y); context.stroke();
  }
  points.forEach((point, i) => {
    context.beginPath(); context.arc(point.x, point.y, 8, 0, Math.PI * 2); context.fillStyle = CLASS_COLORS[i]; context.fill();
    context.fillStyle = "#080b15"; context.font = "700 8px ui-monospace, monospace"; context.textAlign = "center"; context.textBaseline = "middle"; context.fillText(String(i), point.x, point.y + .5);
  });
  context.fillStyle = "#aeb3c6"; context.font = "650 9px ui-monospace, monospace"; context.textAlign = "center"; context.fillText("9D CLASS-ANGLE WHEEL", cx, cy + radius + 22);
  context.restore();
}

function drawSpark(id, values, active, color) {
  const target = document.getElementById(id);
  const ctx = target.getContext("2d"), ratio = Math.min(window.devicePixelRatio || 1, 2), rect = target.getBoundingClientRect();
  target.width = Math.round(rect.width * ratio); target.height = Math.round(rect.height * ratio); ctx.scale(ratio, ratio);
  const width = rect.width, height = rect.height, min = Math.min(...values), max = Math.max(...values), span = max - min || 1;
  ctx.beginPath();
  values.forEach((value, index) => { const px = 2 + index / (values.length - 1) * (width - 4), py = height - 3 - (value - min) / span * (height - 8); if (index === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); });
  ctx.strokeStyle = "rgba(255,255,255,.28)"; ctx.lineWidth = 1.3; ctx.stroke();
  const value = values[active], px = 2 + active / (values.length - 1) * (width - 4), py = height - 3 - (value - min) / span * (height - 8);
  ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
}

function updateMetrics() {
  const entry = current(), nc = entry.metrics.neural_collapse, states = trajectory();
  Object.entries(METRICS).forEach(([short, key]) => {
    document.getElementById(`${short}-value`).textContent = nc[key].toFixed(3);
    drawSpark(`${short}-spark`, states.map((item) => item.metrics.neural_collapse[key]), state.index, "#4fe1d2");
  });
  const accuracy = entry.metrics.test.accuracy;
  document.getElementById("accuracy-value").textContent = `${(accuracy * 100).toFixed(1)}%`;
  drawSpark("accuracy-spark", states.map((item) => item.metrics.test.accuracy), state.index, "#ffb35c");
  document.getElementById("epoch-output").textContent = `Epoch ${entry.epoch}`;
  document.getElementById("state-description").textContent = `${state.condition}, epoch ${entry.epoch}. Held-out accuracy ${(accuracy * 100).toFixed(1)} percent. NC1 ${nc.nc1_variability.toFixed(3)}, NC2 ${nc.nc2_simplex_deviation.toFixed(3)}, NC3 ${nc.nc3_self_duality_deviation.toFixed(3)}, NC4 ${nc.nc4_disagreement.toFixed(3)}.`;
}

function setIndex(index, animate = true) {
  const old = current(); state.index = Number(index); slider.value = String(state.index); state.previous = old;
  updateMetrics();
  if (!animate || reduceMotion.matches) { draw(1); state.previous = null; return; }
  state.transitionStart = performance.now();
  const frame = (time) => { const progress = Math.min(1, (time - state.transitionStart) / 520); draw(progress); if (progress < 1) requestAnimationFrame(frame); else state.previous = null; };
  requestAnimationFrame(frame);
}

function stop() { state.playing = false; clearInterval(state.timer); state.timer = null; playButton.innerHTML = '<span aria-hidden="true">▶</span> Play training'; }
function togglePlay() {
  if (state.playing) { stop(); return; }
  state.playing = true; playButton.innerHTML = '<span aria-hidden="true">Ⅱ</span> Pause';
  if (state.index >= trajectory().length - 1) setIndex(0, false);
  state.timer = setInterval(() => { if (state.index >= trajectory().length - 1) { stop(); return; } setIndex(state.index + 1); }, reduceMotion.matches ? 900 : 1150);
}

document.querySelectorAll("[data-condition]").forEach((button) => button.addEventListener("click", () => {
  stop(); state.condition = button.dataset.condition; state.index = 0; state.previous = null;
  document.querySelectorAll("[data-condition]").forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
  slider.max = String(trajectory().length - 1); slider.value = "0"; updateMetrics(); draw(1);
}));
slider.addEventListener("input", () => { stop(); setIndex(slider.value); });
playButton.addEventListener("click", togglePlay);
document.getElementById("show-train").addEventListener("change", () => draw(1));
document.getElementById("show-test").addEventListener("change", () => draw(1));
new ResizeObserver(resizeCanvas).observe(canvas);

fetch("data/study-v1.0.json")
  .then((response) => { if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.json(); })
  .then((data) => { state.data = data; slider.max = String(trajectory().length - 1); updateMetrics(); resizeCanvas(); })
  .catch((error) => { document.querySelector(".stage-wrap").innerHTML = `<p role="alert">The study data could not be loaded: ${error.message}</p>`; });
