"use strict";

// ---- State ----
let imageList = [];
let experiments = [];   // [{rootEl, lossChart, iouChart, statusEl}]
let ws = null;
let running = false;
const MAX_EXPERIMENTS = 6;

// ---- Chart: dependency-free canvas line chart ----
class MiniChart {
  constructor(canvas, label, color) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.label = label;
    this.color = color;
    this.points = [];
    this.resize();
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.w = rect.width;
    this.h = rect.height;
    this.canvas.width = Math.round(this.w * dpr);
    this.canvas.height = Math.round(this.h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.draw();
  }

  push(x, y) {
    this.points.push([x, y]);
    this.draw();
  }

  clear() {
    this.points = [];
    this.draw();
  }

  fmt(v) {
    if (v !== 0 && Math.abs(v) < 0.01) return v.toExponential(0);
    return v.toFixed(3);
  }

  draw() {
    const ctx = this.ctx, w = this.w, h = this.h;
    const pad = { l: 38, r: 8, t: 24, b: 18 };
    ctx.clearRect(0, 0, w, h);

    ctx.fillStyle = "#1d1d1f";
    ctx.font = "600 11px var(--font), system-ui";
    ctx.font = "600 11px system-ui";
    ctx.textAlign = "left";
    ctx.fillText(this.label, 4, 12);

    if (this.points.length < 2) {
      ctx.fillStyle = "#aeaeb2";
      ctx.font = "11px system-ui";
      ctx.textAlign = "center";
      ctx.fillText("等待数据...", w / 2, h / 2);
      return;
    }

    const xs = this.points.map(p => p[0]);
    const ys = this.points.map(p => p[1]);
    const xMin = xs[0], xMax = xs[xs.length - 1];
    let yMin = Math.min(...ys), yMax = Math.max(...ys);
    if (yMax - yMin < 1e-15) yMax = yMin + 1;

    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;

    // Grid
    ctx.strokeStyle = "#f0f0f0";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = Math.round(pad.t + ph * i / 4) + 0.5;
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(w - pad.r, y);
      ctx.stroke();
    }

    // Y labels
    ctx.fillStyle = "#86868b";
    ctx.font = "9px system-ui";
    ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
      const val = yMax - (yMax - yMin) * i / 4;
      ctx.fillText(this.fmt(val), pad.l - 4, pad.t + ph * i / 4 + 3);
    }

    // X labels
    ctx.textAlign = "left";
    ctx.fillText(String(xMin), pad.l, h - 4);
    ctx.textAlign = "right";
    ctx.fillText(String(xMax), w - pad.r, h - 4);

    // Line
    ctx.strokeStyle = this.color;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (let i = 0; i < this.points.length; i++) {
      const px = pad.l + (this.points[i][0] - xMin) / (xMax - xMin || 1) * pw;
      const py = pad.t + (1 - (this.points[i][1] - yMin) / (yMax - yMin)) * ph;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }
}

// ---- API ----
async function fetchImages() {
  const res = await fetch("/api/images");
  const data = await res.json();
  return data.images;
}

async function fetchPreview(image, expand, contourSigma, contourThreshold) {
  const res = await fetch("/api/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image, expand, contour_sigma: contourSigma, contour_threshold: contourThreshold }),
  });
  return await res.json();
}

// ---- Experiment rows ----
function setExpCount(n) {
  n = Math.max(1, Math.min(MAX_EXPERIMENTS, n));
  while (experiments.length < n) experiments.push(createRow(experiments.length));
  while (experiments.length > n) {
    const exp = experiments.pop();
    exp.rootEl.remove();
  }
  document.getElementById("exp-count-display").textContent = n;
  document.getElementById("dec-btn").disabled = n <= 1;
  document.getElementById("inc-btn").disabled = n >= MAX_EXPERIMENTS || running;
}

function createRow(index) {
  const root = document.createElement("div");
  root.className = "exp-row";

  // ---- Top section: params + charts side by side ----
  const main = document.createElement("div");
  main.className = "exp-main";

  const params = document.createElement("div");
  params.className = "exp-params";

  const top = document.createElement("div");
  top.className = "exp-params-top";
  top.innerHTML = '<span class="exp-label">实验 ' + (index + 1) + "</span>";
  const status = document.createElement("span");
  status.className = "exp-status status-idle";
  status.textContent = "待机";
  top.appendChild(status);

  // 第 1 行：图片（独占整行）
  const fieldsRow1 = document.createElement("div");
  fieldsRow1.className = "fields-row";

  const imgWrap = document.createElement("div");
  imgWrap.className = "field";
  imgWrap.innerHTML = '<span class="field-label">图片</span>';
  const imgSel = document.createElement("select");
  imageList.forEach(f => {
    const opt = document.createElement("option");
    opt.value = f;
    opt.textContent = f;
    imgSel.appendChild(opt);
  });
  imgWrap.appendChild(imgSel);
  fieldsRow1.appendChild(imgWrap);

  // 第 2 行：画布扩大 + 训练轮数
  const fieldsRow2 = document.createElement("div");
  fieldsRow2.className = "fields-row";

  const expWrap = document.createElement("div");
  expWrap.className = "field";
  expWrap.innerHTML = '<span class="field-label">画布扩大</span>';
  const expInput = document.createElement("input");
  expInput.type = "text";
  expInput.value = "1";
  expWrap.appendChild(expInput);

  const iterWrap = document.createElement("div");
  iterWrap.className = "field";
  iterWrap.innerHTML = '<span class="field-label">训练轮数</span>';
  const iterInput = document.createElement("input");
  iterInput.type = "text";
  iterInput.value = "3000";
  iterWrap.appendChild(iterInput);

  fieldsRow2.append(expWrap, iterWrap);

  // 第 3 行：高斯半径 + 轮廓阈值
  const fieldsRow3 = document.createElement("div");
  fieldsRow3.className = "fields-row";

  const sigmaWrap = document.createElement("div");
  sigmaWrap.className = "field";
  sigmaWrap.innerHTML = '<span class="field-label">高斯半径</span>';
  const sigmaInput = document.createElement("input");
  sigmaInput.type = "text";
  sigmaInput.value = "16";
  sigmaWrap.appendChild(sigmaInput);

  const threshWrap = document.createElement("div");
  threshWrap.className = "field";
  threshWrap.innerHTML = '<span class="field-label">轮廓阈值</span>';
  const threshInput = document.createElement("input");
  threshInput.type = "text";
  threshInput.value = "0.20";
  threshWrap.appendChild(threshInput);

  fieldsRow3.append(sigmaWrap, threshWrap);

  // 第 4 行：直方图权重 + 背景权重（相对振幅项的初始贡献占比）
  const fieldsRow4 = document.createElement("div");
  fieldsRow4.className = "fields-row";

  const shareHistWrap = document.createElement("div");
  shareHistWrap.className = "field";
  shareHistWrap.innerHTML = '<span class="field-label">直方图权重</span>';
  const shareHistInput = document.createElement("input");
  shareHistInput.type = "text";
  shareHistInput.value = "0.5";
  shareHistWrap.appendChild(shareHistInput);

  const shareBgWrap = document.createElement("div");
  shareBgWrap.className = "field";
  shareBgWrap.innerHTML = '<span class="field-label">背景权重</span>';
  const shareBgInput = document.createElement("input");
  shareBgInput.type = "text";
  shareBgInput.value = "0.10";
  shareBgWrap.appendChild(shareBgInput);

  fieldsRow4.append(shareHistWrap, shareBgWrap);

  params.append(top, fieldsRow1, fieldsRow2, fieldsRow3, fieldsRow4);

  // Charts
  const charts = document.createElement("div");
  charts.className = "exp-charts";

  const lossBox = document.createElement("div");
  lossBox.className = "chart-box";
  const lossCanvas = document.createElement("canvas");
  lossBox.appendChild(lossCanvas);

  const iouBox = document.createElement("div");
  iouBox.className = "chart-box";
  const iouCanvas = document.createElement("canvas");
  iouBox.appendChild(iouCanvas);

  charts.append(lossBox, iouBox);
  main.append(params, charts);

  // ---- Bottom section: result buttons (full width) ----
  const btnRow = document.createElement("div");
  btnRow.className = "btn-row";

  const previewBtn = document.createElement("button");
  previewBtn.className = "preview-btn";
  previewBtn.textContent = "轮廓对比图";
  previewBtn.addEventListener("click", () => {
    const exp = experiments[index];
    if (exp && exp.outputDir) {
      showResultImage(index, "support.png", "轮廓对比图");
    } else {
      doPreview(index);
    }
  });
  btnRow.appendChild(previewBtn);

  const realBtn = document.createElement("button");
  realBtn.className = "preview-btn";
  realBtn.textContent = "效果对比图";
  realBtn.disabled = true;
  realBtn.addEventListener("click", () => showResultImage(index, "real_space.png", "效果对比图"));
  btnRow.appendChild(realBtn);

  root.append(main, btnRow);
  document.getElementById("experiments").appendChild(root);

  const lossChart = new MiniChart(lossCanvas, "总损失", "#0066cc");
  const iouChart = new MiniChart(iouCanvas, "IoU", "#34c759");

  return {
    rootEl: root,
    imgSel, expInput, iterInput, sigmaInput, threshInput,
    shareHistInput, shareBgInput,
    statusEl: status,
    previewBtn, realBtn,
    lossChart, iouChart,
    outputDir: null,
  };
}

function setStatus(exp, text, cls) {
  exp.statusEl.textContent = text;
  exp.statusEl.className = "exp-status " + cls;
}

function getConfig(exp) {
  return {
    image: "images/" + exp.imgSel.value,
    expand: parseFloat(exp.expInput.value) || 1,
    iterations: parseInt(exp.iterInput.value, 10) || 1,
    contour_sigma: parseFloat(exp.sigmaInput.value) || 16,
    contour_threshold: parseFloat(exp.threshInput.value) || 0.2,
    share_histogram: parseFloat(exp.shareHistInput.value) || 0.5,
    share_background: parseFloat(exp.shareBgInput.value) || 0.1,
  };
}

// ---- Preview / Result images ----
function openModal(title, src) {
  document.getElementById("modal-title").textContent = title;
  document.getElementById("preview-img").src = src;
  document.getElementById("modal-loading").classList.add("hidden");
  document.getElementById("preview-modal").classList.remove("hidden");
}

function openModalLoading(title) {
  document.getElementById("modal-title").textContent = title;
  document.getElementById("preview-img").src = "";
  document.getElementById("modal-loading").classList.remove("hidden");
  document.getElementById("preview-modal").classList.remove("hidden");
}

async function doPreview(index) {
  const exp = experiments[index];
  const cfg = getConfig(exp);
  openModalLoading("轮廓对比图");
  try {
    const data = await fetchPreview(cfg.image, cfg.expand, cfg.contour_sigma, cfg.contour_threshold);
    document.getElementById("modal-title").textContent =
      "轮廓对比图 · 轮廓占比 " + (data.ratio * 100).toFixed(2) + "%";
    document.getElementById("preview-img").src = data.image;
    document.getElementById("modal-loading").classList.add("hidden");
  } catch (e) {
    appendTerminal("预览出错：" + e.message, true);
    document.getElementById("preview-modal").classList.add("hidden");
  }
}

function showResultImage(index, filename, title) {
  const exp = experiments[index];
  if (!exp || !exp.outputDir) return;
  openModal(title, "/" + exp.outputDir + "/" + filename);
}

// ---- Terminal ----
function appendTerminal(text, isError) {
  const term = document.getElementById("terminal");
  const span = document.createElement("div");
  if (isError) span.className = "term-err";
  span.textContent = text;
  term.appendChild(span);
  term.scrollTop = term.scrollHeight;
}

function terminalSeparator(label) {
  const term = document.getElementById("terminal");
  const sep = document.createElement("div");
  sep.className = "term-sep";
  sep.textContent = "──── " + label + " ────";
  term.appendChild(sep);
  term.scrollTop = term.scrollHeight;
}

// ---- WebSocket ----
function connectWS() {
  ws = new WebSocket("ws://" + location.host + "/ws");
  ws.onmessage = (evt) => {
    const msg = JSON.parse(evt.data);
    handleMsg(msg);
  };
  ws.onclose = () => {
    if (running) {
      appendTerminal("连接断开。", true);
      running = false;
      document.getElementById("start-btn").disabled = false;
      document.getElementById("start-btn").textContent = "开始运行";
      document.getElementById("pause-btn").disabled = true;
      document.getElementById("stop-btn").disabled = true;
    }
    setTimeout(connectWS, 2000);
  };
}

function handleMsg(msg) {
  const exp = experiments[msg.index];
  switch (msg.type) {
    case "exp_start":
      if (exp) {
        setStatus(exp, "运行中", "status-running");
        exp.lossChart.clear();
        exp.iouChart.clear();
        terminalSeparator("实验 " + (msg.index + 1) + " — " +
          msg.config.image + " / 扩大" + msg.config.expand + "倍");
      }
      break;
    case "terminal":
      appendTerminal(msg.text);
      break;
    case "metric":
      if (exp) {
        exp.lossChart.push(msg.iteration, msg.total);
        exp.iouChart.push(msg.iteration, msg.iou);
      }
      break;
    case "exp_done":
      if (exp) {
        setStatus(exp, "已完成", "status-done");
        if (msg.output_dir) {
          exp.outputDir = msg.output_dir;
          exp.realBtn.disabled = false;
        }
      }
      break;
    case "exp_failed":
      if (exp) setStatus(exp, "失败", "status-failed");
      appendTerminal("实验 " + (msg.index + 1) + " 异常退出（代码 " + msg.code + "）", true);
      break;
    case "exp_stopped":
      if (exp) setStatus(exp, "已终止", "status-failed");
      break;
    case "paused":
      document.getElementById("pause-btn").textContent = "继续";
      appendTerminal("── 实验已暂停 ──");
      break;
    case "resumed":
      document.getElementById("pause-btn").textContent = "暂停";
      appendTerminal("── 实验已恢复 ──");
      break;
    case "all_done":
      running = false;
      document.getElementById("start-btn").disabled = false;
      document.getElementById("start-btn").textContent = "开始运行";
      document.getElementById("pause-btn").disabled = true;
      document.getElementById("pause-btn").textContent = "暂停";
      document.getElementById("stop-btn").disabled = true;
      appendTerminal("全部实验已结束。");
      break;
  }
}

// ---- Start / Pause / Stop ----
function startExperiments() {
  if (running || !ws || ws.readyState !== WebSocket.OPEN) return;
  experiments.forEach(exp => {
    setStatus(exp, "排队中", "status-idle");
    exp.lossChart.clear();
    exp.iouChart.clear();
    exp.outputDir = null;
    exp.realBtn.disabled = true;
  });
  document.getElementById("terminal").innerHTML = "";
  running = true;
  document.getElementById("start-btn").disabled = true;
  document.getElementById("start-btn").textContent = "运行中…";
  document.getElementById("pause-btn").disabled = false;
  document.getElementById("stop-btn").disabled = false;
  document.getElementById("inc-btn").disabled = true;
  ws.send(JSON.stringify({
    type: "start",
    experiments: experiments.map(getConfig),
  }));
  appendTerminal("正在启动 " + experiments.length + " 组实验…");
}

function togglePause() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const btn = document.getElementById("pause-btn");
  ws.send(JSON.stringify({ type: btn.textContent === "暂停" ? "pause" : "resume" }));
}

function stopExperiments() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: "stop" }));
  appendTerminal("正在终止实验…");
}

function resetAll() {
  if (running) return;
  experiments.forEach(exp => {
    setStatus(exp, "待机", "status-idle");
    exp.lossChart.clear();
    exp.iouChart.clear();
    exp.outputDir = null;
    exp.realBtn.disabled = true;
  });
  document.getElementById("terminal").innerHTML = "";
  document.getElementById("inc-btn").disabled = experiments.length >= MAX_EXPERIMENTS;
  document.getElementById("dec-btn").disabled = experiments.length <= 1;
  appendTerminal("已重置。");
}

// ---- Init ----
async function init() {
  imageList = await fetchImages();
  setExpCount(1);
  connectWS();

  document.getElementById("inc-btn").addEventListener("click", () => {
    if (!running) setExpCount(experiments.length + 1);
  });
  document.getElementById("dec-btn").addEventListener("click", () => {
    if (!running) setExpCount(experiments.length - 1);
  });
  document.getElementById("start-btn").addEventListener("click", startExperiments);
  document.getElementById("pause-btn").addEventListener("click", togglePause);
  document.getElementById("stop-btn").addEventListener("click", stopExperiments);
  document.getElementById("reset-btn").addEventListener("click", resetAll);
  document.getElementById("preview-close").addEventListener("click", () => {
    document.getElementById("preview-modal").classList.add("hidden");
  });
  document.querySelector(".modal-backdrop").addEventListener("click", () => {
    document.getElementById("preview-modal").classList.add("hidden");
  });

  window.addEventListener("resize", () => {
    experiments.forEach(exp => {
      exp.lossChart.resize();
      exp.iouChart.resize();
    });
  });
}

init();
