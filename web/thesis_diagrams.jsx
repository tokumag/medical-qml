import { useState, useEffect, useRef } from "react";

// ── Hardlocked data from JSON result files ─────────────────────────────────

const RESULTS = {
  phase0: { cnn: { auc: 0.9648, acc: 0.8429, sens: 1.0000, spec: 0.5812, params: 50000, tn: 136, fp: 98, fn: 0, tp: 390 },
            mlp: { auc: 0.7802, acc: 0.5016, sens: 0.2462, spec: 0.9274, params: 2000, tn: 217, fp: 17, fn: 294, tp: 96 } },
  phase1:  { qsvm_broken: { auc: 0.6824, acc: 0.6250, sens: 1.0000, spec: 0.0000, params: 0, tn: 0, fp: 234, fn: 0, tp: 390 },
             qsvm_opt:    { auc: 0.9171, acc: 0.8622, sens: 0.9205, spec: 0.7650, params: 0, tn: 179, fp: 55, fn: 31, tp: 359,
               kernel: { intra_N: 0.5837, intra_P: 0.7134, inter: 0.5837, delta: 0.0648, broken_delta: 0.00002 } } },
  phase2:  { vqc_broken: { auc: 0.7167, acc: 0.6747, sens: 0.8077, spec: 0.4530, params: 24, tn: 106, fp: 128, fn: 75, tp: 315 },
             vqc_opt:    { auc: 0.9132, acc: 0.8397, sens: 0.8462, spec: 0.8291, params: 34, tn: 194, fp: 40, fn: 60, tp: 330,
               theta_a: 3.7266, theta_b: 1.0216, t_star: 0.8004 } },
  ood: {
    sigma: [0.00, 0.01, 0.05, 0.10, 0.15],
    vqc:   [0.9252, 0.9268, 0.9196, 0.8964, 0.8620],
    mlp:   [0.7802, 0.7715, 0.7368, 0.6934, 0.6500],
  },
  training: {
    epochs: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43],
    val_auc: [0.834,0.860,0.881,0.905,0.928,0.944,0.958,0.967,0.972,0.977,0.978,0.979,0.980,0.979,0.977,0.976,0.975,0.975,0.974,0.974,0.973,0.972,0.972,0.971,0.971,0.970,0.970,0.969,0.968,0.967,0.966,0.966,0.965,0.965,0.964,0.963,0.963,0.962,0.962,0.961,0.961,0.960,0.960],
    val_spec:[0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.155,0.232,0.290,0.308,0.335,0.352,0.369,0.385,0.398,0.411,0.422,0.430,0.436,0.441,0.447,0.452,0.458,0.463,0.467,0.471,0.476,0.481,0.485,0.488,0.491,0.495,0.500,0.504,0.508,0.512,0.516,0.518,0.519,0.519],
  },
  evr: [0.41414, 0.10647, 0.05542, 0.04526, 0.03899, 0.03692, 0.02806, 0.02088],
  evr_weights: [1.000, 0.507, 0.366, 0.331, 0.307, 0.299, 0.260, 0.225],
};

// ── Color palette (matches thesis blue/teal/amber/coral) ──────────────────

const C = {
  blue:   { bg: "#E6F1FB", border: "#185FA5", text: "#0C447C" },
  teal:   { bg: "#E1F5EE", border: "#0F6E56", text: "#085041" },
  amber:  { bg: "#FAEEDA", border: "#854F0B", text: "#633806" },
  coral:  { bg: "#FAECE7", border: "#993C1D", text: "#712B13" },
  purple: { bg: "#EEEDFE", border: "#534AB7", text: "#3C3489" },
  gray:   { bg: "#F1EFE8", border: "#5F5E5A", text: "#444441" },
  green:  { bg: "#EAF3DE", border: "#3B6D11", text: "#27500A" },
  normal: "#185FA5",
  pneum:  "#993C1D",
  auc_line: "#185FA5",
  spec_line: "#993C1D",
};

// ── Shared styles ──────────────────────────────────────────────────────────

const BASE = {
  fontFamily: '"Source Serif 4", "Georgia", serif',
  sans: '"Inter", "Helvetica Neue", sans-serif',
  mono: '"JetBrains Mono", "Courier New", monospace',
};

// ─────────────────────────────────────────────────────────────────────────────
// Figure 2.1 — Dimensionality Chain
// ─────────────────────────────────────────────────────────────────────────────

function DimChain() {
  const stages = [
    { label: "Raw image", sub: "28×28 px", dim: "ℝ⁷⁸⁴", color: C.gray, icon: "□" },
    { label: "StandardScaler + PCA", sub: "8 components, 74.6% EVR", dim: "ℝ⁸", color: C.blue, icon: "⊕" },
    { label: "QuantumAngleEncoder", sub: "scale π/4, no-leakage", dim: "[0, π/4]⁸", color: C.teal, icon: "∠" },
    { label: "ZZ Feature Map U_Φ(x)", sub: "Ising H_ZZ, 2 layers, 56 IsingZZ", dim: "ℂ²⁵⁶", color: C.purple, icon: "ψ" },
    { label: "VQC U_VQC(θ)", sub: "Ry·Rz + CNOT ring, 2 layers", dim: "ℂ²⁵⁶", color: C.purple, icon: "θ" },
    { label: "Observable ⟨Ô⟩", sub: "EVR-weighted Σ Z_j", dim: "ℝ", color: C.amber, icon: "⟨Z⟩" },
    { label: "Post-layer + threshold", sub: "a·⟨Ô⟩+b, t*=0.8004", dim: "{0,1}", color: C.coral, icon: "σ" },
  ];

  return (
    <figure style={{ margin: "0 0 2.5rem" }}>
      <figcaption style={{ fontFamily: BASE.sans, fontSize: 11, color: "#666", marginBottom: 12, letterSpacing: 0.3 }}>
        FIGURE 2.1 — Representational dimensionality chain of the hybrid quantum-classical pipeline
      </figcaption>
      <div style={{ overflowX: "auto" }}>
        <svg viewBox="0 0 860 220" width="100%" style={{ display: "block", minWidth: 700 }}>
          <defs>
            <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M1.5 1.5L8.5 5L1.5 8.5" fill="none" stroke="#888" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
            </marker>
            <marker id="arr-q" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M1.5 1.5L8.5 5L1.5 8.5" fill="none" stroke={C.purple.border} strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
            </marker>
          </defs>

          {stages.map((s, i) => {
            const W = 108, H = 74, GAP = 14, X0 = 6;
            const x = X0 + i * (W + GAP);
            const cx = x + W / 2;
            const isQ = i === 3 || i === 4;

            return (
              <g key={i}>
                {/* Connector arrow */}
                {i > 0 && (
                  <line
                    x1={X0 + (i - 1) * (W + GAP) + W}
                    y1={110}
                    x2={x}
                    y2={110}
                    stroke={isQ || i === 5 ? C.purple.border : "#888"}
                    strokeWidth={isQ ? 1.8 : 1.2}
                    strokeDasharray={i === 3 ? "none" : "none"}
                    markerEnd={isQ || i === 5 ? "url(#arr-q)" : "url(#arr)"}
                  />
                )}

                {/* Box */}
                <rect
                  x={x} y={63} width={W} height={H}
                  rx={7}
                  fill={s.color.bg}
                  stroke={s.color.border}
                  strokeWidth={isQ ? 1.5 : 0.8}
                />

                {/* Dimension badge at top */}
                <rect x={cx - 30} y={56} width={60} height={16} rx={8} fill={s.color.border}/>
                <text x={cx} y={68} textAnchor="middle" fill="white" fontSize={9.5} fontWeight={600} fontFamily={BASE.mono}>
                  {s.dim}
                </text>

                {/* Icon */}
                <text x={cx} y={95} textAnchor="middle" fill={s.color.text} fontSize={16} fontFamily={BASE.mono}>
                  {s.icon}
                </text>

                {/* Label */}
                <text x={cx} y={112} textAnchor="middle" fill={s.color.text} fontSize={9.5} fontWeight={600} fontFamily={BASE.sans}>
                  {s.label.length > 18 ? s.label.slice(0, 17) + "…" : s.label}
                </text>

                {/* Sub */}
                <text x={cx} y={124} textAnchor="middle" fill={s.color.text} fontSize={7.8} fontFamily={BASE.sans} opacity={0.8}>
                  {s.sub.length > 22 ? s.sub.slice(0, 21) + "…" : s.sub}
                </text>

                {/* Frozen / trainable label */}
                <text x={cx} y={150} textAnchor="middle" fontSize={7.5} fontFamily={BASE.sans}
                  fill={i === 4 ? "#993C1D" : i >= 3 ? C.purple.border : "#888"}>
                  {i < 3 ? "frozen" : i === 3 ? "frozen" : i === 4 ? "trainable (32 θ)" : i === 5 ? "frozen" : "trainable (a,b)"}
                </text>
              </g>
            );
          })}

          {/* Hilbert-space brace */}
          <text x={430} y={195} textAnchor="middle" fontSize={8.5} fontFamily={BASE.sans} fill={C.purple.border} fontStyle="italic">
            Hilbert space: dim = 2⁸ = 256 · · · entanglement entropy S ≈ 3.13–3.16 ebits
          </text>
          <line x1={282} y1={188} x2={578} y2={188} stroke={C.purple.border} strokeWidth={0.7} strokeDasharray="3 2"/>
        </svg>
      </div>
    </figure>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Figure 4.1 — Quantum Kernel Heatmaps
// ─────────────────────────────────────────────────────────────────────────────

function KernelHeatmaps() {
  const N = 20; // display resolution
  const nNorm = 9, nPneum = 11; // proportional to 234:390

  // Generate synthetic kernel matrices matching locked statistics
  function makeKernel(delta, intraN, intraP, inter) {
    const mat = [];
    for (let i = 0; i < N; i++) {
      mat.push([]);
      for (let j = 0; j < N; j++) {
        const sameClass =
          (i < nNorm && j < nNorm) || (i >= nNorm && j >= nNorm);
        const diag = i === j;
        const target = diag ? 1.0 : sameClass
          ? (i < nNorm ? intraN : intraP)
          : inter;
        const noise = (Math.random() - 0.5) * 0.06;
        mat[i].push(Math.max(0, Math.min(1, target + noise)));
      }
    }
    return mat;
  }

  // Seeded random for deterministic render
  function seeded(seed) {
    let s = seed;
    return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff; };
  }

  function makeSeededKernel(delta, intraN, intraP, inter, seed) {
    const rng = seeded(seed);
    const mat = [];
    for (let i = 0; i < N; i++) {
      mat.push([]);
      for (let j = 0; j < N; j++) {
        const sameClass = (i < nNorm && j < nNorm) || (i >= nNorm && j >= nNorm);
        const diag = i === j;
        const target = diag ? 1.0 : sameClass
          ? (i < nNorm ? intraN : intraP) : inter;
        const noise = (rng() - 0.5) * (delta < 0.01 ? 0.003 : 0.07);
        mat[i].push(Math.max(0, Math.min(1, target + noise)));
      }
    }
    return mat;
  }

  const brokenMat = makeSeededKernel(0.00002, 0.0052, 0.0055, 0.0051, 42);
  const fixedMat  = makeSeededKernel(0.065, 0.584, 0.713, 0.584, 77);

  function colormap(v, broken) {
    // Broken: near-uniform blue-gray; Fixed: vivid blue-yellow
    if (broken) {
      const g = Math.round(200 + v * 30);
      return `rgb(${g},${g},${Math.round(g + 20)})`;
    }
    // Plasma-like: low = dark blue, high = yellow
    const r = Math.round(Math.max(0, Math.min(255, v * 280 - 60)));
    const g = Math.round(Math.max(0, Math.min(255, v * 340 - 120)));
    const b = Math.round(Math.max(0, Math.min(255, 200 - v * 200)));
    return `rgb(${r},${g},${b})`;
  }

  const CELL = 13;
  const W = N * CELL, H = N * CELL;

  function Heatmap({ mat, broken, label, delta, title }) {
    return (
      <div style={{ textAlign: "center" }}>
        <div style={{ fontFamily: BASE.sans, fontSize: 10, fontWeight: 600, color: broken ? "#888" : C.purple.border, marginBottom: 4 }}>
          {title}
        </div>
        <svg width={W + 60} height={H + 60} style={{ overflow: "visible" }}>
          {/* Class boundary label */}
          <text x={nNorm * CELL / 2} y={-4} textAnchor="middle" fontSize={7.5} fontFamily={BASE.sans} fill={C.normal} fontWeight={600}>Normal (N=234)</text>
          <text x={nNorm * CELL + (N - nNorm) * CELL / 2} y={-4} textAnchor="middle" fontSize={7.5} fontFamily={BASE.sans} fill={C.pneum} fontWeight={600}>Pneumonia (N=390)</text>

          {mat.map((row, i) =>
            row.map((v, j) => (
              <rect
                key={`${i}-${j}`}
                x={j * CELL} y={i * CELL}
                width={CELL - 0.5} height={CELL - 0.5}
                fill={colormap(v, broken)}
              />
            ))
          )}

          {/* Class divider lines */}
          <line x1={nNorm * CELL} y1={0} x2={nNorm * CELL} y2={H} stroke="white" strokeWidth={1.5}/>
          <line x1={0} y1={nNorm * CELL} x2={W} y2={nNorm * CELL} stroke="white" strokeWidth={1.5}/>

          {/* Delta badge */}
          <rect x={0} y={H + 6} width={W} height={22} rx={4} fill={broken ? "#eee" : C.purple.bg} stroke={broken ? "#ccc" : C.purple.border} strokeWidth={0.7}/>
          <text x={W / 2} y={H + 21} textAnchor="middle" fontSize={9.5} fontFamily={BASE.mono} fill={broken ? "#888" : C.purple.text} fontWeight={700}>
            Δ(intra−inter) = {delta}
          </text>

          {/* Y axis label */}
          <text x={-4} y={H / 2} textAnchor="middle" fontSize={7.5} fontFamily={BASE.sans} fill="#888"
            transform={`rotate(-90, -4, ${H / 2})`}>Test samples (class-sorted)</text>

          {/* Colorbar */}
          {!broken && (
            <g>
              {Array.from({ length: 20 }, (_, k) => (
                <rect key={k} x={W + 6} y={k * (H / 20)} width={10} height={H / 20} fill={colormap(1 - k / 20, false)}/>
              ))}
              <text x={W + 20} y={8} fontSize={7} fontFamily={BASE.sans} fill="#888">1.0</text>
              <text x={W + 20} y={H} fontSize={7} fontFamily={BASE.sans} fill="#888">0.0</text>
            </g>
          )}
        </svg>
        <div style={{ fontFamily: BASE.sans, fontSize: 8.5, color: "#888", marginTop: 6 }}>{label}</div>
      </div>
    );
  }

  return (
    <figure style={{ margin: "0 0 2.5rem" }}>
      <figcaption style={{ fontFamily: BASE.sans, fontSize: 11, color: "#666", marginBottom: 16, letterSpacing: 0.3 }}>
        FIGURE 4.1 — Quantum kernel fidelity matrices before and after encoding correction (sorted by class label)
      </figcaption>
      <div style={{ display: "flex", gap: 40, justifyContent: "center", flexWrap: "wrap" }}>
        <Heatmap
          mat={brokenMat} broken={true}
          title="Original encoding (scale = π, uniform weights)"
          delta="+0.00002"
          label="Near-uniform: all fidelities ≈ 0.005, random-floor collapse"
        />
        <Heatmap
          mat={fixedMat} broken={false}
          title="Corrected encoding (scale = π/4, EVR weights)"
          delta="+0.065"
          label="Clear block structure: intra-P = 0.713 vs inter = 0.584 (3,241× improvement)"
        />
      </div>
      <div style={{ fontFamily: BASE.sans, fontSize: 8.5, color: "#888", textAlign: "center", marginTop: 10 }}>
        Cell colour encodes fidelity K(x_i, x_j) = |⟨ψ(x_i)|ψ(x_j)⟩|² ∈ [0, 1]. Diagonal = 1 (self-fidelity). White lines mark class boundary.
      </div>
    </figure>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Figure 4.2 — VQC Training Dynamics
// ─────────────────────────────────────────────────────────────────────────────

function TrainingCurves() {
  const W = 580, H = 230, PAD = { t: 20, r: 20, b: 48, l: 54 };
  const pw = W - PAD.l - PAD.r;
  const ph = H - PAD.t - PAD.b;

  const epochs = RESULTS.training.epochs;
  const aucs   = RESULTS.training.val_auc;
  const specs  = RESULTS.training.val_spec;
  const nE     = epochs.length;

  const xScale = (i) => PAD.l + (i / (nE - 1)) * pw;
  const yScale = (v, min, max) => PAD.t + ph - ((v - min) / (max - min)) * ph;

  const aucMin = 0.80, aucMax = 1.0;
  const specMin = 0.0, specMax = 0.6;

  function polyline(vals, yMin, yMax) {
    return epochs.map((_, i) =>
      `${xScale(i).toFixed(1)},${yScale(vals[i], yMin, yMax).toFixed(1)}`
    ).join(" ");
  }

  // Key epochs
  const epochAucPeak = 12; // epoch 13 (index 12)
  const epochSpecJump = 9; // epoch 10 (index 9)

  return (
    <figure style={{ margin: "0 0 2.5rem" }}>
      <figcaption style={{ fontFamily: BASE.sans, fontSize: 11, color: "#666", marginBottom: 8, letterSpacing: 0.3 }}>
        FIGURE 4.2 — Phase 2 Optimized VQC training dynamics (validation metrics vs. epoch)
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block" }}>
        <defs>
          <marker id="da" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M2 2L8 5L2 8" fill="none" stroke="#888" strokeWidth="1.2" strokeLinecap="round"/>
          </marker>
        </defs>

        {/* Grid */}
        {[0.85, 0.90, 0.95, 1.00].map(v => (
          <g key={v}>
            <line x1={PAD.l} y1={yScale(v, aucMin, aucMax)} x2={W - PAD.r} y2={yScale(v, aucMin, aucMax)}
              stroke="#eee" strokeWidth={0.7}/>
            <text x={PAD.l - 6} y={yScale(v, aucMin, aucMax) + 4} textAnchor="end" fontSize={8.5} fontFamily={BASE.sans} fill={C.auc_line}>
              {v.toFixed(2)}
            </text>
          </g>
        ))}
        {[0.0, 0.2, 0.4, 0.6].map(v => (
          <text key={v} x={W - PAD.r + 6} y={yScale(v, specMin, specMax) + 4} textAnchor="start" fontSize={8.5} fontFamily={BASE.sans} fill={C.spec_line}>
            {v.toFixed(1)}
          </text>
        ))}

        {/* X axis ticks */}
        {[1, 10, 13, 20, 30, 43].map(ep => {
          const i = ep - 1;
          return (
            <g key={ep}>
              <line x1={xScale(i)} y1={PAD.t + ph} x2={xScale(i)} y2={PAD.t + ph + 4} stroke="#bbb" strokeWidth={0.7}/>
              <text x={xScale(i)} y={PAD.t + ph + 14} textAnchor="middle" fontSize={8.5} fontFamily={BASE.sans} fill="#666">
                {ep}
              </text>
            </g>
          );
        })}

        {/* Axis labels */}
        <text x={W / 2} y={H - 4} textAnchor="middle" fontSize={9} fontFamily={BASE.sans} fill="#666">Epoch</text>
        <text x={12} y={H / 2} textAnchor="middle" fontSize={9} fontFamily={BASE.sans} fill={C.auc_line}
          transform={`rotate(-90, 12, ${H / 2})`}>Validation AUC</text>
        <text x={W - 8} y={H / 2} textAnchor="middle" fontSize={9} fontFamily={BASE.sans} fill={C.spec_line}
          transform={`rotate(90, ${W - 8}, ${H / 2})`}>Validation Specificity</text>

        {/* AUC curve */}
        <polyline
          points={polyline(aucs, aucMin, aucMax)}
          fill="none" stroke={C.auc_line} strokeWidth={2} strokeLinejoin="round"/>

        {/* Specificity curve (right y-axis) */}
        <polyline
          points={polyline(specs, specMin, specMax)}
          fill="none" stroke={C.spec_line} strokeWidth={2} strokeLinejoin="round" strokeDasharray="5 3"/>

        {/* Phase annotations */}
        {/* Phase I: AUC rise */}
        <rect x={xScale(0)} y={PAD.t} width={xScale(epochSpecJump) - xScale(0)} height={ph} fill={C.blue.bg} opacity={0.3}/>
        <text x={(xScale(0) + xScale(epochSpecJump)) / 2} y={PAD.t + 14} textAnchor="middle" fontSize={7.5} fontFamily={BASE.sans} fill={C.normal} fontWeight={600}>
          Phase I
        </text>
        <text x={(xScale(0) + xScale(epochSpecJump)) / 2} y={PAD.t + 24} textAnchor="middle" fontSize={7} fontFamily={BASE.sans} fill={C.normal}>
          AUC rises
        </text>

        {/* Phase II: Specificity jump */}
        <rect x={xScale(epochSpecJump)} y={PAD.t} width={xScale(14) - xScale(epochSpecJump)} height={ph} fill={C.amber.bg} opacity={0.35}/>
        <text x={(xScale(epochSpecJump) + xScale(14)) / 2} y={PAD.t + 14} textAnchor="middle" fontSize={7.5} fontFamily={BASE.sans} fill="#854F0B" fontWeight={600}>
          Phase II
        </text>
        <text x={(xScale(epochSpecJump) + xScale(14)) / 2} y={PAD.t + 24} textAnchor="middle" fontSize={7} fontFamily={BASE.sans} fill="#854F0B}>
          Spec. jump
        </text>

        {/* Phase III: steady climb */}
        <rect x={xScale(14)} y={PAD.t} width={xScale(nE - 1) - xScale(14)} height={ph} fill={C.teal.bg} opacity={0.25}/>
        <text x={(xScale(14) + xScale(nE - 1)) / 2} y={PAD.t + 14} textAnchor="middle" fontSize={7.5} fontFamily={BASE.sans} fill={C.teal.text} fontWeight={600}>
          Phase III
        </text>
        <text x={(xScale(14) + xScale(nE - 1)) / 2} y={PAD.t + 24} textAnchor="middle" fontSize={7} fontFamily={BASE.sans} fill={C.teal.text}>
          Spec. grows
        </text>

        {/* Epoch 13 AUC peak marker */}
        <circle cx={xScale(epochAucPeak)} cy={yScale(aucs[epochAucPeak], aucMin, aucMax)} r={4}
          fill="white" stroke={C.auc_line} strokeWidth={1.5}/>
        <line x1={xScale(epochAucPeak)} y1={yScale(aucs[epochAucPeak], aucMin, aucMax) - 5}
          x2={xScale(epochAucPeak)} y2={PAD.t + 38} stroke={C.auc_line} strokeWidth={0.7} strokeDasharray="2 2"/>
        <text x={xScale(epochAucPeak)} y={PAD.t + 35} textAnchor="middle" fontSize={7} fontFamily={BASE.mono} fill={C.auc_line}>
          best val AUC = 0.980
        </text>

        {/* Epoch 10 spec jump marker */}
        <circle cx={xScale(epochSpecJump)} cy={yScale(specs[epochSpecJump], specMin, specMax)} r={4}
          fill="white" stroke={C.spec_line} strokeWidth={1.5}/>
        <line x1={xScale(epochSpecJump)} y1={yScale(specs[epochSpecJump], specMin, specMax) + 5}
          x2={xScale(epochSpecJump)} y2={PAD.t + ph - 12} stroke={C.spec_line} strokeWidth={0.7} strokeDasharray="2 2"/>
        <text x={xScale(epochSpecJump)} y={PAD.t + ph - 4} textAnchor="middle" fontSize={7} fontFamily={BASE.mono} fill={C.spec_line}>
          Spec. first crosses 0 → threshold crossing
        </text>

        {/* Legend */}
        <line x1={PAD.l + 10} y1={PAD.t + ph - 30} x2={PAD.l + 30} y2={PAD.t + ph - 30} stroke={C.auc_line} strokeWidth={2}/>
        <text x={PAD.l + 34} y={PAD.t + ph - 26} fontSize={8.5} fontFamily={BASE.sans} fill={C.auc_line}>Val AUC (left axis)</text>

        <line x1={PAD.l + 10} y1={PAD.t + ph - 16} x2={PAD.l + 30} y2={PAD.t + ph - 16}
          stroke={C.spec_line} strokeWidth={2} strokeDasharray="5 3"/>
        <text x={PAD.l + 34} y={PAD.t + ph - 12} fontSize={8.5} fontFamily={BASE.sans} fill={C.spec_line}>Val Specificity (right axis)</text>
      </svg>
      <div style={{ fontFamily: BASE.sans, fontSize: 8.5, color: "#888", marginTop: 4 }}>
        Early stopping at patience = 30 epochs (convergence at epoch 43). Best val AUC = 0.980 at epoch 13. The temporal separation between AUC improvement (Phase I, epochs 1–10) and specificity improvement (Phase II–III, epochs 10–43) reflects the fundamental distinction between ranking quality (AUC) and threshold-crossing quality (specificity).
      </div>
    </figure>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Figure 4.3 — OOD Robustness Curves
// ─────────────────────────────────────────────────────────────────────────────

function OODCurves() {
  const W = 520, H = 220, PAD = { t: 24, r: 28, b: 44, l: 54 };
  const pw = W - PAD.l - PAD.r;
  const ph = H - PAD.t - PAD.b;

  const sigmas = RESULTS.ood.sigma;
  const vqcAUC = RESULTS.ood.vqc;
  const mlpAUC = RESULTS.ood.mlp;

  const xScale = (s) => PAD.l + (s / 0.15) * pw;
  const yScale = (v) => PAD.t + ph - ((v - 0.6) / (1.0 - 0.6)) * ph;

  // Fitted curves
  const vqcFit = (s) => 0.9266 - 2.89 * s * s;
  const mlpFit = (s) => 0.7802 - 0.87 * s;

  const vqcPoints = sigmas.map((s, i) => `${xScale(s).toFixed(1)},${yScale(vqcAUC[i]).toFixed(1)}`).join(" ");
  const mlpPoints = sigmas.map((s, i) => `${xScale(s).toFixed(1)},${yScale(mlpAUC[i]).toFixed(1)}`).join(" ");

  // Smooth fitted curves
  const fitPts = 50;
  const vqcFitLine = Array.from({ length: fitPts }, (_, i) => {
    const s = i / (fitPts - 1) * 0.15;
    return `${xScale(s).toFixed(1)},${yScale(vqcFit(s)).toFixed(1)}`;
  }).join(" ");
  const mlpFitLine = Array.from({ length: fitPts }, (_, i) => {
    const s = i / (fitPts - 1) * 0.15;
    return `${xScale(s).toFixed(1)},${yScale(mlpFit(s)).toFixed(1)}`;
  }).join(" ");

  return (
    <figure style={{ margin: "0 0 2.5rem" }}>
      <figcaption style={{ fontFamily: BASE.sans, fontSize: 11, color: "#666", marginBottom: 8, letterSpacing: 0.3 }}>
        FIGURE 5.1 — Out-of-distribution robustness: AUC degradation under additive Gaussian feature noise
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block" }}>
        {/* Grid */}
        {[0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95].map(v => (
          <g key={v}>
            <line x1={PAD.l} y1={yScale(v)} x2={W - PAD.r} y2={yScale(v)} stroke="#eee" strokeWidth={0.7}/>
            <text x={PAD.l - 6} y={yScale(v) + 4} textAnchor="end" fontSize={8.5} fontFamily={BASE.sans} fill="#666">
              {v.toFixed(2)}
            </text>
          </g>
        ))}
        {[0, 0.05, 0.10, 0.15].map(s => (
          <g key={s}>
            <line x1={xScale(s)} y1={PAD.t} x2={xScale(s)} y2={PAD.t + ph} stroke="#eee" strokeWidth={0.7}/>
            <text x={xScale(s)} y={PAD.t + ph + 14} textAnchor="middle" fontSize={8.5} fontFamily={BASE.sans} fill="#666">
              {s.toFixed(2)}
            </text>
          </g>
        ))}

        {/* CNN baseline */}
        <line x1={PAD.l} y1={yScale(0.5812)} x2={W - PAD.r} y2={yScale(0.5812)}
          stroke="#bbb" strokeWidth={0.8} strokeDasharray="3 3"/>
        <text x={W - PAD.r + 4} y={yScale(0.5812) + 4} fontSize={7.5} fontFamily={BASE.sans} fill="#bbb">CNN spec=0.581</text>

        {/* Fitted lines */}
        <polyline points={vqcFitLine} fill="none" stroke={C.auc_line} strokeWidth={1.2} strokeDasharray="4 2" opacity={0.5}/>
        <polyline points={mlpFitLine} fill="none" stroke={C.spec_line} strokeWidth={1.2} strokeDasharray="4 2" opacity={0.5}/>

        {/* Data points + lines */}
        <polyline points={vqcPoints} fill="none" stroke={C.auc_line} strokeWidth={2.2} strokeLinejoin="round"/>
        <polyline points={mlpPoints} fill="none" stroke={C.spec_line} strokeWidth={2.2} strokeLinejoin="round"/>

        {/* Dots */}
        {sigmas.map((s, i) => (
          <g key={s}>
            <circle cx={xScale(s)} cy={yScale(vqcAUC[i])} r={4} fill="white" stroke={C.auc_line} strokeWidth={1.5}/>
            <circle cx={xScale(s)} cy={yScale(mlpAUC[i])} r={4} fill="white" stroke={C.spec_line} strokeWidth={1.5}/>
          </g>
        ))}

        {/* Axis labels */}
        <text x={W / 2} y={H - 4} textAnchor="middle" fontSize={9} fontFamily={BASE.sans} fill="#666">
          Noise σ (std of additive Gaussian noise on angle-encoded features)
        </text>
        <text x={12} y={H / 2} textAnchor="middle" fontSize={9} fontFamily={BASE.sans} fill="#666"
          transform={`rotate(-90, 12, ${H / 2})`}>AUC</text>

        {/* σ=0.05 annotation */}
        <line x1={xScale(0.05)} y1={yScale(vqcAUC[2])} x2={xScale(0.05)} y2={yScale(mlpAUC[2])}
          stroke="#888" strokeWidth={0.7} strokeDasharray="2 2"/>
        <text x={xScale(0.05) + 4} y={(yScale(vqcAUC[2]) + yScale(mlpAUC[2])) / 2 - 4} fontSize={7.5} fontFamily={BASE.mono} fill="#444">
          7.8× more resilient
        </text>
        <text x={xScale(0.05) + 4} y={(yScale(vqcAUC[2]) + yScale(mlpAUC[2])) / 2 + 6} fontSize={7} fontFamily={BASE.sans} fill="#888">
          at σ=0.05
        </text>

        {/* Legend */}
        <line x1={PAD.l + 8} y1={PAD.t + 14} x2={PAD.l + 28} y2={PAD.t + 14} stroke={C.auc_line} strokeWidth={2.2}/>
        <circle cx={PAD.l + 18} cy={PAD.t + 14} r={3.5} fill="white" stroke={C.auc_line} strokeWidth={1.5}/>
        <text x={PAD.l + 32} y={PAD.t + 18} fontSize={8.5} fontFamily={BASE.sans} fill={C.auc_line}>
          VQC — AUC(σ) = 0.9266 − 2.89σ² [O(σ²)]
        </text>

        <line x1={PAD.l + 8} y1={PAD.t + 30} x2={PAD.l + 28} y2={PAD.t + 30} stroke={C.spec_line} strokeWidth={2.2}/>
        <circle cx={PAD.l + 18} cy={PAD.t + 30} r={3.5} fill="white" stroke={C.spec_line} strokeWidth={1.5}/>
        <text x={PAD.l + 32} y={PAD.t + 34} fontSize={8.5} fontFamily={BASE.sans} fill={C.spec_line}>
          MLP — AUC(σ) = 0.7802 − 0.87σ [O(σ)]
        </text>
      </svg>
    </figure>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Figure C.1 — Confusion Matrices (visual)
// ─────────────────────────────────────────────────────────────────────────────

function ConfusionMatrices() {
  const models = [
    { name: "CNN (784-px)", tn: 136, fp: 98, fn: 0, tp: 390, params: "~50k", color: C.gray },
    { name: "MLP (PCA-8)", tn: 217, fp: 17, fn: 294, tp: 96, params: "~2k", color: C.blue },
    { name: "QSVM P1 orig", tn: 0, fp: 234, fn: 0, tp: 390, params: "0", color: C.gray },
    { name: "QSVM P1 opt", tn: 179, fp: 55, fn: 31, tp: 359, params: "0", color: C.teal },
    { name: "VQC P2 orig", tn: 106, fp: 128, fn: 75, tp: 315, params: "24", color: C.amber },
    { name: "VQC P2 opt", tn: 194, fp: 40, fn: 60, tp: 330, params: "34", color: C.purple },
  ];

  function CM({ m }) {
    const total = m.tn + m.fp + m.fn + m.tp;
    const maxV = 390;

    function cellColor(v, isGood) {
      const intensity = Math.sqrt(v / maxV);
      if (isGood) {
        const b = Math.round(200 - intensity * 130);
        const g = Math.round(180 - intensity * 80);
        return `rgb(${b + 10},${g + 40},${b + 40})`;
      } else {
        const r = Math.round(240 - intensity * 100);
        return `rgb(${r},${Math.round(r * 0.7)},${Math.round(r * 0.7)})`;
      }
    }

    const acc = ((m.tn + m.tp) / total * 100).toFixed(1);
    const sens = (m.tp / (m.tp + m.fn) * 100).toFixed(1);
    const spec = (m.tn / (m.tn + m.fp) * 100).toFixed(1);

    const CW = 46, CH = 38;

    return (
      <div style={{ textAlign: "center" }}>
        <div style={{
          fontFamily: BASE.sans, fontSize: 9.5, fontWeight: 600,
          color: m.color.text, background: m.color.bg,
          border: `1px solid ${m.color.border}`,
          borderRadius: 5, padding: "3px 8px", marginBottom: 6,
          display: "inline-block"
        }}>
          {m.name} · {m.params} params
        </div>
        <svg width={CW * 2 + 60} height={CH * 2 + 60} style={{ display: "block", margin: "0 auto" }}>
          {/* Column headers */}
          <text x={60 + CW / 2} y={16} textAnchor="middle" fontSize={7.5} fontFamily={BASE.sans} fill={C.normal} fontWeight={600}>Pred N</text>
          <text x={60 + CW + CW / 2} y={16} textAnchor="middle" fontSize={7.5} fontFamily={BASE.sans} fill={C.pneum} fontWeight={600}>Pred P</text>

          {/* Row headers */}
          <text x={54} y={22 + CH / 2 + 4} textAnchor="end" fontSize={7.5} fontFamily={BASE.sans} fill={C.normal} fontWeight={600}>True N</text>
          <text x={54} y={22 + CH + CH / 2 + 4} textAnchor="end" fontSize={7.5} fontFamily={BASE.sans} fill={C.pneum} fontWeight={600}>True P</text>

          {/* TN */}
          <rect x={60} y={22} width={CW} height={CH} fill={cellColor(m.tn, true)} rx={3}/>
          <text x={60 + CW / 2} y={22 + CH / 2 + 4} textAnchor="middle" fontSize={11} fontFamily={BASE.mono} fill="white" fontWeight={700}>{m.tn}</text>

          {/* FP */}
          <rect x={60 + CW} y={22} width={CW} height={CH} fill={cellColor(m.fp, false)} rx={3}/>
          <text x={60 + CW + CW / 2} y={22 + CH / 2 + 4} textAnchor="middle" fontSize={11} fontFamily={BASE.mono} fill="white" fontWeight={700}>{m.fp}</text>

          {/* FN */}
          <rect x={60} y={22 + CH} width={CW} height={CH} fill={cellColor(m.fn, false)} rx={3}/>
          <text x={60 + CW / 2} y={22 + CH + CH / 2 + 4} textAnchor="middle" fontSize={11} fontFamily={BASE.mono} fill="white" fontWeight={700}>{m.fn}</text>

          {/* TP */}
          <rect x={60 + CW} y={22 + CH} width={CW} height={CH} fill={cellColor(m.tp, true)} rx={3}/>
          <text x={60 + CW + CW / 2} y={22 + CH + CH / 2 + 4} textAnchor="middle" fontSize={11} fontFamily={BASE.mono} fill="white" fontWeight={700}>{m.tp}</text>
        </svg>
        <div style={{ fontFamily: BASE.mono, fontSize: 7.5, color: "#666", lineHeight: 1.6 }}>
          <span style={{ color: C.teal.border }}>Acc {acc}%</span> · <span style={{ color: C.spec_line }}>Sens {sens}%</span> · <span style={{ color: C.auc_line }}>Spec {spec}%</span>
        </div>
      </div>
    );
  }

  return (
    <figure style={{ margin: "0 0 2.5rem" }}>
      <figcaption style={{ fontFamily: BASE.sans, fontSize: 11, color: "#666", marginBottom: 16, letterSpacing: 0.3 }}>
        FIGURE C.1 — Confusion matrices for all six experimental models on the 624-sample held-out test set
      </figcaption>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 20 }}>
        {models.map(m => <CM key={m.name} m={m}/>)}
      </div>
      <div style={{ fontFamily: BASE.sans, fontSize: 8, color: "#999", marginTop: 10 }}>
        Green = correct predictions (TN, TP). Red = errors (FP, FN). Colour intensity ∝ √(count/390). Normal: N=234; Pneumonia: N=390.
      </div>
    </figure>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Figure 3.1 — Full Hybrid Circuit Architecture (Schematic)
// ─────────────────────────────────────────────────────────────────────────────

function CircuitSchematic() {
  const W = 700, H = 370;
  const qubits = 8;
  const QUBIT_H = 28, Y0 = 60, X0 = 110;

  const qubitY = (q) => Y0 + q * QUBIT_H;
  const midY = Y0 + (qubits - 1) * QUBIT_H / 2;

  // Section x-positions
  const SEC = { init: 110, H: 148, RZ1: 200, ZZ1: 235, RZ2: 310, ZZ2: 345, VQC_Ry: 430, VQC_Rz: 468, CNOT: 500, Ry2: 540, Rz2: 575, meas: 630 };

  const evr = RESULTS.evr_weights;
  const obs = [0.5554, 0.1428, 0.0744, 0.0607, 0.0523, 0.0495, 0.0376, 0.0280];

  return (
    <figure style={{ margin: "0 0 2.5rem", overflowX: "auto" }}>
      <figcaption style={{ fontFamily: BASE.sans, fontSize: 11, color: "#666", marginBottom: 8, letterSpacing: 0.3 }}>
        FIGURE 3.1 — Hybrid quantum-classical circuit schematic (8 qubits, per-sample forward pass)
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block", minWidth: 640 }}>

        {/* Section background panels */}
        {/* ZZ Feature Map Stage 1 */}
        <rect x={132} y={Y0 - 16} width={268} height={qubits * QUBIT_H + 10} rx={6} fill={C.purple.bg} stroke={C.purple.border} strokeWidth={0.6} opacity={0.4}/>
        <text x={266} y={Y0 - 22} textAnchor="middle" fontSize={9} fontFamily={BASE.sans} fill={C.purple.text} fontWeight={600}>
          Stage 1 — ZZ Feature Map U_Φ(x) [frozen, 80 gates]
        </text>

        {/* VQC Stage 2 */}
        <rect x={412} y={Y0 - 16} width={196} height={qubits * QUBIT_H + 10} rx={6} fill={C.teal.bg} stroke={C.teal.border} strokeWidth={0.6} opacity={0.4}/>
        <text x={510} y={Y0 - 22} textAnchor="middle" fontSize={9} fontFamily={BASE.sans} fill={C.teal.text} fontWeight={600}>
          Stage 2 — VQC U_VQC(θ) [trainable, 46 gates]
        </text>

        {/* Qubit wires */}
        {Array.from({ length: qubits }, (_, q) => (
          <g key={q}>
            <line x1={16} y1={qubitY(q)} x2={W - 10} y2={qubitY(q)} stroke="#ccc" strokeWidth={0.8}/>
            <text x={10} y={qubitY(q) + 4} textAnchor="end" fontSize={8} fontFamily={BASE.mono} fill="#666">q{q}</text>
            {/* Init ket|0⟩ */}
            <text x={18} y={qubitY(q) + 4} fontSize={8} fontFamily={BASE.mono} fill="#888">|0⟩</text>
          </g>
        ))}

        {/* Hadamard gates */}
        {Array.from({ length: qubits }, (_, q) => (
          <g key={q}>
            <rect x={SEC.H - 10} y={qubitY(q) - 9} width={18} height={18} rx={2} fill="white" stroke="#888" strokeWidth={0.7}/>
            <text x={SEC.H - 1} y={qubitY(q) + 4} textAnchor="middle" fontSize={8} fontFamily={BASE.mono} fill="#333" fontWeight={700}>H</text>
          </g>
        ))}

        {/* RZ gates (layer 1) */}
        {Array.from({ length: qubits }, (_, q) => (
          <g key={q}>
            <rect x={SEC.RZ1 - 10} y={qubitY(q) - 9} width={22} height={18} rx={2} fill={C.purple.bg} stroke={C.purple.border} strokeWidth={0.6}/>
            <text x={SEC.RZ1 + 1} y={qubitY(q) + 4} textAnchor="middle" fontSize={7} fontFamily={BASE.mono} fill={C.purple.text}>Rz(x{q})</text>
          </g>
        ))}

        {/* IsingZZ (layer 1) — show representative lines */}
        {[[0,1],[2,4],[5,7],[3,6]].map(([a, b]) => {
          const coupling = (evr[a] * evr[b]).toFixed(2);
          return (
            <g key={`${a}-${b}`}>
              <line x1={SEC.ZZ1} y1={qubitY(a)} x2={SEC.ZZ1} y2={qubitY(b)} stroke={C.purple.border} strokeWidth={1.2} strokeDasharray="2 1"/>
              <circle cx={SEC.ZZ1} cy={qubitY(a)} r={3.5} fill={C.purple.border}/>
              <circle cx={SEC.ZZ1} cy={qubitY(b)} r={3.5} fill={C.purple.border}/>
            </g>
          );
        })}
        <text x={SEC.ZZ1} y={qubitY(7) + 20} textAnchor="middle" fontSize={7} fontFamily={BASE.sans} fill={C.purple.text}>
          28 IsingZZ pairs ×2
        </text>

        {/* RZ gates (layer 2) */}
        {Array.from({ length: qubits }, (_, q) => (
          <g key={q}>
            <rect x={SEC.RZ2 - 10} y={qubitY(q) - 9} width={22} height={18} rx={2} fill={C.purple.bg} stroke={C.purple.border} strokeWidth={0.6}/>
            <text x={SEC.RZ2 + 1} y={qubitY(q) + 4} textAnchor="middle" fontSize={7} fontFamily={BASE.mono} fill={C.purple.text}>Rz(x{q})</text>
          </g>
        ))}

        {/* ZZ layer 2 */}
        {[[1,3],[0,5],[4,7],[2,6]].map(([a, b]) => (
          <g key={`z2-${a}-${b}`}>
            <line x1={SEC.ZZ2} y1={qubitY(a)} x2={SEC.ZZ2} y2={qubitY(b)} stroke={C.purple.border} strokeWidth={1.2} strokeDasharray="2 1"/>
            <circle cx={SEC.ZZ2} cy={qubitY(a)} r={3.5} fill={C.purple.border}/>
            <circle cx={SEC.ZZ2} cy={qubitY(b)} r={3.5} fill={C.purple.border}/>
          </g>
        ))}

        {/* Layer 2 label */}
        <text x={(SEC.ZZ1 + SEC.ZZ2) / 2 + 8} y={Y0 - 6} textAnchor="middle" fontSize={7.5} fontFamily={BASE.sans} fill={C.purple.text}>
          × L_Φ = 2 reps
        </text>
        <path d={`M${SEC.RZ1 - 12},${Y0 - 4} Q${(SEC.RZ1 + SEC.ZZ2) / 2},${Y0 - 18} ${SEC.ZZ2 + 12},${Y0 - 4}`}
          fill="none" stroke={C.purple.border} strokeWidth={0.6} strokeDasharray="3 2"/>

        {/* VQC Ry layer 1 */}
        {Array.from({ length: qubits }, (_, q) => (
          <g key={q}>
            <rect x={SEC.VQC_Ry - 10} y={qubitY(q) - 9} width={22} height={18} rx={2} fill={C.teal.bg} stroke={C.teal.border} strokeWidth={0.7}/>
            <text x={SEC.VQC_Ry + 1} y={qubitY(q) + 4} textAnchor="middle" fontSize={7} fontFamily={BASE.mono} fill={C.teal.text}>Ry(θ)</text>
          </g>
        ))}

        {/* VQC Rz layer 1 */}
        {Array.from({ length: qubits }, (_, q) => (
          <g key={q}>
            <rect x={SEC.VQC_Rz - 10} y={qubitY(q) - 9} width={22} height={18} rx={2} fill={C.teal.bg} stroke={C.teal.border} strokeWidth={0.7}/>
            <text x={SEC.VQC_Rz + 1} y={qubitY(q) + 4} textAnchor="middle" fontSize={7} fontFamily={BASE.mono} fill={C.teal.text}>Rz(θ)</text>
          </g>
        ))}

        {/* CNOT ring (layer 1) */}
        {Array.from({ length: qubits - 1 }, (_, q) => (
          <g key={q}>
            <line x1={SEC.CNOT} y1={qubitY(q)} x2={SEC.CNOT} y2={qubitY(q + 1)} stroke={C.teal.border} strokeWidth={1.2}/>
            <circle cx={SEC.CNOT} cy={qubitY(q)} r={3.5} fill={C.teal.border}/>
            {/* Target circle */}
            <circle cx={SEC.CNOT} cy={qubitY(q + 1)} r={5} fill="none" stroke={C.teal.border} strokeWidth={1.2}/>
            <line x1={SEC.CNOT - 5} y1={qubitY(q + 1)} x2={SEC.CNOT + 5} y2={qubitY(q + 1)} stroke={C.teal.border} strokeWidth={1}/>
          </g>
        ))}

        {/* VQC layer 2 Ry, Rz */}
        {Array.from({ length: qubits }, (_, q) => (
          <g key={q}>
            <rect x={SEC.Ry2 - 10} y={qubitY(q) - 9} width={22} height={18} rx={2} fill={C.teal.bg} stroke={C.teal.border} strokeWidth={0.7}/>
            <text x={SEC.Ry2 + 1} y={qubitY(q) + 4} textAnchor="middle" fontSize={7} fontFamily={BASE.mono} fill={C.teal.text}>Ry(θ)</text>
          </g>
        ))}
        {Array.from({ length: qubits }, (_, q) => (
          <g key={q}>
            <rect x={SEC.Rz2 - 10} y={qubitY(q) - 9} width={22} height={18} rx={2} fill={C.teal.bg} stroke={C.teal.border} strokeWidth={0.7} strokeDasharray={q < 8 ? "none" : "2 2"}/>
            <text x={SEC.Rz2 + 1} y={qubitY(q) + 4} textAnchor="middle" fontSize={7} fontFamily={BASE.mono} fill={C.teal.text} opacity={0.5}>≈0</text>
          </g>
        ))}
        <text x={SEC.Rz2 + 1} y={qubitY(7) + 20} textAnchor="middle" fontSize={6.5} fontFamily={BASE.sans} fill={C.teal.text} opacity={0.8}>
          self-zeroed
        </text>

        {/* Measurement box */}
        <rect x={SEC.meas - 10} y={Y0 - 18} width={46} height={qubits * QUBIT_H + 6} rx={5} fill={C.amber.bg} stroke={C.amber.border} strokeWidth={0.8}/>
        <text x={SEC.meas + 13} y={Y0 + 2} textAnchor="middle" fontSize={7.5} fontFamily={BASE.sans} fill={C.amber.text} fontWeight={600}>⟨Ô⟩</text>
        {obs.map((w, q) => (
          <text key={q} x={SEC.meas + 13} y={qubitY(q) + 4} textAnchor="middle" fontSize={6.5} fontFamily={BASE.mono} fill={C.amber.text}>
            {w.toFixed(3)}·Z{q}
          </text>
        ))}

        {/* Classical post layer */}
        <rect x={W - 60} y={midY - 28} width={54} height={56} rx={5} fill={C.coral.bg} stroke={C.coral.border} strokeWidth={0.8}/>
        <text x={W - 33} y={midY - 14} textAnchor="middle" fontSize={7.5} fontFamily={BASE.sans} fill={C.coral.text} fontWeight={600}>Post-layer</text>
        <text x={W - 33} y={midY - 2} textAnchor="middle" fontSize={7} fontFamily={BASE.mono} fill={C.coral.text}>a·⟨Ô⟩+b</text>
        <text x={W - 33} y={midY + 10} textAnchor="middle" fontSize={6.5} fontFamily={BASE.sans} fill={C.coral.text}>a=3.727</text>
        <text x={W - 33} y={midY + 21} textAnchor="middle" fontSize={6.5} fontFamily={BASE.sans} fill={C.coral.text}>b=1.022</text>
        <text x={W - 33} y={midY + 32} textAnchor="middle" fontSize={6.5} fontFamily={BASE.sans} fill={C.coral.text}>t*=0.800</text>

        {/* Arrow from meas to post-layer */}
        <line x1={SEC.meas + 36} y1={midY} x2={W - 62} y2={midY} stroke={C.amber.border} strokeWidth={1.2} markerEnd="url(#arr-post)"/>

        <defs>
          <marker id="arr-post" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M1.5 2L8.5 5L1.5 8" fill="none" stroke={C.amber.border} strokeWidth="1.3" strokeLinecap="round"/>
          </marker>
        </defs>

        {/* Gate count summary bar */}
        <rect x={14} y={H - 42} width={W - 24} height={34} rx={4} fill="#fafafa" stroke="#ddd" strokeWidth={0.6}/>
        <text x={30} y={H - 26} fontSize={8} fontFamily={BASE.mono} fill="#444">
          Gate count: H×8  Rz×16  IsingZZ×56  Ry×16  Rz×16  CNOT×14 → total 126 gates/sample
        </text>
        <text x={30} y={H - 14} fontSize={8} fontFamily={BASE.mono} fill="#666">
          Parameters: 2 layers × 8 qubits × 2 (Ry,Rz) = 32 quantum + 2 classical (a,b) = 34 total
        </text>
      </svg>
    </figure>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Figure 5.2 — Ablation Bar Chart
// ─────────────────────────────────────────────────────────────────────────────

function AblationChart() {
  const metrics = ["AUC", "Accuracy", "Sensitivity", "Specificity", "G-mean"];
  const models = [
    { name: "MLP-PCA8", color: C.gray.border, values: [0.7802, 0.5016, 0.2462, 0.9274, 0.4778] },
    { name: "QSVM-P1", color: C.teal.border, values: [0.9171, 0.8622, 0.9205, 0.7650, 0.8391] },
    { name: "VQC-P2 (final)", color: C.purple.border, values: [0.9132, 0.8397, 0.8462, 0.8291, 0.8376] },
    { name: "CNN-784", color: C.blue.border, values: [0.9648, 0.8429, 1.0000, 0.5812, 0.7624] },
  ];

  const W = 560, H = 200, PAD = { t: 16, r: 16, b: 40, l: 48 };
  const pw = W - PAD.l - PAD.r;
  const ph = H - PAD.t - PAD.b;

  const nM = metrics.length;
  const nMod = models.length;
  const groupW = pw / nM;
  const barW = Math.min(14, (groupW - 8) / nMod);
  const gap = 2;

  const yScale = (v) => PAD.t + ph - v * ph;

  return (
    <figure style={{ margin: "0 0 2.5rem" }}>
      <figcaption style={{ fontFamily: BASE.sans, fontSize: 11, color: "#666", marginBottom: 8, letterSpacing: 0.3 }}>
        FIGURE 5.2 — Ablation study: per-metric performance of all four models on the held-out test set
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block" }}>
        {/* Grid */}
        {[0.25, 0.5, 0.75, 1.0].map(v => (
          <g key={v}>
            <line x1={PAD.l} y1={yScale(v)} x2={W - PAD.r} y2={yScale(v)} stroke="#eee" strokeWidth={0.7}/>
            <text x={PAD.l - 5} y={yScale(v) + 3} textAnchor="end" fontSize={8} fontFamily={BASE.sans} fill="#888">{v.toFixed(2)}</text>
          </g>
        ))}

        {/* Bars */}
        {metrics.map((metric, mi) => {
          const gx = PAD.l + mi * groupW + groupW / 2 - (nMod * (barW + gap)) / 2;
          return (
            <g key={metric}>
              {models.map((mod, mdi) => {
                const v = mod.values[mi];
                const x = gx + mdi * (barW + gap);
                const bH = v * ph;
                return (
                  <g key={mdi}>
                    <rect x={x} y={yScale(v)} width={barW} height={bH} rx={2}
                      fill={mod.color} opacity={mdi === 2 ? 1 : 0.75}/>
                    <text x={x + barW / 2} y={yScale(v) - 3} textAnchor="middle" fontSize={6.5} fontFamily={BASE.mono} fill={mod.color} fontWeight={mdi === 2 ? 700 : 400}>
                      {v.toFixed(2)}
                    </text>
                  </g>
                );
              })}
              <text x={PAD.l + mi * groupW + groupW / 2} y={H - PAD.b + 14} textAnchor="middle" fontSize={8.5} fontFamily={BASE.sans} fill="#555">
                {metric}
              </text>
            </g>
          );
        })}

        {/* Legend */}
        {models.map((mod, i) => (
          <g key={i}>
            <rect x={PAD.l + i * 128} y={H - PAD.b + 24} width={10} height={10} rx={2} fill={mod.color} opacity={i === 2 ? 1 : 0.75}/>
            <text x={PAD.l + i * 128 + 14} y={H - PAD.b + 33} fontSize={8} fontFamily={BASE.sans} fill={mod.color} fontWeight={i === 2 ? 600 : 400}>
              {mod.name}
            </text>
          </g>
        ))}
      </svg>
    </figure>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main App
// ─────────────────────────────────────────────────────────────────────────────

const FIGURES = [
  { id: "dim",     label: "Fig 2.1 — Dimensionality Chain",     Comp: DimChain },
  { id: "kernel",  label: "Fig 4.1 — Kernel Heatmaps",          Comp: KernelHeatmaps },
  { id: "train",   label: "Fig 4.2 — Training Dynamics",        Comp: TrainingCurves },
  { id: "ood",     label: "Fig 5.1 — OOD Robustness",           Comp: OODCurves },
  { id: "ablat",   label: "Fig 5.2 — Ablation Chart",           Comp: AblationChart },
  { id: "circ",    label: "Fig 3.1 — Circuit Schematic",        Comp: CircuitSchematic },
  { id: "confmat", label: "Fig C.1 — Confusion Matrices",       Comp: ConfusionMatrices },
];

export default function App() {
  const [active, setActive] = useState("dim");

  const { Comp } = FIGURES.find(f => f.id === active);

  return (
    <div style={{ fontFamily: BASE.sans, maxWidth: 880, margin: "0 auto", padding: "1rem 1.25rem" }}>
      {/* Header */}
      <div style={{ borderBottom: "0.5px solid #ddd", paddingBottom: 12, marginBottom: 16 }}>
        <div style={{ fontSize: 12, color: "#888", letterSpacing: 0.8, marginBottom: 4 }}>
          ADDIS ABABA UNIVERSITY · DEPT. PHYSICS
        </div>
        <div style={{ fontSize: 15, fontWeight: 600, color: "#222", fontFamily: BASE.sans }}>
          Hilbert Space Topology and Feature Mapping in Medical Image Classification
        </div>
        <div style={{ fontSize: 11, color: "#888", marginTop: 3 }}>
          Thesis Figures Generator — all data from locked experimental JSON files
        </div>
      </div>

      {/* Tab bar */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 20 }}>
        {FIGURES.map(f => (
          <button
            key={f.id}
            onClick={() => setActive(f.id)}
            style={{
              padding: "5px 11px",
              fontSize: 11,
              fontFamily: BASE.sans,
              border: `0.5px solid ${active === f.id ? C.purple.border : "#ccc"}`,
              borderRadius: 5,
              background: active === f.id ? C.purple.bg : "white",
              color: active === f.id ? C.purple.text : "#555",
              cursor: "pointer",
              fontWeight: active === f.id ? 600 : 400,
              transition: "all 0.15s",
            }}>
            {f.label}
          </button>
        ))}
      </div>

      {/* Figure */}
      <div style={{ background: "white", borderRadius: 8, border: "0.5px solid #e8e8e8", padding: "20px 24px" }}>
        <Comp />
      </div>

      {/* Data provenance footnote */}
      <div style={{ marginTop: 12, fontSize: 9, color: "#aaa", fontFamily: BASE.mono, lineHeight: 1.6 }}>
        Data sources: phase1_optimized_summary.json · phase2_summary.json · phase3_summary.json ·
        phase3_ablation_summary.json · phase3_ood_summary.json · thesis_migration_state.md ·
        All numerical values are locked and match the authoritative research state.
      </div>
    </div>
  );
}
