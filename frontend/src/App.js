import { useState, useEffect, useRef, useCallback } from "react";
import {
  trainModel, predictChurn,
  getHealth, getResults, getFeatureImportance,
  getStats, getTopRisk, getAnalytics, getMlflowRuns,
} from "./services/api";

/* ─────────────────────────────────────────────
   COLOUR TOKENS
───────────────────────────────────────────── */
const C = {
  bg:"#f5f2ed", bg2:"#ede9e2", bg3:"#e4dfd6",
  surface:"#ffffff", surface2:"#faf8f5",
  ink:"#1a1714", ink2:"#3d3830", muted:"#7a7268", hint:"#afa99f",
  border:"#dbd5cc", border2:"#c8c1b5",
  red:"#c0392b", redM:"#e74c3c", redL:"#fdf0ef", redB:"#f5c6c2",
  amber:"#c0760a", amberM:"#e8920e", amberL:"#fef8ee", amberB:"#f5dba8",
  green:"#1e7e4a", greenM:"#27ae60", greenL:"#edf7f1", greenB:"#a8dfc0",
  blue:"#1a4f8a", blueM:"#2980b9", blueL:"#edf4fb", blueB:"#a8d0ef",
  purple:"#7b2d8b", purpleL:"#f5eef8", purpleB:"#d7bde2",
  orange:"#d35400", orangeL:"#fdf0e6", orangeB:"#f5cba7",
  topbar:"#1a1714",
};

/* ─────────────────────────────────────────────
   MODEL CONFIGS
───────────────────────────────────────────── */
const ALL_MODELS = [
  { id:"random_forest",       short:"RF",   label:"Random Forest",          color:C.red,    colorL:C.redL,    colorB:C.redB,
    family:"ensemble", badge:"best",
    desc:"Ensemble de Decision Trees par bagging. Meilleur compromis accuracy/recall.",
    params:[
      {name:"n_estimators",     label:"n_estimators",     type:"range", min:10, max:500, default:100, step:10},
      {name:"max_depth",        label:"max_depth",         type:"range", min:1,  max:30,  default:10,  step:1},
      {name:"min_samples_split",label:"min_samples_split", type:"range", min:2,  max:20,  default:5,   step:1},
    ]},
  { id:"xgboost",             short:"XGB",  label:"XGBoost",                color:C.purple, colorL:C.purpleL, colorB:C.purpleB,
    family:"boosting", badge:"boosting",
    desc:"Gradient Boosting optimisé. Très performant sur données tabulaires déséquilibrées.",
    params:[
      {name:"n_estimators", label:"n_estimators",  type:"range", min:50,   max:500, default:200,  step:10},
      {name:"max_depth",    label:"max_depth",      type:"range", min:2,    max:10,  default:5,    step:1},
      {name:"learning_rate",label:"learning_rate",  type:"range", min:0.01, max:0.3, default:0.05, step:0.01},
      {name:"subsample",    label:"subsample",      type:"range", min:0.5,  max:1.0, default:0.8,  step:0.05},
    ]},
  { id:"adaboost",            short:"ADA",  label:"AdaBoost",               color:C.orange, colorL:C.orangeL, colorB:C.orangeB,
    family:"boosting", badge:"boosting",
    desc:"Adaptive Boosting. Combine des arbres faibles en un classifieur fort.",
    params:[
      {name:"n_estimators",   label:"n_estimators",  type:"range", min:10,   max:300, default:100, step:10},
      {name:"learning_rate",  label:"learning_rate",  type:"range", min:0.01, max:2.0, default:0.5, step:0.01},
    ]},
  { id:"svm",                 short:"SVM",  label:"Support Vector Machine", color:C.blue,   colorL:C.blueL,   colorB:C.blueB,
    family:"kernel",
    desc:"Maximise la marge de séparation. Robuste sur données de haute dimension.",
    params:[
      {name:"C",     label:"C",      type:"range",  min:0.01, max:10,  default:1,   step:0.01},
      {name:"kernel",label:"kernel", type:"select", options:["rbf","linear","poly","sigmoid"], default:"rbf"},
    ]},
  { id:"logistic_regression", short:"LR",   label:"Logistic Regression",    color:C.green,  colorL:C.greenL,  colorB:C.greenB,
    family:"linear",
    desc:"Modèle linéaire probabiliste. Rapide, interprétable, bonne baseline.",
    params:[
      {name:"C",       label:"C",       type:"range", min:0.001, max:10,   default:1,    step:0.001},
      {name:"max_iter",label:"max_iter",type:"range", min:100,   max:2000, default:1000, step:100},
    ]},
  { id:"knn",                 short:"KNN",  label:"K-Nearest Neighbors",    color:C.amber,  colorL:C.amberL,  colorB:C.amberB,
    family:"instance",
    desc:"Classification par voisins les plus proches. Simple et intuitif.",
    params:[
      {name:"n_neighbors",label:"n_neighbors",type:"range",  min:1, max:30, default:7,        step:1},
      {name:"weights",    label:"weights",    type:"select", options:["uniform","distance"],   default:"distance"},
    ]},
];

/* ─────────────────────────────────────────────
   FEATURE BUILDER for /predict
───────────────────────────────────────────── */
function buildFeatures(tenure, monthly, contract, internet, hasSec, hasTech) {
  return {
    tenure,
    MonthlyCharges: monthly,
    TotalCharges: Math.round(tenure * monthly),
    "Contract_One year":       contract === "1" ? 1 : 0,
    "Contract_Two year":       contract === "2" ? 1 : 0,
    "InternetService_Fiber optic": internet === "f" ? 1 : 0,
    "InternetService_No":      internet === "n" ? 1 : 0,
    "OnlineSecurity_Yes":      hasSec  ? 1 : 0,
    "TechSupport_Yes":         hasTech ? 1 : 0,
  };
}

/* ─────────────────────────────────────────────
   PRIMITIVE COMPONENTS
───────────────────────────────────────────── */
function Panel({ children, title, sub, style = {} }) {
  return (
    <div style={{ background: C.surface, borderRadius: 9, border: `0.5px solid ${C.border}`, padding: 14, ...style }}>
      {title && (
        <div style={{ fontSize: 12, fontWeight: 500, color: C.ink, marginBottom: 10, display: "flex",
          justifyContent: "space-between", alignItems: "center", borderBottom: `0.5px solid ${C.bg2}`, paddingBottom: 8 }}>
          {title}
          {sub && <span style={{ fontSize: 11, fontWeight: 400, color: C.muted }}>{sub}</span>}
        </div>
      )}
      {children}
    </div>
  );
}

function KCard({ label, value, sub, pct, color, badge }) {
  return (
    <div style={{ background: C.surface, borderRadius: 9, border: `0.5px solid ${C.border}`, padding: "12px 14px" }}>
      <div style={{ fontSize: 11, color: C.muted, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6,
        display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        {label}
        {badge && <span style={{ fontSize: 10, padding: "1px 7px", borderRadius: 10,
          background: badge.bg, color: badge.c, border: `0.5px solid ${badge.b}` }}>{badge.text}</span>}
      </div>
      <div style={{ fontSize: 24, fontWeight: 500, color: color || C.ink, lineHeight: 1 }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: C.muted, marginTop: 5 }}>{sub}</div>}
      {pct != null && (
        <div style={{ height: 3, background: C.bg2, borderRadius: 2, marginTop: 8, overflow: "hidden" }}>
          <div style={{ width: `${pct}%`, height: "100%", background: color || C.border, borderRadius: 2 }} />
        </div>
      )}
    </div>
  );
}

function HBar({ name, val, max = 100, color, nameW = 120 }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
      <div style={{ fontSize: 11, color: C.ink2, flexShrink: 0, width: nameW,
        overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{name}</div>
      <div style={{ flex: 1, background: C.bg2, borderRadius: 2, height: 6, overflow: "hidden" }}>
        <div style={{ width: `${(val / max) * 100}%`, height: "100%", background: color, borderRadius: 2, transition: "width 0.5s" }} />
      </div>
      <div style={{ fontSize: 11, color: C.muted, width: 40, textAlign: "right", flexShrink: 0 }}>
        {typeof val === "number" && val < 1 ? val.toFixed(3) : val + "%"}
      </div>
    </div>
  );
}

function Badge({ text, variant = "gray" }) {
  const V = {
    red:    { bg: C.redL,    c: "#791F1F", b: C.redB },
    green:  { bg: C.greenL,  c: "#27500A", b: C.greenB },
    blue:   { bg: C.blueL,   c: "#0C447C", b: C.blueB },
    amber:  { bg: C.amberL,  c: "#633806", b: C.amberB },
    purple: { bg: C.purpleL, c: "#6c2680", b: C.purpleB },
    orange: { bg: C.orangeL, c: "#b84800", b: C.orangeB },
    gray:   { bg: C.bg2,     c: "#444441", b: C.border },
  };
  const v = V[variant] || V.gray;
  return (
    <span style={{ display: "inline-flex", alignItems: "center", padding: "2px 8px", borderRadius: 10,
      fontSize: 10, fontWeight: 500, background: v.bg, color: v.c, border: `0.5px solid ${v.b}` }}>{text}</span>
  );
}

function Spinner() {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, color: C.muted, fontSize: 12, padding: 20 }}>
      <div style={{ width: 16, height: 16, border: `2px solid ${C.border}`, borderTopColor: C.red,
        borderRadius: "50%", animation: "spin 0.8s linear infinite" }} />
      Chargement...
    </div>
  );
}

function ApiError({ msg, retry }) {
  return (
    <div style={{ background: C.redL, border: `0.5px solid ${C.redB}`, borderRadius: 7,
      padding: "10px 14px", fontSize: 12, color: "#791F1F", display: "flex", gap: 8, alignItems: "center" }}>
      <span>⚠ {msg}</span>
      {retry && (
        <button onClick={retry} style={{ marginLeft: "auto", padding: "3px 10px", borderRadius: 5,
          background: C.red, border: "none", color: "#fff", fontSize: 11, cursor: "pointer", fontFamily: "inherit" }}>
          Réessayer
        </button>
      )}
    </div>
  );
}

/* ─────────────────────────────────────────────
   CHART.JS WRAPPER
───────────────────────────────────────────── */
function ChartBox({ id, setup, height = 180 }) {
  const ref = useRef(null);
  const chart = useRef(null);
  useEffect(() => {
    if (!window.Chart || !ref.current) return;
    if (chart.current) { chart.current.destroy(); chart.current = null; }
    chart.current = setup(ref.current);
    return () => { if (chart.current) { chart.current.destroy(); chart.current = null; } };
  }, [id]);
  return <div style={{ position: "relative", height, width: "100%" }}><canvas ref={ref} id={id} /></div>;
}

const CD = {
  responsive: true, maintainAspectRatio: false,
  plugins: { legend: { display: false } },
  scales: {
    x: { grid: { color: "rgba(0,0,0,0.05)" }, ticks: { font: { size: 10 }, color: C.muted } },
    y: { grid: { color: "rgba(0,0,0,0.05)" }, ticks: { font: { size: 10 }, color: C.muted } },
  },
};

/* ═══════════════════════════════════════════════════════════════
   PAGE: OVERVIEW
═══════════════════════════════════════════════════════════════ */
function PageOverview({ liveMetrics, apiStatus, stats }) {
  const models  = Object.keys(liveMetrics);
  const bestId  = models.length
    ? models.reduce((a, b) => liveMetrics[a].accuracy > liveMetrics[b].accuracy ? a : b)
    : null;
  const bestM   = bestId ? liveMetrics[bestId] : {};
  const bestCfg = ALL_MODELS.find(m => m.id === bestId) || ALL_MODELS[0];
  const runCount = models.length;

  const barLabels = ALL_MODELS.map(m => m.short);
  const barData   = ALL_MODELS.map(m => (liveMetrics[m.id]?.accuracy || 0) * 100);
  const barColors = ALL_MODELS.map(m => m.color);

  const total    = stats?.total    ?? 7043;
  const churned  = stats?.churned  ?? 1869;
  const churnPct = stats?.churn_rate != null ? (stats.churn_rate * 100).toFixed(1) : "27.3";
  const trainN   = stats?.train    ?? 5634;
  const testN    = stats?.test     ?? 1409;
  const featN    = stats?.features ?? 30;

  return (
    <div>
      {!apiStatus && (
        <div style={{ marginBottom: 12 }}>
          <ApiError msg="Backend Flask non connecté — lance 'python app.py' puis 'python train.py'." />
        </div>
      )}

      {/* KPI row */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,minmax(0,1fr))", gap: 10, marginBottom: 14 }}>
        <KCard label="Total customers" value={total.toLocaleString()}
          sub="Dataset Telco nettoyé" pct={100} color={C.ink}
          badge={{ text: "dataset", bg: C.bg2, c: "#444441", b: C.border }} />
        <KCard label="Churned" value={churned.toLocaleString()}
          sub="Segment à haut risque" pct={parseFloat(churnPct)} color={C.red}
          badge={{ text: churnPct + "%", bg: C.redL, c: "#791F1F", b: C.redB }} />
        <KCard label="Best model"
          value={bestId ? ((bestM.accuracy || 0) * 100).toFixed(1) + "%" : "—"}
          sub={`${bestCfg.label} · AUC ${(bestM.roc_auc || 0).toFixed(3)}`}
          pct={(bestM.accuracy || 0) * 100} color={C.green}
          badge={{ text: bestCfg.short, bg: C.greenL, c: "#27500A", b: C.greenB }} />
        <KCard label="MLflow runs"
          value={runCount || "—"}
          sub={runCount ? `${runCount} modèles entraînés` : "Pas encore entraîné"}
          pct={Math.min(runCount * 10, 100)} color={C.blue}
          badge={{ text: "tracked", bg: C.blueL, c: "#0C447C", b: C.blueB }} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 12, marginBottom: 12 }}>
        <Panel title="Accuracy — 6 algorithmes" sub={models.length ? "live · test set" : "en attente d'entraînement"}>
          {models.length > 0
            ? <ChartBox id="ov-acc" height={185} setup={c => new window.Chart(c, {
                type: "bar",
                data: { labels: barLabels, datasets: [{ data: barData, backgroundColor: barColors, borderRadius: 5, borderSkipped: false }] },
                options: { ...CD, scales: { ...CD.scales, y: { ...CD.scales.y, min: 60, max: 90, ticks: { ...CD.scales.y.ticks, callback: v => v + "%" } } } },
              })} />
            : <div style={{ height: 185, display: "flex", alignItems: "center", justifyContent: "center", color: C.muted, fontSize: 12 }}>
                Entraîne un modèle pour voir les vraies métriques
              </div>
          }
          <div style={{ display: "flex", gap: 10, marginTop: 8, flexWrap: "wrap" }}>
            {ALL_MODELS.map(m => (
              <span key={m.id} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 10, color: C.muted }}>
                <span style={{ width: 8, height: 8, borderRadius: 2, background: m.color, display: "inline-block" }} />
                {m.short}
                {liveMetrics[m.id] && (
                  <span style={{ color: m.color, fontWeight: 500 }}>
                    {((liveMetrics[m.id].accuracy || 0) * 100).toFixed(1)}%
                  </span>
                )}
              </span>
            ))}
          </div>
        </Panel>

        <Panel title="Dataset">
          <div style={{ display: "flex", justifyContent: "center", padding: "6px 0 10px" }}>
            <svg width={110} height={110} viewBox="0 0 110 110">
              {[{ p: parseFloat(churnPct), c: C.red }, { p: 100 - parseFloat(churnPct), c: C.bg3 }].reduce((acc, s, i) => {
                const circ = 2 * Math.PI * 38;
                const dash = circ * (s.p / 100);
                const prev = acc.offset;
                acc.els.push(
                  <circle key={i} cx={55} cy={55} r={38} fill="none" stroke={s.c}
                    strokeWidth={14} strokeDasharray={`${dash} ${circ - dash}`}
                    strokeDashoffset={circ * 0.25 - prev} strokeLinecap="round" />
                );
                acc.offset += dash;
                return acc;
              }, { els: [], offset: 0 }).els}
              <text x={55} y={51} textAnchor="middle" fontSize={16} fontWeight={500} fill={C.ink}>{churnPct}%</text>
              <text x={55} y={65} textAnchor="middle" fontSize={9} fill={C.muted}>churn rate</text>
            </svg>
          </div>
          {[["Train set", trainN.toLocaleString()], ["Test set", testN.toLocaleString()], ["Features", featN]].map(([k, v]) => (
            <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "4px 0",
              borderBottom: `0.5px solid ${C.bg2}`, fontSize: 11 }}>
              <span style={{ color: C.muted }}>{k}</span>
              <span style={{ fontWeight: 500, color: C.ink }}>{v}</span>
            </div>
          ))}
        </Panel>
      </div>

      {models.length > 0 && (
        <Panel title="Métriques live — tous les modèles entraînés" sub="from /results">
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, tableLayout: "fixed" }}>
            <thead>
              <tr>
                {["Modèle", "Accuracy", "Precision", "Recall", "F1", "AUC", "Train Acc"].map(h => (
                  <th key={h} style={{ padding: "5px 8px", textAlign: "left", color: C.muted, fontWeight: 500,
                    fontSize: 10, borderBottom: `0.5px solid ${C.border}`, textTransform: "uppercase" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ALL_MODELS.filter(m => liveMetrics[m.id]).map(m => {
                const mt = liveMetrics[m.id];
                const isBest = m.id === bestId;
                return (
                  <tr key={m.id} style={{ background: isBest ? m.colorL : "transparent" }}>
                    <td style={{ padding: "5px 8px", fontWeight: isBest ? 500 : 400, color: m.color,
                      borderBottom: `0.5px solid ${C.bg2}` }}>
                      {m.label}{isBest && <span style={{ marginLeft: 5, fontSize: 9 }}>★</span>}
                    </td>
                    {[mt.accuracy, mt.precision, mt.recall, mt.f1, mt.roc_auc, mt.train_accuracy].map((v, i) => (
                      <td key={i} style={{ padding: "5px 8px", color: C.ink2, borderBottom: `0.5px solid ${C.bg2}` }}>
                        {v != null ? (v * 100).toFixed(1) + "%" : "—"}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Panel>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   PAGE: ANALYTICS
═══════════════════════════════════════════════════════════════ */
function PageAnalytics() {
  const [data,    setData]    = useState(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(null);

  useEffect(() => {
    setLoading(true);
    getAnalytics()
      .then(r => { setData(r.data); setLoading(false); })
      .catch(() => { setError("Analytics unavailable."); setLoading(false); });
  }, []);

  if (loading) return <Spinner />;
  if (error)   return <ApiError msg={error} />;
  if (!data)   return null;

  const contractData = data.by_contract || {};
  const internetData = data.by_internet || {};
  const paymentData  = data.by_payment  || {};

  const maxChurn = Math.max(
    ...Object.values(contractData),
    ...Object.values(internetData),
    ...Object.values(paymentData),
    0.01,
  ) * 100;

  const contractColors = { "Month-to-month": C.red, "One year": C.amber, "Two year": C.green };
  const internetColors = { "Fiber optic": C.purple, "DSL": C.blue, "No": C.green };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
      <Panel title="Churn rate par contrat" sub="from /analytics">
        {Object.entries(contractData).map(([k, v]) => (
          <HBar key={k} name={k} val={parseFloat((v * 100).toFixed(1))}
            max={maxChurn} color={contractColors[k] || C.blue} nameW={160} />
        ))}
        {Object.keys(contractData).length === 0 && (
          <div style={{ color: C.muted, fontSize: 12, padding: 10 }}>Aucune donnée disponible.</div>
        )}
      </Panel>

      <Panel title="Churn rate par service internet" sub="from /analytics">
        {Object.entries(internetData).map(([k, v]) => (
          <HBar key={k} name={k} val={parseFloat((v * 100).toFixed(1))}
            max={maxChurn} color={internetColors[k] || C.purple} nameW={140} />
        ))}
        {Object.keys(internetData).length === 0 && (
          <div style={{ color: C.muted, fontSize: 12, padding: 10 }}>Aucune donnée disponible.</div>
        )}
      </Panel>

      <Panel title="Churn rate par méthode de paiement" sub="from /analytics" style={{ gridColumn: "1 / -1" }}>
        {Object.entries(paymentData).map(([k, v], i) => (
          <HBar key={k} name={k} val={parseFloat((v * 100).toFixed(1))}
            max={maxChurn} color={[C.red, C.blue, C.green, C.amber][i % 4]} nameW={240} />
        ))}
        {Object.keys(paymentData).length === 0 && (
          <div style={{ color: C.muted, fontSize: 12, padding: 10 }}>Aucune donnée disponible.</div>
        )}
      </Panel>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   PAGE: MODELS
═══════════════════════════════════════════════════════════════ */
function PageModels({ liveMetrics }) {
  const models = Object.keys(liveMetrics);
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(6,minmax(0,1fr))", gap: 8, marginBottom: 14 }}>
        {ALL_MODELS.map(m => {
          const mt = liveMetrics[m.id];
          return (
            <div key={m.id} style={{ background: C.surface, borderRadius: 9,
              border: `0.5px solid ${mt ? m.colorB : C.border}`, padding: "10px 12px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
                <span style={{ fontSize: 10, color: C.muted, textTransform: "uppercase", letterSpacing: "0.06em" }}>{m.short}</span>
                {mt
                  ? <Badge text={(mt.accuracy * 100).toFixed(1) + "%"} variant={m.id === "xgboost" ? "purple" : m.id === "adaboost" ? "orange" : "gray"} />
                  : <Badge text="—" variant="gray" />
                }
              </div>
              <div style={{ fontSize: 20, fontWeight: 500, color: mt ? m.color : C.hint, lineHeight: 1 }}>
                {mt ? (mt.f1 || 0).toFixed(3) : "—"}
              </div>
              <div style={{ fontSize: 10, color: C.muted, marginTop: 4 }}>
                {mt ? `AUC: ${mt.roc_auc?.toFixed(3) || "—"}` : "Not trained"}
              </div>
              {mt && (
                <div style={{ height: 3, background: C.bg2, borderRadius: 2, marginTop: 6, overflow: "hidden" }}>
                  <div style={{ width: `${(mt.accuracy || 0) * 100}%`, height: "100%", background: m.color, borderRadius: 2 }} />
                </div>
              )}
            </div>
          );
        })}
      </div>

      {models.length > 0 && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <Panel title="Accuracy comparison" sub="test set · 1 409 examples">
            <ChartBox id="md-acc" height={200} setup={c => new window.Chart(c, {
              type: "bar",
              data: {
                labels: ALL_MODELS.filter(m => liveMetrics[m.id]).map(m => m.short),
                datasets: [{
                  data: ALL_MODELS.filter(m => liveMetrics[m.id]).map(m => (liveMetrics[m.id].accuracy || 0) * 100),
                  backgroundColor: ALL_MODELS.filter(m => liveMetrics[m.id]).map(m => m.color),
                  borderRadius: 4, borderSkipped: false,
                }],
              },
              options: { ...CD, scales: { ...CD.scales, y: { ...CD.scales.y, min: 60, max: 90, ticks: { ...CD.scales.y.ticks, callback: v => v + "%" } } } },
            })} />
          </Panel>
          <Panel title="F1 Score comparison">
            <ChartBox id="md-f1" height={200} setup={c => new window.Chart(c, {
              type: "bar",
              data: {
                labels: ALL_MODELS.filter(m => liveMetrics[m.id]).map(m => m.short),
                datasets: [{
                  data: ALL_MODELS.filter(m => liveMetrics[m.id]).map(m => (liveMetrics[m.id].f1 || 0) * 100),
                  backgroundColor: ALL_MODELS.filter(m => liveMetrics[m.id]).map(m => m.color),
                  borderRadius: 4, borderSkipped: false,
                }],
              },
              options: { ...CD, scales: { ...CD.scales, y: { ...CD.scales.y, min: 0, max: 80, ticks: { ...CD.scales.y.ticks, callback: v => v + "%" } } } },
            })} />
          </Panel>
        </div>
      )}

      {models.length === 0 && (
        <Panel title="Aucun modèle entraîné">
          <div style={{ padding: 20, textAlign: "center", color: C.muted, fontSize: 13 }}>
            Va dans l'onglet <strong>Train</strong> et entraîne un modèle pour voir les métriques réelles.
          </div>
        </Panel>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   PAGE: FEATURES
═══════════════════════════════════════════════════════════════ */
function PageFeatures() {
  const [importances, setImportances] = useState({});
  const [loading, setLoading]         = useState(true);
  const [error,   setError]           = useState(null);

  useEffect(() => {
    setLoading(true);
    getFeatureImportance()
      .then(r => { setImportances(r.data || {}); setLoading(false); })
      .catch(() => { setError("Feature importances unavailable — train Random Forest first."); setLoading(false); });
  }, []);

  const top10  = Object.entries(importances).sort((a, b) => b[1] - a[1]).slice(0, 10);
  const maxVal = top10[0]?.[1] || 1;

  return (
    <Panel title="Feature importances — Random Forest" sub="from /feature_importance">
      {loading && <Spinner />}
      {error && <ApiError msg={error} />}
      {!loading && !error && top10.length > 0 && (
        <>
          {top10.map(([name, val], i) => (
            <HBar key={name} name={name}
              val={parseFloat((val * 100).toFixed(2))}
              max={maxVal * 100}
              color={i < 3 ? C.red : i < 6 ? C.blue : C.green}
              nameW={200} />
          ))}
          <div style={{ marginTop: 8, padding: "7px 10px", background: C.blueL, borderRadius: 6,
            fontSize: 11, color: C.blue, border: `0.5px solid ${C.blueB}` }}>
            Données réelles du modèle Random Forest entraîné sur 7 043 clients.
          </div>
        </>
      )}
      {!loading && !error && top10.length === 0 && (
        <div style={{ padding: 20, textAlign: "center", color: C.muted, fontSize: 12 }}>
          Entraîne Random Forest pour voir les importances.
        </div>
      )}
    </Panel>
  );
}

/* ═══════════════════════════════════════════════════════════════
   PAGE: LIVE PREDICTION — 6 models in parallel
═══════════════════════════════════════════════════════════════ */
function PagePredict() {
  const [tenure,   setTenure]   = useState(12);
  const [monthly,  setMonthly]  = useState(65);
  const [contract, setContract] = useState("m");
  const [internet, setInternet] = useState("d");
  const [hasSec,   setHasSec]   = useState(false);
  const [hasTech,  setHasTech]  = useState(false);
  const [preds,    setPreds]    = useState({});
  const [loading,  setLoading]  = useState(false);
  const debounceRef = useRef(null);

  useEffect(() => {
    clearTimeout(debounceRef.current);
    setLoading(true);
    debounceRef.current = setTimeout(async () => {
      const features = buildFeatures(tenure, monthly, contract, internet, hasSec, hasTech);
      const settled = await Promise.allSettled(
        ALL_MODELS.map(m => predictChurn(m.id, features).then(r => ({ id: m.id, data: r.data })))
      );
      const newPreds = {};
      settled.forEach(r => { if (r.status === "fulfilled") newPreds[r.value.id] = r.value.data; });
      setPreds(newPreds);
      setLoading(false);
    }, 600);
    return () => clearTimeout(debounceRef.current);
  }, [tenure, monthly, contract, internet, hasSec, hasTech]);

  const riskColor = pct => pct > 60 ? C.red : pct > 35 ? C.amber : C.green;
  const riskLabel = pct => pct > 60 ? "HIGH" : pct > 35 ? "MED" : "LOW";

  return (
    <div style={{ display: "grid", gridTemplateColumns: "260px 1fr", gap: 12 }}>
      {/* Left: inputs */}
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        <Panel title="Profil client">
          {/* Tenure */}
          <div style={{ marginBottom: 10 }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
              <span style={{ fontSize: 11, color: C.muted }}>Ancienneté (mois)</span>
              <span style={{ fontSize: 11, fontWeight: 500, color: C.ink }}>{tenure}</span>
            </div>
            <input type="range" min={0} max={72} value={tenure} step={1}
              onChange={e => setTenure(+e.target.value)} style={{ width: "100%" }} />
          </div>

          {/* Monthly */}
          <div style={{ marginBottom: 10 }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
              <span style={{ fontSize: 11, color: C.muted }}>Charges mensuelles ($)</span>
              <span style={{ fontSize: 11, fontWeight: 500, color: C.ink }}>{monthly}</span>
            </div>
            <input type="range" min={18} max={120} value={monthly} step={1}
              onChange={e => setMonthly(+e.target.value)} style={{ width: "100%" }} />
          </div>

          {/* Contract */}
          <div style={{ marginBottom: 10 }}>
            <div style={{ fontSize: 11, color: C.muted, marginBottom: 4 }}>Contrat</div>
            <div style={{ display: "flex", gap: 4 }}>
              {[["m","Month-to-month"],["1","One year"],["2","Two year"]].map(([v, l]) => (
                <button key={v} onClick={() => setContract(v)} style={{
                  flex: 1, padding: "5px 4px", borderRadius: 5, cursor: "pointer", fontFamily: "inherit",
                  fontSize: 10, border: `0.5px solid ${contract === v ? C.red : C.border}`,
                  background: contract === v ? C.redL : "#fff",
                  color: contract === v ? C.red : C.muted, fontWeight: contract === v ? 600 : 400,
                }}>{l}</button>
              ))}
            </div>
          </div>

          {/* Internet */}
          <div style={{ marginBottom: 10 }}>
            <div style={{ fontSize: 11, color: C.muted, marginBottom: 4 }}>Service internet</div>
            <div style={{ display: "flex", gap: 4 }}>
              {[["d","DSL"],["f","Fiber"],["n","Aucun"]].map(([v, l]) => (
                <button key={v} onClick={() => setInternet(v)} style={{
                  flex: 1, padding: "5px 4px", borderRadius: 5, cursor: "pointer", fontFamily: "inherit",
                  fontSize: 10, border: `0.5px solid ${internet === v ? C.blue : C.border}`,
                  background: internet === v ? C.blueL : "#fff",
                  color: internet === v ? C.blue : C.muted, fontWeight: internet === v ? 600 : 400,
                }}>{l}</button>
              ))}
            </div>
          </div>

          {/* Checkboxes */}
          {[["Sécurité en ligne", hasSec, setHasSec], ["Support technique", hasTech, setHasTech]].map(([l, v, s]) => (
            <label key={l} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 11,
              color: C.ink2, cursor: "pointer", marginBottom: 6 }}>
              <input type="checkbox" checked={v} onChange={e => s(e.target.checked)}
                style={{ accentColor: C.green }} />
              {l}
            </label>
          ))}

          <div style={{ marginTop: 6, padding: "6px 10px", background: C.bg, borderRadius: 5,
            fontSize: 10, color: C.muted, fontFamily: "monospace" }}>
            Total: ${Math.round(tenure * monthly).toLocaleString()}
          </div>
        </Panel>

        <Panel title="Info" style={{ fontSize: 11, color: C.muted }}>
          <div>Mise à jour automatique · 600ms debounce</div>
          <div style={{ marginTop: 4 }}>6 modèles en parallèle via /predict</div>
          {loading && <div style={{ marginTop: 6, color: C.amber }}>⟳ Calcul en cours…</div>}
        </Panel>
      </div>

      {/* Right: 6 model results */}
      <div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 10, marginBottom: 12 }}>
          {ALL_MODELS.map(m => {
            const pred = preds[m.id];
            const pct  = pred?.churn_probability != null ? Math.round(pred.churn_probability * 100) : null;
            const rc   = pct != null ? riskColor(pct) : C.hint;
            return (
              <div key={m.id} style={{ background: C.surface, borderRadius: 9,
                border: `0.5px solid ${pct != null ? m.colorB : C.border}`, padding: "14px 16px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                  <span style={{ fontSize: 11, color: m.color, fontWeight: 600 }}>{m.label}</span>
                  {pct != null && (
                    <span style={{ fontSize: 9, padding: "2px 6px", borderRadius: 8,
                      background: rc, color: "#fff", fontWeight: 600 }}>{riskLabel(pct)}</span>
                  )}
                </div>
                <div style={{ fontSize: 36, fontWeight: 500, color: pct != null ? rc : C.hint, lineHeight: 1 }}>
                  {pct != null ? pct + "%" : "—"}
                </div>
                <div style={{ fontSize: 10, color: C.muted, marginTop: 4 }}>
                  {pct != null ? "churn probability" : loading ? "calcul…" : "non disponible"}
                </div>
                {pct != null && (
                  <div style={{ height: 4, background: C.bg2, borderRadius: 2, marginTop: 8, overflow: "hidden" }}>
                    <div style={{ width: `${pct}%`, height: "100%", background: rc, borderRadius: 2, transition: "width 0.6s" }} />
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Consensus */}
        {Object.keys(preds).length > 0 && (() => {
          const probs = Object.values(preds).map(p => p.churn_probability).filter(v => v != null);
          const avg   = probs.reduce((a, b) => a + b, 0) / probs.length;
          const pct   = Math.round(avg * 100);
          const rc    = riskColor(pct);
          return (
            <Panel title="Consensus — moyenne des 6 modèles">
              <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
                <div style={{ fontSize: 48, fontWeight: 500, color: rc, lineHeight: 1 }}>{pct}%</div>
                <div>
                  <div style={{ display: "inline-block", padding: "3px 14px", borderRadius: 16,
                    background: rc, color: "#fff", fontSize: 11, fontWeight: 600, marginBottom: 6 }}>
                    {pct > 60 ? "RISQUE ÉLEVÉ" : pct > 35 ? "RISQUE MODÉRÉ" : "RISQUE FAIBLE"}
                  </div>
                  <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.6 }}>
                    {pct > 60 ? "Offrir une remise ou un contrat annuel pour fidéliser."
                      : pct > 35 ? "Surveiller proactivement ce client."
                      : "Engagement standard suffisant."}
                  </div>
                </div>
                <div style={{ flex: 1, background: C.bg2, borderRadius: 4, height: 8, overflow: "hidden" }}>
                  <div style={{ width: `${pct}%`, height: "100%", background: rc, borderRadius: 4, transition: "width 0.6s" }} />
                </div>
              </div>
            </Panel>
          );
        })()}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   PAGE: TRAIN
═══════════════════════════════════════════════════════════════ */
function PageTrain({ onTrainComplete }) {
  const [selModel, setSelModel] = useState("random_forest");
  const [params,   setParams]   = useState({});
  const [progress, setProgress] = useState(0);
  const [msg,      setMsg]      = useState("Sélectionne un modèle et clique sur Démarrer.");
  const [done,     setDone]     = useState(false);
  const [results,  setResults]  = useState(null);
  const [ts,       setTs]       = useState("");
  const [apiError, setApiError] = useState(null);
  const animRef = useRef(null);

  const m    = ALL_MODELS.find(x => x.id === selModel) || ALL_MODELS[0];
  const defP = Object.fromEntries(m.params.map(p => [p.name, params[m.id + p.name] ?? p.default]));

  const STEPS = [
    [8,  "Chargement dataset et splits…"],
    [18, "Application StandardScaler…"],
    [28, "Initialisation du pipeline…"],
    [40, "Fitting — 25%…"],
    [54, "Fitting — 55%…"],
    [67, "Fitting — 85%…"],
    [78, "Évaluation sur le test set…"],
    [87, "Calcul accuracy, F1, Recall, AUC…"],
    [94, "Logging MLflow — attente backend…"],
  ];

  const startTrain = async () => {
    clearTimeout(animRef.current);
    setDone(false); setResults(null); setProgress(0); setTs(""); setApiError(null);
    let i = 0;
    const animate = () => {
      if (i >= STEPS.length) return;
      const [pct, txt] = STEPS[i++];
      setProgress(pct);
      setMsg("[" + pct + "%] " + txt);
      if (i < STEPS.length) animRef.current = setTimeout(animate, 340 + Math.random() * 180);
    };
    animate();
    try {
      const { data } = await trainModel(selModel, defP);
      clearTimeout(animRef.current);
      setResults(data);
      setDone(true);
      setProgress(100);
      setMsg("[100%] Entraînement terminé — modèle sauvegardé.");
      setTs("Terminé à " + new Date().toLocaleTimeString());
      if (onTrainComplete) onTrainComplete();
    } catch (e) {
      clearTimeout(animRef.current);
      setApiError(e.response?.data?.error || "Entraînement échoué.");
    }
  };

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 12 }}>
        <Panel title="Sélectionner le modèle" sub="6 algorithmes">
          {ALL_MODELS.map(md => (
            <div key={md.id}
              onClick={() => { setSelModel(md.id); setDone(false); setResults(null); setProgress(0); setApiError(null); }}
              style={{ display: "flex", alignItems: "center", gap: 8, padding: "9px 10px", borderRadius: 7,
                cursor: "pointer", marginBottom: 5,
                background: selModel === md.id ? md.colorL : "transparent",
                border: `0.5px solid ${selModel === md.id ? md.colorB : C.bg2}`,
                transition: "all 0.15s" }}>
              <div style={{ width: 9, height: 9, borderRadius: "50%", background: md.color, flexShrink: 0 }} />
              <span style={{ fontSize: 12, fontWeight: selModel === md.id ? 500 : 400,
                color: selModel === md.id ? md.color : C.ink2, flex: 1 }}>{md.label}</span>
              {md.badge === "best" && <Badge text="best" variant="red" />}
              {md.badge === "boosting" && <Badge text="boosting" variant={md.id === "xgboost" ? "purple" : "orange"} />}
            </div>
          ))}
        </Panel>

        <Panel title="Hyperparamètres" sub={m.label}>
          <div style={{ marginBottom: 10, fontSize: 11, color: C.muted, lineHeight: 1.5 }}>{m.desc}</div>
          {m.params.map(param => (
            <div key={param.name} style={{ marginBottom: 12 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                <span style={{ fontSize: 11, color: C.muted }}>{param.label}</span>
                <span style={{ fontSize: 11, fontWeight: 500, color: C.ink, fontFamily: "monospace" }}>{defP[param.name]}</span>
              </div>
              {param.type === "range" ? (
                <input type="range" min={param.min} max={param.max} value={defP[param.name]} step={param.step}
                  onChange={e => setParams(prev => ({ ...prev, [m.id + param.name]: parseFloat(e.target.value) }))}
                  style={{ width: "100%" }} />
              ) : (
                <select value={defP[param.name]}
                  onChange={e => setParams(prev => ({ ...prev, [m.id + param.name]: e.target.value }))}
                  style={{ width: "100%", padding: "6px 8px", background: C.bg,
                    border: `0.5px solid ${C.border}`, borderRadius: 5, color: C.ink, fontSize: 11, fontFamily: "inherit" }}>
                  {param.options.map(o => <option key={o} value={o}>{o}</option>)}
                </select>
              )}
            </div>
          ))}
        </Panel>
      </div>

      <Panel title="Progression" sub={ts}>
        {apiError && <ApiError msg={apiError} />}
        <div style={{ background: C.bg2, borderRadius: 4, height: 10, overflow: "hidden", marginBottom: 8 }}>
          <div style={{ width: `${progress}%`, height: "100%",
            background: done ? C.green : m.color, borderRadius: 4, transition: "width 0.35s" }} />
        </div>
        <div style={{ fontSize: 11, color: C.muted, fontFamily: "monospace", marginBottom: 14 }}>{msg}</div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(5,minmax(0,1fr))", gap: 8, marginBottom: 14 }}>
          {[
            ["Accuracy",  results ? (results.accuracy * 100).toFixed(1) + "%" : "—",  m.color],
            ["F1 Score",  results ? results.f1?.toFixed(3) : "—",                      C.blue],
            ["Recall",    results ? (results.recall * 100).toFixed(1) + "%" : "—",    C.green],
            ["Precision", results ? (results.precision * 100).toFixed(1) + "%" : "—", C.amber],
            ["AUC",       results ? results.roc_auc?.toFixed(3) : "—",                C.purple],
          ].map(([l, v, c]) => (
            <div key={l} style={{ background: C.bg, borderRadius: 8, padding: "8px 10px" }}>
              <div style={{ fontSize: 10, color: C.muted, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 4 }}>{l}</div>
              <div style={{ fontSize: 16, fontWeight: 500, color: results ? c : C.hint }}>{v}</div>
            </div>
          ))}
        </div>

        {results && (
          <div style={{ background: C.greenL, border: `0.5px solid ${C.greenB}`, borderRadius: 6,
            padding: "8px 12px", fontSize: 11, color: C.green, marginBottom: 14 }}>
            ✓ Modèle entraîné · Sauvegardé dans models/{selModel}.pkl · Run MLflow enregistré
          </div>
        )}

        <button onClick={startTrain} style={{ padding: "9px 24px", background: m.color, border: "none",
          borderRadius: 7, color: "#fff", fontWeight: 500, fontSize: 12, cursor: "pointer", fontFamily: "inherit" }}>
          Démarrer → {m.label}
        </button>
      </Panel>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   PAGE: MLFLOW RUNS
═══════════════════════════════════════════════════════════════ */
function PageMlflow({ refreshKey = 0 }) {
  const [runs,    setRuns]    = useState([]);
  const [total,   setTotal]   = useState(0);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(null);

  const fetchRuns = useCallback(() => {
    setLoading(true); setError(null);
    getMlflowRuns()
      .then(r => {
        const data = r.data || {};
        const list = Array.isArray(data) ? data : (data.runs || []);
        setRuns(list);
        setTotal(data.total ?? list.length);
      })
      .catch(() => setError("Impossible de charger les runs MLflow."))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { fetchRuns(); }, [fetchRuns, refreshKey]);

  const fmtDate = ts => {
    if (!ts) return "—";
    try { return new Date(parseInt(ts)).toLocaleString(); } catch { return ts; }
  };
  const fmtPct  = v => v != null ? (v * 100).toFixed(1) + "%" : "—";

  return (
    <div>
      <div style={{ background: C.blueL, border: `0.5px solid ${C.blueB}`, borderRadius: 8,
        padding: "10px 14px", marginBottom: 12, fontSize: 12, color: C.blue,
        display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span>
          ℹ Pour l'interface complète MLflow :&nbsp;
          <a href="http://localhost:5001" target="_blank" rel="noreferrer"
            style={{ color: C.red, fontWeight: 500, textDecoration: "underline" }}>
            http://localhost:5001
          </a>
        </span>
        <button onClick={fetchRuns} style={{ padding: "4px 12px", borderRadius: 5, border: `0.5px solid ${C.blueB}`,
          background: "#fff", color: C.blue, fontSize: 11, cursor: "pointer", fontFamily: "inherit" }}>
          ⟳ Refresh
        </button>
      </div>

      {/* Stats */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3,minmax(0,1fr))", gap: 10, marginBottom: 12 }}>
        <KCard label="Total runs" value={loading ? "…" : total}
          sub="all experiments" color={C.blue}
          badge={{ text: "MLflow", bg: C.blueL, c: "#0C447C", b: C.blueB }} />
        <KCard label="Experiment" value="churn_prediction"
          sub="task3" color={C.ink}
          badge={{ text: "active", bg: C.greenL, c: "#27500A", b: C.greenB }} />
        <KCard label="Tracking URI" value="local" sub="./mlruns" color={C.muted} />
      </div>

      <Panel title="Runs enregistrés" sub={`${total} runs · de /mlflow_runs`}>
        {loading && <Spinner />}
        {error   && <ApiError msg={error} retry={fetchRuns} />}
        {!loading && !error && runs.length === 0 && (
          <div style={{ padding: 20, textAlign: "center", color: C.muted, fontSize: 12 }}>
            Aucun run trouvé. Entraîne un modèle pour créer un run MLflow.
          </div>
        )}
        {!loading && !error && runs.length > 0 && (
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, tableLayout: "fixed" }}>
            <thead>
              <tr>
                {["Run ID", "Modèle", "Accuracy", "F1", "Params", "Date"].map(h => (
                  <th key={h} style={{ padding: "5px 8px", textAlign: "left", color: C.muted, fontWeight: 500,
                    fontSize: 10, borderBottom: `0.5px solid ${C.border}`, textTransform: "uppercase" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {runs.map((run, i) => {
                const modelId = (run.model || "").toLowerCase().replace(" (ui)", "").replace(/ /g, "_");
                const mCfg = ALL_MODELS.find(m => m.id === modelId || run.model?.toLowerCase().includes(m.short.toLowerCase()));
                return (
                  <tr key={run.id || i} style={{ background: i % 2 === 0 ? C.bg : "transparent" }}>
                    <td style={{ padding: "5px 8px", color: C.muted, borderBottom: `0.5px solid ${C.bg2}`,
                      fontFamily: "monospace", fontSize: 10 }}>
                      {(run.id || "").slice(0, 8)}…
                    </td>
                    <td style={{ padding: "5px 8px", borderBottom: `0.5px solid ${C.bg2}` }}>
                      <span style={{ color: mCfg ? mCfg.color : C.ink2, fontWeight: 500 }}>
                        {run.model || "—"}
                      </span>
                    </td>
                    <td style={{ padding: "5px 8px", color: C.ink2, borderBottom: `0.5px solid ${C.bg2}` }}>
                      {run.accuracy != null ? (run.accuracy * 100).toFixed(1) + "%" : "—"}
                    </td>
                    <td style={{ padding: "5px 8px", color: C.ink2, borderBottom: `0.5px solid ${C.bg2}` }}>
                      {run.f1 != null ? (run.f1 * 100).toFixed(1) + "%" : "—"}
                    </td>
                    <td style={{ padding: "5px 8px", color: C.muted, borderBottom: `0.5px solid ${C.bg2}`,
                      fontSize: 10, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {run.params || "—"}
                    </td>
                    <td style={{ padding: "5px 8px", color: C.muted, borderBottom: `0.5px solid ${C.bg2}`, fontSize: 10 }}>
                      {fmtDate(run.timestamp)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </Panel>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   ROOT APP
═══════════════════════════════════════════════════════════════ */
const TABS = [
  { id: "overview",  label: "Overview" },
  { id: "analytics", label: "Analytics" },
  { id: "models",    label: "Models" },
  { id: "features",  label: "Features" },
  { id: "predict",   label: "Live prediction" },
  { id: "train",     label: "Train" },
  { id: "mlflow",    label: "MLflow runs" },
];

export default function App() {
  const [tab,             setTab]             = useState("overview");
  const [ready,           setReady]           = useState(false);
  const [apiStatus,       setApiStatus]       = useState(false);
  const [liveMetrics,     setLiveMetrics]     = useState({});
  const [stats,           setStats]           = useState(null);
  const [mlflowRefreshKey,setMlflowRefreshKey]= useState(0);

  /* Load Chart.js */
  useEffect(() => {
    if (window.Chart) { setReady(true); return; }
    const s = document.createElement("script");
    s.src = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js";
    s.onload = () => setReady(true);
    document.head.appendChild(s);
  }, []);

  /* Fetch all live data */
  const fetchData = useCallback(async () => {
    try {
      await getHealth();
      setApiStatus(true);
      const [resRes, statsRes] = await Promise.all([getResults(), getStats()]);
      const results = resRes.data || {};
      if (Object.keys(results).length > 0) setLiveMetrics(results);
      setStats(statsRes.data || null);
    } catch {
      setApiStatus(false);
    }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  /* Auto-refresh every 30s */
  useEffect(() => {
    const t = setInterval(fetchData, 30000);
    return () => clearInterval(t);
  }, [fetchData]);

  const handleTrainComplete = () => {
    fetchData();
    getFeatureImportance().catch(() => {});
    setMlflowRefreshKey(k => k + 1);
  };

  const bestId  = Object.keys(liveMetrics).length
    ? Object.keys(liveMetrics).reduce((a, b) => liveMetrics[a].accuracy > liveMetrics[b].accuracy ? a : b)
    : null;
  const bestAcc = bestId ? ((liveMetrics[bestId].accuracy || 0) * 100).toFixed(1) + "%" : "—";
  const bestAuc = bestId ? (liveMetrics[bestId].roc_auc || 0).toFixed(3) : "—";

  return (
    <div style={{ fontFamily: "'Epilogue','Segoe UI',system-ui,sans-serif", background: C.bg, minHeight: "100vh" }}>
      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>

      {/* Topbar */}
      <div style={{ background: C.topbar, display: "flex", alignItems: "center",
        justifyContent: "space-between", padding: "0 20px", height: 52 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 30, height: 30, background: C.red, borderRadius: 6,
            display: "flex", alignItems: "center", justifyContent: "center" }}>
            <svg width={15} height={15} viewBox="0 0 18 18" fill="#fff">
              <path d="M9 1L1 6.5V17h5v-5h6v5h5V6.5z" />
            </svg>
          </div>
          <span style={{ fontSize: 15, fontWeight: 500, color: "#fff", letterSpacing: "-0.02em" }}>
            Churn<span style={{ color: C.red }}>ML</span>
          </span>
          <span style={{ fontSize: 10, color: "rgba(255,255,255,0.35)",
            borderLeft: "1px solid rgba(255,255,255,0.12)", paddingLeft: 10, marginLeft: 4 }}>
            Telecom Churn Prediction · 6 Models
          </span>
        </div>

        <div style={{ display: "flex", gap: 0 }}>
          {[["1 869", "at risk", "#ff6b5b"], ["27.3%", "churn rate", "#f5a623"],
            [bestAcc, "best acc.", "#2ecc71"], [bestAuc, "best AUC", "#fff"]].map(([v, l, c]) => (
            <div key={l} style={{ textAlign: "center", borderLeft: "1px solid rgba(255,255,255,0.1)", padding: "0 14px" }}>
              <div style={{ fontSize: 13, fontWeight: 500, color: c }}>{v}</div>
              <div style={{ fontSize: 9, color: "rgba(255,255,255,0.35)", letterSpacing: "0.08em",
                textTransform: "uppercase", marginTop: 1 }}>{l}</div>
            </div>
          ))}
        </div>

        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <span style={{ fontSize: 10, color: "rgba(255,255,255,0.4)" }}>
            <span style={{ display: "inline-block", width: 6, height: 6, borderRadius: "50%",
              background: apiStatus ? "#2ecc71" : "#e74c3c", marginRight: 5 }} />
            {apiStatus ? "API connected" : "API disconnected"}
          </span>
          <button onClick={() => setTab("mlflow")} style={{ padding: "5px 12px", borderRadius: 5,
            border: "1px solid rgba(255,255,255,0.15)", background: "rgba(255,255,255,0.07)",
            color: "#fff", fontSize: 11, cursor: "pointer", fontFamily: "inherit" }}>MLflow</button>
          <button onClick={() => setTab("train")} style={{ padding: "5px 12px", borderRadius: 5,
            border: "none", background: C.red, color: "#fff", fontSize: 11, fontWeight: 500,
            cursor: "pointer", fontFamily: "inherit" }}>Train model</button>
          <button onClick={fetchData} title="Refresh" style={{ padding: "5px 9px", borderRadius: 5,
            border: "1px solid rgba(255,255,255,0.15)", background: "rgba(255,255,255,0.07)",
            color: "rgba(255,255,255,0.6)", fontSize: 14, cursor: "pointer", fontFamily: "inherit" }}>⟳</button>
        </div>
      </div>

      {/* Nav tabs */}
      <div style={{ background: C.surface, borderBottom: `1px solid ${C.border}`, display: "flex", padding: "0 20px" }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: "9px 14px", fontSize: 12, fontWeight: 500, cursor: "pointer",
            color: tab === t.id ? C.red : C.muted, background: "none", border: "none",
            borderBottom: `2px solid ${tab === t.id ? C.red : "transparent"}`,
            whiteSpace: "nowrap", transition: "color 0.15s", fontFamily: "inherit",
          }}>{t.label}</button>
        ))}
      </div>

      {/* Content */}
      <div style={{ padding: "16px 20px" }}>
        {!ready
          ? <div style={{ textAlign: "center", padding: 60, color: C.muted, fontSize: 13 }}>Chargement des graphiques…</div>
          : (
            <>
              {tab === "overview"  && <PageOverview liveMetrics={liveMetrics} apiStatus={apiStatus} stats={stats} />}
              {tab === "analytics" && <PageAnalytics />}
              {tab === "models"    && <PageModels liveMetrics={liveMetrics} />}
              {tab === "features"  && <PageFeatures />}
              {tab === "predict"   && <PagePredict />}
              {tab === "train"     && <PageTrain onTrainComplete={handleTrainComplete} />}
              {tab === "mlflow"    && <PageMlflow refreshKey={mlflowRefreshKey} />}
            </>
          )
        }
      </div>
    </div>
  );
}
