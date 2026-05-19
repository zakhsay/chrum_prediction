/**
 * PredictForm.js
 * Standalone customer churn prediction form.
 * Can be used independently or embedded inside App.js.
 */
import React, { useState } from 'react';
import { predictChurn } from '../services/api';

const C = {
  red: '#c0392b', redM: '#e74c3c', redL: '#fdf0ef', redB: '#f5c6c2',
  green: '#1e7e4a', greenM: '#27ae60', greenL: '#edf7f1', greenB: '#a8dfc0',
  amber: '#c0760a', amberL: '#fef8ee',
  ink: '#1a1714', muted: '#7a7268', border: '#dbd5cc', bg: '#f5f2ed', bg2: '#ede9e2',
};

const MODELS = [
  { key: 'random_forest',       label: 'Random Forest'          },
  { key: 'knn',                 label: 'K-Nearest Neighbors'    },
  { key: 'svm',                 label: 'Support Vector Machine'  },
  { key: 'logistic_regression', label: 'Logistic Regression'    },
];

export default function PredictForm() {
  const [model,    setModel]    = useState('random_forest');
  const [tenure,   setTenure]   = useState(12);
  const [monthly,  setMonthly]  = useState(65);
  const [total,    setTotal]    = useState(780);
  const [result,   setResult]   = useState(null);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState(null);

  const handleSubmit = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const { data } = await predictChurn(model, { tenure, MonthlyCharges: monthly, TotalCharges: total });
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.error || 'Prediction failed. Make sure the backend is running and the model is trained.');
    } finally {
      setLoading(false);
    }
  };

  const prob    = result ? Math.round(result.churn_probability * 100) : 0;
  const isHigh  = result?.churn_label === 1;
  const rColor  = isHigh ? C.red : C.green;

  return (
    <div style={s.card}>
      <h2 style={s.title}>Churn Prediction</h2>

      <div style={s.field}>
        <label style={s.lbl}>Model</label>
        <select value={model} onChange={e => setModel(e.target.value)} style={s.select}>
          {MODELS.map(m => <option key={m.key} value={m.key}>{m.label}</option>)}
        </select>
      </div>

      {[
        { key: 'tenure',  label: 'Tenure (months)',      val: tenure,  set: setTenure,  min: 0,   max: 72,   step: 1   },
        { key: 'monthly', label: 'Monthly charges ($)',  val: monthly, set: setMonthly, min: 18,  max: 120,  step: 0.5 },
        { key: 'total',   label: 'Total charges ($)',    val: total,   set: setTotal,   min: 18,  max: 8700, step: 10  },
      ].map(f => (
        <div key={f.key} style={s.field}>
          <div style={s.row}>
            <label style={s.lbl}>{f.label}</label>
            <span style={s.val}>{f.val}</span>
          </div>
          <input type="range" min={f.min} max={f.max} step={f.step}
                 value={f.val} onChange={e => f.set(parseFloat(e.target.value))}
                 style={{ width: '100%', accentColor: C.red }} />
        </div>
      ))}

      <button onClick={handleSubmit} disabled={loading} style={s.btn}>
        {loading ? 'Predicting…' : 'Predict churn risk'}
      </button>

      {error && <div style={s.err}>{error}</div>}

      {result && (
        <div style={{ ...s.result, borderColor: rColor }}>
          <div style={{ ...s.badge, background: rColor }}>{isHigh ? 'HIGH RISK' : 'LOW RISK'}</div>
          <div style={{ fontSize: 48, fontWeight: 500, color: rColor }}>{prob}%</div>
          <div style={{ fontSize: 12, color: C.muted, margin: '2px 0 10px' }}>churn probability</div>
          <div style={s.bar}><div style={{ ...s.barFill, width: `${prob}%` }} /></div>
          <p style={s.advice}>
            {isHigh
              ? 'High churn risk detected. Consider a loyalty discount, contract upgrade, or dedicated retention outreach.'
              : 'Low churn risk. Continue standard engagement and monitor the contract renewal date.'}
          </p>
        </div>
      )}
    </div>
  );
}

const s = {
  card:    { background: '#fff', borderRadius: 12, padding: 22, maxWidth: 480, border: `0.5px solid ${C.border}`, fontFamily: "'Epilogue','Segoe UI',system-ui,sans-serif", color: C.ink },
  title:   { margin: '0 0 16px', fontSize: 16, fontWeight: 500, color: C.ink, borderBottom: `0.5px solid ${C.bg2}`, paddingBottom: 10 },
  field:   { marginBottom: 12 },
  row:     { display: 'flex', justifyContent: 'space-between', marginBottom: 3 },
  lbl:     { fontSize: 11, color: C.muted },
  val:     { fontSize: 11, fontWeight: 500, color: C.ink },
  select:  { width: '100%', padding: '7px 9px', background: C.bg, border: `0.5px solid ${C.border}`, borderRadius: 6, color: C.ink, fontSize: 11 },
  btn:     { width: '100%', padding: 10, background: C.redM, border: 'none', borderRadius: 7, color: '#fff', fontWeight: 500, fontSize: 13, marginTop: 6, fontFamily: 'inherit' },
  err:     { marginTop: 10, padding: '8px 12px', background: C.redL, border: `0.5px solid ${C.redB}`, borderRadius: 7, fontSize: 12, color: C.red },
  result:  { marginTop: 16, padding: 18, borderRadius: 9, border: '1.5px solid', textAlign: 'center', background: C.bg },
  badge:   { display: 'inline-block', padding: '3px 14px', borderRadius: 20, color: '#fff', fontSize: 11, fontWeight: 500, marginBottom: 8 },
  bar:     { background: C.bg2, borderRadius: 5, height: 8, overflow: 'hidden', margin: '0 0 12px' },
  barFill: { height: '100%', borderRadius: 5, background: `linear-gradient(90deg,${C.greenM},${C.red})`, transition: 'width 0.8s' },
  advice:  { fontSize: 11, color: C.muted, lineHeight: 1.6, margin: 0 },
};
