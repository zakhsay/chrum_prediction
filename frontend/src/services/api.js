/**
 * services/api.js
 * All HTTP calls to the Flask backend.
 * The "proxy" field in package.json forwards requests to http://localhost:5000.
 */
import axios from 'axios';

const API = axios.create({
  baseURL: '',
  headers: { 'Content-Type': 'application/json' },
  timeout: 120000,
});

export const getHealth            = ()               => API.get('/health');
export const getModels            = ()               => API.get('/models');
export const trainModel           = (model, params)  => API.post('/train', { model, params });
export const getResults           = ()               => API.get('/results');
export const predictChurn         = (model, features)=> API.post('/predict', { model, features });
export const getFeatureImportance = ()               => API.get('/feature_importance');
export const getStats             = ()               => API.get('/stats');
export const getTopRisk           = (model, n = 7)   => API.get(`/top_risk?model=${model}&n=${n}`);
export const getAnalytics         = ()               => API.get('/analytics');
export const getMlflowRuns        = ()               => API.get('/mlflow_runs');

export default API;
