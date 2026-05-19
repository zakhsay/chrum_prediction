/**
 * services/api.js
 * All HTTP calls to the Flask backend in one place.
 * The "proxy" field in package.json forwards /api/* to http://localhost:5000.
 */
import axios from 'axios';

const API = axios.create({
  baseURL: '',
  headers: { 'Content-Type': 'application/json' },
  timeout: 120000,
});

export const getHealth          = ()          => API.get('/health');
export const getModels          = ()          => API.get('/models');
export const trainModel         = (model, params={}) => API.post('/train', { model, params });
export const getResults         = ()          => API.get('/results');
export const predictChurn       = (model, features) => API.post('/predict', { model, features });
export const getFeatureImportance = ()        => API.get('/feature_importance');

export default API;
