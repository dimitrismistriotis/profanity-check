import pkg_resources
import numpy as np
import joblib

vectorizer = joblib.load(pkg_resources.resource_filename('profanity_check', 'data/vectorizer.joblib'))
model = joblib.load(pkg_resources.resource_filename('profanity_check', 'data/model.joblib'))

def _get_profane_prob(prob):
  return prob[1]

def predict(texts):
  if model.predict_proba(vectorizer.transform(texts)) >= 0.75:
    return 1
  else:
    return 0

def predict_prob(texts):
  return np.apply_along_axis(_get_profane_prob, 1, model.predict_proba(vectorizer.transform(texts)))
