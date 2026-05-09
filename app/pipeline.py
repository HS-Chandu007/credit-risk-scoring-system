import joblib
import json
import numpy as np

class CreditRiskPipeline:
    
    def __init__(self, path='artifacts'):
        self.xgb = joblib.load(f"{path}/xgb_model.pkl")
        self.lgbm = joblib.load(f"{path}/lgbm_model.pkl")
        
        with open(f"{path}/config.json") as f:
            config = json.load(f)
            
        self.threshold = config['threshold']
        
        self.w_xgb = config['weights']["xgb"]
        self.w_lgbm = config['weights']["lgbm"]
        
    def predict_proba(self, X):
        
        xgb_p = self.xgb.predict_proba(X)[:,1]
        lgbm_p = self.lgbm.predict_proba(X)[:,1]
        
        final_proba = (
            self.w_xgb * xgb_p + 
            self.w_lgbm * lgbm_p
        )
        
        return final_proba
    
    def predict(self, X):
        
        proba = self.predict_proba(X)
        
        preds = (
            proba >= self.threshold
        ).astype(int)
        
        return preds