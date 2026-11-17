from pathlib import Path
import json
import numpy as np
import pandas as pd
from signal_simulator import make_signal,CLASSES
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split

def extract(x,fs):
    rms=np.sqrt(np.mean(x*x)); mag=np.abs(np.fft.rfft(x)); freq=np.fft.rfftfreq(len(x),1/fs)
    peak=freq[1+np.argmax(mag[1:])]; centroid=(freq*mag).sum()/(mag.sum()+1e-9)
    return [x.mean(),x.std(),rms,np.max(np.abs(x)),np.percentile(np.abs(x),95),peak,centroid,np.sum(mag**2)]
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"; OUT.mkdir(exist_ok=True)
rows=[]; rng=np.random.default_rng(42)
for i in range(2400):
    label=CLASSES[i%4]; rpm=int(rng.integers(800,8000)); torque=float(rng.uniform(10,180))
    a,v=make_signal(label,rpm,torque,seed=i)
    rows.append(extract(a,2000)+extract(v,2000)+[label,rpm,torque])
cols=[f"f{i}" for i in range(16)]+["label","speed_rpm","torque_nm"]
df=pd.DataFrame(rows,columns=cols); df.to_csv(OUT/"signal_features.csv",index=False)
X=df[[c for c in cols if c.startswith("f")]]; y=df.label
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
m=RandomForestClassifier(n_estimators=300,max_depth=14,min_samples_leaf=2,
    class_weight="balanced",random_state=42,n_jobs=-1)
m.fit(Xtr,ytr); pred=m.predict(Xte)
metrics={"accuracy":float(accuracy_score(yte,pred)),"macro_f1":float(f1_score(yte,pred,average="macro")),
         "test_rows":len(yte),"classification_report":classification_report(yte,pred,output_dict=True)}
(OUT/"classification_metrics.json").write_text(json.dumps(metrics,indent=2))
pd.DataFrame(confusion_matrix(yte,pred,labels=CLASSES),index=CLASSES,columns=CLASSES).to_csv(OUT/"confusion_matrix.csv")
grid=[]
for rpm in np.linspace(500,9000,35):
 for tq in np.linspace(5,200,30):
    mech=2*np.pi*rpm/60*tq; losses=250+.0008*rpm**2+.018*tq**2
    grid.append((rpm,tq,mech/(mech+losses)))
pd.DataFrame(grid,columns=["speed_rpm","torque_nm","illustrative_efficiency"]).to_csv(OUT/"efficiency_map.csv",index=False)
print(json.dumps({k:v for k,v in metrics.items() if k!="classification_report"},indent=2)); print("Saved:",OUT)
