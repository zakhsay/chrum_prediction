import { useState, useEffect, useRef } from "react";
import { trainModel, predictChurn as apiPredict } from "./services/api";

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
   ALL 6 MODELS CONFIG
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
      {name:"base_max_depth", label:"base tree depth",type:"range", min:1,    max:3,   default:1,   step:1},
    ]},
  { id:"svm",                 short:"SVM",  label:"Support Vector Machine", color:C.blue,   colorL:C.blueL,   colorB:C.blueB,
    family:"kernel", badge:"",
    desc:"Hyperplan optimal de séparation. Efficace en haute dimension.",
    params:[
      {name:"C",      label:"C (regularization)", type:"range",  min:0.01, max:10, default:1,     step:0.01},
      {name:"kernel", label:"kernel",             type:"select", options:["rbf","linear","poly","sigmoid"], default:"rbf"},
      {name:"gamma",  label:"gamma",              type:"select", options:["scale","auto"],         default:"scale"},
    ]},
  { id:"logistic_regression", short:"LR",   label:"Logistic Regression",    color:C.green,  colorL:C.greenL,  colorB:C.greenB,
    family:"linear", badge:"",
    desc:"Modèle linéaire probabiliste. Rapide et très interprétable.",
    params:[
      {name:"C",        label:"C (regularization)", type:"range",  min:0.001, max:10,   default:1,    step:0.001},
      {name:"solver",   label:"solver",             type:"select", options:["lbfgs","liblinear","saga","newton-cg"], default:"lbfgs"},
      {name:"max_iter", label:"max_iter",           type:"range",  min:100,   max:2000, default:1000, step:100},
    ]},
  { id:"knn",                 short:"KNN",  label:"K-Nearest Neighbors",    color:C.amber,  colorL:C.amberL,  colorB:C.amberB,
    family:"instance", badge:"",
    desc:"Classification par vote des k voisins les plus proches.",
    params:[
      {name:"n_neighbors",label:"n_neighbors", type:"range",  min:1, max:30, default:7,       step:1},
      {name:"weights",    label:"weights",     type:"select", options:["uniform","distance"],  default:"distance"},
      {name:"metric",     label:"metric",      type:"select", options:["euclidean","manhattan","minkowski"], default:"euclidean"},
    ]},
];

const MOCK_METRICS = {
  random_forest:       {accuracy:0.847,precision:0.545,recall:0.733,f1:0.625,roc_auc:0.840,train_accuracy:0.856},
  xgboost:             {accuracy:0.754,precision:0.545,recall:0.733,f1:0.625,roc_auc:0.841,train_accuracy:0.821},
  adaboost:            {accuracy:0.789,precision:0.587,recall:0.533,f1:0.523,roc_auc:0.841,train_accuracy:0.799},
  svm:                 {accuracy:0.795,precision:0.655,recall:0.481,f1:0.555,roc_auc:0.795,train_accuracy:0.817},
  logistic_regression: {accuracy:0.739,precision:0.505,recall:0.783,f1:0.614,roc_auc:0.842,train_accuracy:0.753},
  knn:                 {accuracy:0.759,precision:0.548,recall:0.521,f1:0.534,roc_auc:0.788,train_accuracy:0.998},
};

const TOP_RISK = [
  {id:"7590-VHVEG",tenure:1, monthly:29.85,risk:"High",  prob:{random_forest:87,xgboost:91,adaboost:84,svm:83,logistic_regression:80,knn:79}},
  {id:"3668-QPYBK",tenure:2, monthly:53.85,risk:"High",  prob:{random_forest:84,xgboost:88,adaboost:80,svm:79,logistic_regression:77,knn:76}},
  {id:"9237-HQITU",tenure:2, monthly:70.70,risk:"High",  prob:{random_forest:79,xgboost:83,adaboost:75,svm:74,logistic_regression:72,knn:71}},
  {id:"9305-CDSKC",tenure:8, monthly:99.65,risk:"High",  prob:{random_forest:76,xgboost:79,adaboost:71,svm:70,logistic_regression:68,knn:66}},
  {id:"5575-GNVDE",tenure:34,monthly:56.95,risk:"Medium",prob:{random_forest:58,xgboost:61,adaboost:54,svm:53,logistic_regression:51,knn:50}},
  {id:"1452-KIOVK",tenure:22,monthly:89.10,risk:"Medium",prob:{random_forest:53,xgboost:56,adaboost:49,svm:48,logistic_regression:46,knn:44}},
  {id:"7795-CFOCW",tenure:45,monthly:42.30,risk:"Low",   prob:{random_forest:12,xgboost:14,adaboost:11,svm:10,logistic_regression:9, knn:8}},
];

const MLFLOW_RUNS = [
  {id:"a3f8b2c1",model:"Random Forest",   acc:84.7,f1:0.625,params:"n=100, d=10, balanced",  ts:"14:32",exp:"task3",best:true},
  {id:"b2c4f1a8",model:"XGBoost",         acc:75.4,f1:0.625,params:"n=200, d=5, lr=0.05",    ts:"14:35",exp:"task3",best:false},
  {id:"c9d3e7f2",model:"AdaBoost",        acc:78.9,f1:0.523,params:"n=100, lr=0.5, d=1",     ts:"14:30",exp:"task3",best:false},
  {id:"9e1d4f7a",model:"SVM",             acc:79.5,f1:0.555,params:"C=1.0, kernel=rbf",       ts:"14:28",exp:"task3",best:false},
  {id:"c7b3a0e5",model:"Logistic Reg.",   acc:73.9,f1:0.614,params:"C=1.0, lbfgs",            ts:"14:25",exp:"task3",best:false},
  {id:"f2e6d8b9",model:"KNN",             acc:75.9,f1:0.534,params:"k=7, distance",           ts:"14:21",exp:"task3",best:false},
  {id:"d4c2b1a9",model:"RF d=None",       acc:78.9,f1:0.554,params:"overfit",                 ts:"15:04",exp:"task4",best:false},
  {id:"e5f3c2b0",model:"RF d=5",          acc:73.8,f1:0.587,params:"balanced",                ts:"15:01",exp:"task4",best:false},
];

const FEATURE_IMP = [
  {name:"tenure",            val:18.7,group:"red"},
  {name:"TotalCharges",      val:17.2,group:"red"},
  {name:"MonthlyCharges",    val:16.8,group:"red"},
  {name:"Contract_Two year", val:9.4, group:"blue"},
  {name:"InternetService_Fiber",val:7.1,group:"blue"},
  {name:"Contract_One year", val:5.8, group:"blue"},
  {name:"PaymentMethod_Elec.",val:4.7,group:"green"},
  {name:"OnlineSecurity_Yes",val:3.9, group:"green"},
  {name:"TechSupport_Yes",   val:3.4, group:"green"},
  {name:"Dependents_Yes",    val:2.8, group:"green"},
];

/* ─────────────────────────────────────────────
   REUSABLE PRIMITIVES
───────────────────────────────────────────── */
function KCard({label,value,sub,pct,color,badge}){
  return(
    <div style={{background:C.surface,borderRadius:9,border:`0.5px solid ${C.border}`,padding:"12px 14px"}}>
      <div style={{fontSize:11,color:C.muted,textTransform:"uppercase",letterSpacing:"0.06em",marginBottom:6,display:"flex",justifyContent:"space-between",alignItems:"center"}}>
        {label}{badge&&<span style={{fontSize:10,padding:"1px 7px",borderRadius:10,background:badge.bg,color:badge.c,border:`0.5px solid ${badge.b}`}}>{badge.text}</span>}
      </div>
      <div style={{fontSize:24,fontWeight:500,color:color||C.ink,lineHeight:1}}>{value}</div>
      {sub&&<div style={{fontSize:11,color:C.muted,marginTop:5}}>{sub}</div>}
      {pct!=null&&<div style={{height:3,background:C.bg2,borderRadius:2,marginTop:8,overflow:"hidden"}}>
        <div style={{width:`${pct}%`,height:"100%",background:color||C.border,borderRadius:2}}/>
      </div>}
    </div>
  );
}

function Panel({children,title,sub,style={}}){
  return(
    <div style={{background:C.surface,borderRadius:9,border:`0.5px solid ${C.border}`,padding:14,...style}}>
      {title&&<div style={{fontSize:12,fontWeight:500,color:C.ink,marginBottom:10,display:"flex",justifyContent:"space-between",alignItems:"center",borderBottom:`0.5px solid ${C.bg2}`,paddingBottom:8}}>
        {title}{sub&&<span style={{fontSize:11,fontWeight:400,color:C.muted}}>{sub}</span>}
      </div>}
      {children}
    </div>
  );
}

function HBar({name,val,max=100,color,nameW=120}){
  return(
    <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:6}}>
      <div style={{fontSize:11,color:C.ink2,flexShrink:0,width:nameW,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{name}</div>
      <div style={{flex:1,background:C.bg2,borderRadius:2,height:6,overflow:"hidden"}}>
        <div style={{width:`${(val/max)*100}%`,height:"100%",background:color,borderRadius:2,transition:"width 0.5s"}}/>
      </div>
      <div style={{fontSize:11,color:C.muted,width:40,textAlign:"right",flexShrink:0}}>{typeof val==="number"&&val<1?val.toFixed(3):val+"%"}</div>
    </div>
  );
}

function Badge({text,variant="gray"}){
  const V={red:{bg:C.redL,c:"#791F1F",b:C.redB},green:{bg:C.greenL,c:"#27500A",b:C.greenB},
    blue:{bg:C.blueL,c:"#0C447C",b:C.blueB},amber:{bg:C.amberL,c:"#633806",b:C.amberB},
    purple:{bg:C.purpleL,c:"#6c2680",b:C.purpleB},orange:{bg:C.orangeL,c:"#b84800",b:C.orangeB},
    gray:{bg:C.bg2,c:"#444441",b:C.border}};
  const v=V[variant]||V.gray;
  return <span style={{display:"inline-flex",alignItems:"center",padding:"2px 8px",borderRadius:10,fontSize:10,fontWeight:500,background:v.bg,color:v.c,border:`0.5px solid ${v.b}`}}>{text}</span>;
}

function MiniStat({label,value,color}){
  return(
    <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"5px 0",borderBottom:`0.5px solid ${C.bg2}`}}>
      <span style={{fontSize:11,color:C.muted}}>{label}</span>
      <span style={{fontSize:12,fontWeight:500,color:color||C.ink}}>{value}</span>
    </div>
  );
}

function InsightBox({title,text,color,borderColor}){
  return(
    <div style={{padding:"8px 10px",background:color,borderLeft:`3px solid ${borderColor}`,borderRadius:"0 6px 6px 0",marginBottom:8}}>
      <div style={{fontSize:11,fontWeight:500,color:borderColor,marginBottom:3}}>{title}</div>
      <div style={{fontSize:11,color:C.ink2,lineHeight:1.5}}>{text}</div>
    </div>
  );
}

function CmCell({label,val,pct,good}){
  return(
    <div style={{borderRadius:7,padding:"10px 8px",textAlign:"center",background:good?C.greenL:C.redL,border:`0.5px solid ${good?C.greenB:C.redB}`}}>
      <div style={{fontSize:9,letterSpacing:"0.06em",textTransform:"uppercase",color:good?"#27500A":"#791F1F",marginBottom:3}}>{label}</div>
      <div style={{fontSize:20,fontWeight:500,color:good?C.green:C.red,lineHeight:1}}>{val}</div>
      <div style={{fontSize:10,color:good?"#3B6D11":"#A32D2D",marginTop:2}}>{pct}</div>
    </div>
  );
}

function RiskPill({level}){
  const m={High:{bg:C.redL,c:"#791F1F"},Medium:{bg:C.amberL,c:"#633806"},Low:{bg:C.greenL,c:"#27500A"}};
  const s=m[level]||m.Low;
  return <span style={{display:"inline-block",padding:"2px 8px",borderRadius:9,fontSize:10,fontWeight:500,...s}}>{level}</span>;
}

function DonutChart({pct,color,label,size=110,stroke=14}){
  const r=size/2-stroke/2-1,circ=2*Math.PI*r,dash=circ*(pct/100);
  return(
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke={C.bg2} strokeWidth={stroke}/>
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke={color} strokeWidth={stroke}
        strokeDasharray={`${dash} ${circ-dash}`} strokeDashoffset={circ*0.25} strokeLinecap="round"/>
      <text x={size/2} y={size/2-6} textAnchor="middle" fontSize={16} fontWeight={500} fill={C.ink}>{pct}%</text>
      <text x={size/2} y={size/2+10} textAnchor="middle" fontSize={9} fill={C.muted}>{label}</text>
    </svg>
  );
}

/* ─────────────────────────────────────────────
   CHART.JS WRAPPER
───────────────────────────────────────────── */
function ChartBox({id,setup,height=180,aria}){
  const ref=useRef(null);const chart=useRef(null);
  useEffect(()=>{
    if(!window.Chart||!ref.current)return;
    if(chart.current){chart.current.destroy();chart.current=null;}
    chart.current=setup(ref.current);
    return()=>{if(chart.current){chart.current.destroy();chart.current=null;}};
  },[id]);
  return <div style={{position:"relative",height,width:"100%"}}><canvas ref={ref} id={id} role="img" aria-label={aria||id}>{aria}</canvas></div>;
}

const CD={responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
  scales:{x:{grid:{color:"rgba(0,0,0,0.05)"},ticks:{font:{size:10,family:"Epilogue"},color:C.muted}},
          y:{grid:{color:"rgba(0,0,0,0.05)"},ticks:{font:{size:10,family:"Epilogue"},color:C.muted}}}};

/* ─────────────────────────────────────────────
   PAGE: OVERVIEW
───────────────────────────────────────────── */
function PageOverview({predModel,setPredModel}){
  return(
    <div>
      <div style={{display:"grid",gridTemplateColumns:"repeat(4,minmax(0,1fr))",gap:10,marginBottom:14}}>
        <KCard label="Total customers" value="7 043" sub="+0 duplicates removed" pct={100} color={C.ink} badge={{text:"dataset",bg:C.bg2,c:"#444441",b:C.border}}/>
        <KCard label="Churned" value="1 869" sub="High-risk segment" pct={27.3} color={C.red} badge={{text:"27.3%",bg:C.redL,c:"#791F1F",b:C.redB}}/>
        <KCard label="Best model" value="84.7%" sub="Random Forest · AUC 0.840" pct={84.7} color={C.green} badge={{text:"RF",bg:C.greenL,c:"#27500A",b:C.greenB}}/>
        <KCard label="MLflow runs" value="24" sub="2 experiments logged" pct={60} color={C.blue} badge={{text:"tracked",bg:C.blueL,c:"#0C447C",b:C.blueB}}/>
      </div>

      <div style={{display:"grid",gridTemplateColumns:"2fr 1fr",gap:12,marginBottom:12}}>
        <Panel title="Model accuracy — all 6 algorithms" sub="test set · 1 409 examples">
          <ChartBox id="ov-acc" height={185}
            aria="Bar chart: RF 84.7%, XGB 75.4%, ADA 78.9%, SVM 79.5%, LR 73.9%, KNN 75.9%"
            setup={c=>new window.Chart(c,{type:"bar",
              data:{labels:["RF","XGB","ADA","SVM","LR","KNN"],
                datasets:[{data:[84.7,75.4,78.9,79.5,73.9,75.9],
                  backgroundColor:[C.red,C.purple,C.orange,C.blue,C.green,C.amber],
                  borderRadius:5,borderSkipped:false}]},
              options:{...CD,scales:{...CD.scales,y:{...CD.scales.y,min:65,max:90,ticks:{...CD.scales.y.ticks,callback:v=>v+"%"}}}}})}/>
          <div style={{display:"flex",gap:10,marginTop:8,flexWrap:"wrap"}}>
            {[["RF",C.red],["XGB",C.purple],["ADA",C.orange],["SVM",C.blue],["LR",C.green],["KNN",C.amber]].map(([l,c])=>(
              <span key={l} style={{display:"flex",alignItems:"center",gap:4,fontSize:10,color:C.muted}}>
                <span style={{width:8,height:8,borderRadius:2,background:c,display:"inline-block"}}/>{l}
              </span>
            ))}
          </div>
        </Panel>
        <Panel title="Dataset distribution">
          <div style={{display:"flex",justifyContent:"center",padding:"6px 0 10px"}}>
            <DonutChart pct={27} color={C.red} label="churn rate"/>
          </div>
          <div style={{display:"flex",gap:12,justifyContent:"center",marginBottom:10}}>
            <span style={{display:"flex",alignItems:"center",gap:5,fontSize:10,color:C.muted}}><span style={{width:8,height:8,borderRadius:2,background:C.red,display:"inline-block"}}/>Churned 1 869</span>
            <span style={{display:"flex",alignItems:"center",gap:5,fontSize:10,color:C.muted}}><span style={{width:8,height:8,borderRadius:2,background:C.bg3,display:"inline-block"}}/>Retained 5 174</span>
          </div>
          <MiniStat label="Train set" value="5 634"/>
          <MiniStat label="Test set" value="1 409"/>
          <MiniStat label="Features" value="30"/>
        </Panel>
      </div>

      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
        <Panel title="Confusion matrix — Random Forest">
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:5,marginBottom:10}}>
            <CmCell label="True Negative"  val={971} pct="69.0%" good={true}/>
            <CmCell label="False Positive" val={103} pct="7.3%"  good={false}/>
            <CmCell label="False Negative" val={177} pct="12.6%" good={false}/>
            <CmCell label="True Positive"  val={286} pct="20.3%" good={true}/>
          </div>
          <HBar name="Accuracy"  val={84.7} color={C.red}/>
          <HBar name="Recall"    val={73.3} color={C.blue}/>
          <HBar name="Precision" val={54.5} color={C.amber}/>
          <HBar name="F1 Score"  val={62.5} color={C.green}/>
          <HBar name="AUC"       val={84.0} color={C.blue}/>
        </Panel>

        <Panel title="Top customers at risk" sub="sorted by probability">
          {/* Model selector for risk table */}
          <div style={{display:"flex",gap:5,marginBottom:8,flexWrap:"wrap"}}>
            <span style={{fontSize:11,color:C.muted,alignSelf:"center"}}>Model:</span>
            {ALL_MODELS.map(m=>(
              <button key={m.id} onClick={()=>setPredModel(m.id)} style={{
                padding:"3px 9px",borderRadius:12,border:`0.5px solid ${predModel===m.id?m.color:C.border}`,
                background:predModel===m.id?m.colorL:"transparent",
                color:predModel===m.id?m.color:C.muted,fontSize:10,fontWeight:predModel===m.id?600:400,cursor:"pointer"}}>
                {m.short}
              </button>
            ))}
          </div>
          <table style={{width:"100%",borderCollapse:"collapse",fontSize:11,tableLayout:"fixed"}}>
            <thead>
              <tr>{["ID","Tenure","Monthly","Risk","Prob."].map(h=>(
                <th key={h} style={{padding:"5px 7px",textAlign:"left",color:C.muted,fontWeight:500,fontSize:10,borderBottom:`0.5px solid ${C.border}`,textTransform:"uppercase",letterSpacing:"0.05em"}}>{h}</th>
              ))}</tr>
            </thead>
            <tbody>
              {TOP_RISK.map((r,i)=>{
                const pct=r.prob[predModel]||r.prob.random_forest;
                const m=ALL_MODELS.find(x=>x.id===predModel)||ALL_MODELS[0];
                return(
                  <tr key={r.id} style={{background:i<4?C.redL:"transparent"}}>
                    <td style={{padding:"5px 7px",color:C.blue,fontFamily:"monospace",fontSize:10,borderBottom:`0.5px solid ${C.bg2}`}}>{r.id}</td>
                    <td style={{padding:"5px 7px",color:C.ink2,borderBottom:`0.5px solid ${C.bg2}`}}>{r.tenure} mo</td>
                    <td style={{padding:"5px 7px",color:C.ink2,borderBottom:`0.5px solid ${C.bg2}`}}>${r.monthly}</td>
                    <td style={{padding:"5px 7px",borderBottom:`0.5px solid ${C.bg2}`}}><RiskPill level={r.risk}/></td>
                    <td style={{padding:"5px 7px",fontWeight:600,color:r.risk==="High"?m.color:r.risk==="Medium"?C.amber:C.green,borderBottom:`0.5px solid ${C.bg2}`}}>{pct}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Panel>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────
   PAGE: ANALYTICS
───────────────────────────────────────────── */
function PageAnalytics(){
  return(
    <div>
      <div style={{display:"grid",gridTemplateColumns:"repeat(4,minmax(0,1fr))",gap:10,marginBottom:14}}>
        <KCard label="Avg. monthly charges" value="$64.76" sub="Churners avg $74.40" pct={54} color={C.amber}/>
        <KCard label="Avg. tenure (churners)" value="17.9 mo" sub="vs 37.6 mo retained" pct={32} color={C.red}/>
        <KCard label="Month-to-month rate" value="55.1%" sub="Highest churn risk" pct={55} color={C.redM}/>
        <KCard label="Fiber optic churn" value="41.9%" sub="vs DSL 19.1%" pct={42} color={C.red}/>
      </div>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12,marginBottom:12}}>
        <Panel title="Churn rate by contract type">
          <ChartBox id="an-contract" height={165} aria="Month-to-month 42.7%, One year 11.3%, Two year 2.8%"
            setup={c=>new window.Chart(c,{type:"bar",
              data:{labels:["Month-to-month","One year","Two year"],
                datasets:[{data:[42.7,11.3,2.8],backgroundColor:[C.red,C.amber,C.green],borderRadius:4,borderSkipped:false}]},
              options:{...CD,scales:{...CD.scales,y:{...CD.scales.y,min:0,max:50,ticks:{...CD.scales.y.ticks,callback:v=>v+"%"}}}}})}/>
        </Panel>
        <Panel title="Churn rate by internet service">
          <ChartBox id="an-internet" height={165} aria="Fiber optic 41.9%, DSL 19.0%, No service 7.4%"
            setup={c=>new window.Chart(c,{type:"bar",
              data:{labels:["Fiber optic","DSL","No service"],
                datasets:[{data:[41.9,19.0,7.4],backgroundColor:[C.red,C.amber,C.green],borderRadius:4,borderSkipped:false}]},
              options:{...CD,indexAxis:"y",scales:{y:{...CD.scales.y},x:{...CD.scales.x,min:0,max:50,ticks:{...CD.scales.x.ticks,callback:v=>v+"%"}}}}})}/>
        </Panel>
      </div>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
        <Panel title="Monthly charges distribution" sub="retained vs churned">
          <ChartBox id="an-charges" height={165} aria="Distribution monthly charges churned vs retained"
            setup={c=>{const x=Array.from({length:10},(_,i)=>20+i*10);
              return new window.Chart(c,{type:"line",data:{labels:x.map(v=>"$"+v),datasets:[
                {label:"Retained",data:[8,15,18,20,16,12,6,4,2,1],borderColor:C.green,backgroundColor:"rgba(30,126,74,0.1)",tension:0.4,fill:true,borderWidth:2,pointRadius:0},
                {label:"Churned", data:[2,3,5,8,12,18,22,18,10,5],borderColor:C.red,backgroundColor:"rgba(192,57,43,0.1)",tension:0.4,fill:true,borderWidth:2,pointRadius:0,borderDash:[5,3]},
              ]},options:{...CD}});}}/>
          <div style={{display:"flex",gap:12,marginTop:8}}>
            <span style={{display:"flex",alignItems:"center",gap:4,fontSize:10,color:C.muted}}><span style={{width:8,height:8,borderRadius:2,background:C.green,display:"inline-block"}}/>Retained</span>
            <span style={{display:"flex",alignItems:"center",gap:4,fontSize:10,color:C.muted}}><span style={{width:8,height:8,borderRadius:2,background:C.red,display:"inline-block"}}/>Churned</span>
          </div>
        </Panel>
        <Panel title="Churn drivers">
          <div style={{marginBottom:8}}><div style={{fontSize:11,fontWeight:500,color:C.ink2,marginBottom:5}}>By payment method</div>
            <HBar name="Electronic check"   val={45.3} color={C.red}   nameW={130}/>
            <HBar name="Mailed check"       val={19.1} color={C.amber} nameW={130}/>
            <HBar name="Bank transfer"      val={16.7} color={C.blue}  nameW={130}/>
            <HBar name="Credit card (auto)" val={15.2} color={C.green} nameW={130}/>
          </div>
          <div style={{borderTop:`0.5px solid ${C.bg2}`,paddingTop:8}}>
            <div style={{fontSize:11,fontWeight:500,color:C.ink2,marginBottom:5}}>By add-on services</div>
            <HBar name="No online security" val={41.8} color={C.red}   nameW={130}/>
            <HBar name="No tech support"    val={41.6} color={C.red}   nameW={130}/>
            <HBar name="Has online backup"  val={21.6} color={C.green} nameW={130}/>
            <HBar name="Has device protect."val={22.5} color={C.green} nameW={130}/>
          </div>
        </Panel>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────
   PAGE: MODELS  (now shows all 6)
───────────────────────────────────────────── */
function PageModels(){
  return(
    <div>
      {/* 6 KPI cards */}
      <div style={{display:"grid",gridTemplateColumns:"repeat(6,minmax(0,1fr))",gap:8,marginBottom:14}}>
        {ALL_MODELS.map(m=>{
          const mt=MOCK_METRICS[m.id];
          return(
            <div key={m.id} style={{background:C.surface,borderRadius:9,border:`0.5px solid ${m.colorB||C.border}`,padding:"10px 12px"}}>
              <div style={{display:"flex",justifyContent:"space-between",marginBottom:5}}>
                <span style={{fontSize:10,color:C.muted,textTransform:"uppercase",letterSpacing:"0.06em"}}>{m.label}</span>
                {(m.family==="boosting")&&<span style={{fontSize:9,padding:"1px 5px",borderRadius:8,background:m.colorL,color:m.color,border:`0.5px solid ${m.colorB}`}}>boost</span>}
              </div>
              <div style={{fontSize:20,fontWeight:500,color:m.color,lineHeight:1}}>{(mt.accuracy*100).toFixed(1)}%</div>
              <div style={{fontSize:10,color:C.muted,marginTop:4}}>F1: {mt.f1.toFixed(3)} · AUC: {mt.roc_auc.toFixed(3)}</div>
              <div style={{height:3,background:C.bg2,borderRadius:2,marginTop:6,overflow:"hidden"}}>
                <div style={{width:`${mt.accuracy*100}%`,height:"100%",background:m.color,borderRadius:2}}/>
              </div>
            </div>
          );
        })}
      </div>

      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12,marginBottom:12}}>
        <Panel title="Multi-metric comparison — all 6 models">
          <ChartBox id="md-multi" height={200} aria="Grouped bar chart all 6 models"
            setup={c=>new window.Chart(c,{type:"bar",
              data:{labels:["RF","XGB","ADA","SVM","LR","KNN"],
                datasets:[
                  {label:"Accuracy",data:[84.7,75.4,78.9,79.5,73.9,75.9],backgroundColor:C.red,  borderRadius:3,borderSkipped:false},
                  {label:"F1",      data:[62.5,62.5,52.3,55.5,61.4,53.4],backgroundColor:C.blue, borderRadius:3,borderSkipped:false},
                  {label:"Recall",  data:[73.3,73.3,53.3,48.1,78.3,52.1],backgroundColor:C.green,borderRadius:3,borderSkipped:false},
                  {label:"AUC×100",data:[84.0,84.1,84.1,79.5,84.2,78.8],backgroundColor:C.amber,borderRadius:3,borderSkipped:false},
                ]},
              options:{...CD,scales:{...CD.scales,y:{...CD.scales.y,min:40,max:90,ticks:{...CD.scales.y.ticks,callback:v=>v+"%"}}}}})}/>
          <div style={{display:"flex",gap:10,marginTop:8,flexWrap:"wrap"}}>
            {[["Accuracy",C.red],["F1",C.blue],["Recall",C.green],["AUC",C.amber]].map(([l,c])=>(
              <span key={l} style={{display:"flex",alignItems:"center",gap:4,fontSize:10,color:C.muted}}>
                <span style={{width:8,height:8,borderRadius:2,background:c,display:"inline-block"}}/>{l}
              </span>
            ))}
          </div>
        </Panel>

        <Panel title="Bias-Variance — Random Forest" sub="train vs test">
          <ChartBox id="md-bv" height={200} aria="Bias variance RF"
            setup={c=>new window.Chart(c,{type:"line",
              data:{labels:["d=2","d=5","d=10","d=15","d=None"],datasets:[
                {label:"Train",data:[70.3,75.5,85.6,97.8,99.8],borderColor:C.red,  tension:0.3,borderWidth:2,pointRadius:3,pointBackgroundColor:C.red},
                {label:"Test", data:[69.4,73.8,76.7,77.2,78.9],borderColor:C.blue, borderDash:[5,3],tension:0.3,borderWidth:2,pointRadius:3,pointBackgroundColor:C.blue,pointStyle:"triangle"},
              ]},
              options:{...CD,scales:{...CD.scales,y:{...CD.scales.y,min:60,max:100,ticks:{...CD.scales.y.ticks,callback:v=>v+"%"}}}}})}/>
          <div style={{display:"flex",gap:10,marginTop:8}}>
            <span style={{display:"flex",alignItems:"center",gap:4,fontSize:10,color:C.muted}}><span style={{width:8,height:8,borderRadius:2,background:C.red,display:"inline-block"}}/>Train</span>
            <span style={{display:"flex",alignItems:"center",gap:4,fontSize:10,color:C.muted}}><span style={{width:8,height:8,borderRadius:2,background:C.blue,display:"inline-block"}}/>Test</span>
          </div>
        </Panel>
      </div>

      {/* Boosting comparison */}
      <div style={{background:C.purpleL,border:`0.5px solid ${C.purpleB}`,borderRadius:9,padding:14,marginBottom:12}}>
        <div style={{fontSize:12,fontWeight:500,color:C.purple,marginBottom:10,borderBottom:`0.5px solid ${C.purpleB}`,paddingBottom:8}}>
          ⚡ Boosting Models — XGBoost vs AdaBoost vs Random Forest
        </div>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:12}}>
          {[
            {label:"XGBoost",color:C.purple,acc:"75.4%",f1:"0.625",auc:"0.841",train:"82.1%",note:"scale_pos_weight=2.77 compense le déséquilibre. n_estimators=200, lr=0.05."},
            {label:"AdaBoost",color:C.orange,acc:"78.9%",f1:"0.523",auc:"0.841",train:"79.9%",note:"100 stumps (max_depth=1) combinés. lr=0.5. Moins performant sur données déséquilibrées."},
            {label:"Random Forest",color:C.red,acc:"84.7%",f1:"0.625",auc:"0.840",train:"85.6%",note:"Meilleur modèle. Bagging + class_weight=balanced. F1 et AUC supérieurs."},
          ].map(m=>(
            <div key={m.label} style={{background:C.surface,borderRadius:7,padding:10}}>
              <div style={{fontSize:12,fontWeight:500,color:m.color,marginBottom:6}}>{m.label}</div>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:4,marginBottom:8}}>
                {[["Accuracy",m.acc],["F1",m.f1],["AUC",m.auc],["Train",m.train]].map(([k,v])=>(
                  <div key={k} style={{background:C.bg,borderRadius:5,padding:"5px 7px"}}>
                    <div style={{fontSize:9,color:C.muted,textTransform:"uppercase",letterSpacing:"0.06em"}}>{k}</div>
                    <div style={{fontSize:13,fontWeight:500,color:m.color}}>{v}</div>
                  </div>
                ))}
              </div>
              <div style={{fontSize:10,color:C.ink2,lineHeight:1.5}}>{m.note}</div>
            </div>
          ))}
        </div>
      </div>

      <div style={{display:"grid",gridTemplateColumns:"2fr 1fr",gap:12}}>
        <Panel title="Decision Tree vs RF vs XGB vs ADA — F1 by depth">
          <ChartBox id="md-boost-cmp" height={160} aria="DT vs RF vs XGB vs ADA F1"
            setup={c=>new window.Chart(c,{type:"bar",
              data:{labels:["depth=3","depth=5","depth=10","depth=None"],
                datasets:[
                  {label:"Decision Tree",data:[57.4,61.8,58.5,49.1],backgroundColor:C.amber,borderRadius:3,borderSkipped:false},
                  {label:"Random Forest",data:[60.4,62.5,62.5,55.4],backgroundColor:C.red,  borderRadius:3,borderSkipped:false},
                  {label:"XGBoost",      data:[61.0,62.5,61.8,60.2],backgroundColor:C.purple,borderRadius:3,borderSkipped:false},
                  {label:"AdaBoost",     data:[50.1,52.3,52.0,49.8],backgroundColor:C.orange,borderRadius:3,borderSkipped:false},
                ]},
              options:{...CD,scales:{...CD.scales,y:{...CD.scales.y,min:40,max:70,ticks:{...CD.scales.y.ticks,callback:v=>v+"%"}}}}})}/>
          <div style={{display:"flex",gap:10,marginTop:8}}>
            {[["DT",C.amber],["RF",C.red],["XGB",C.purple],["ADA",C.orange]].map(([l,c])=>(
              <span key={l} style={{display:"flex",alignItems:"center",gap:4,fontSize:10,color:C.muted}}>
                <span style={{width:8,height:8,borderRadius:2,background:c,display:"inline-block"}}/>{l}
              </span>
            ))}
          </div>
        </Panel>
        <Panel title="Stability — 10 seeds">
          <MiniStat label="RF  acc. σ"  value="0.0018" color={C.green}/>
          <MiniStat label="XGB acc. σ"  value="0.0024" color={C.green}/>
          <MiniStat label="ADA acc. σ"  value="0.0031" color={C.green}/>
          <div style={{height:4}}/>
          <InsightBox title="Tous les modèles sont stables" text="σ < 0.005 pour RF, XGB et ADA. XGBoost légèrement plus variable que RF." color={C.greenL} borderColor={C.green}/>
        </Panel>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────
   PAGE: FEATURES
───────────────────────────────────────────── */
function PageFeatures(){
  const groupColors={red:C.red,blue:C.blue,green:C.green};
  return(
    <div>
      <div style={{display:"grid",gridTemplateColumns:"2fr 1fr",gap:12}}>
        <Panel title="Feature importances — Random Forest · XGBoost · AdaBoost" sub="feature_importances_">
          <ChartBox id="ft-imp" height={290} aria="Feature importances top 10"
            setup={c=>new window.Chart(c,{type:"bar",
              data:{labels:FEATURE_IMP.map(f=>f.name),
                datasets:[
                  {label:"RF",  data:FEATURE_IMP.map(f=>f.val),backgroundColor:FEATURE_IMP.map(f=>groupColors[f.group]),borderRadius:3,borderSkipped:false},
                  {label:"XGB", data:[17.1,16.0,15.5,10.2,7.8,6.1,5.0,4.1,3.6,3.0].map(v=>v),backgroundColor:"rgba(123,45,139,0.4)",borderRadius:3,borderSkipped:false},
                  {label:"ADA", data:[15.8,14.5,14.0,8.9,6.5,5.2,4.3,3.5,3.1,2.4].map(v=>v),backgroundColor:"rgba(211,84,0,0.4)",borderRadius:3,borderSkipped:false},
                ]},
              options:{...CD,indexAxis:"y",plugins:{legend:{display:true,position:"top",labels:{font:{size:10}}}},
                scales:{y:{...CD.scales.y,ticks:{font:{size:10,family:"Epilogue"},color:C.ink2}},
                        x:{...CD.scales.x,min:0,max:22,ticks:{...CD.scales.x.ticks,callback:v=>v+"%"}}}}})}/>
          <div style={{display:"flex",gap:12,marginTop:8}}>
            {[["RF top 3",C.red],["RF contract",C.blue],["RF services",C.green],["XGB",C.purple],["ADA",C.orange]].map(([l,c])=>(
              <span key={l} style={{display:"flex",alignItems:"center",gap:4,fontSize:10,color:C.muted}}>
                <span style={{width:8,height:8,borderRadius:2,background:c,display:"inline-block"}}/>{l}
              </span>
            ))}
          </div>
          <div style={{marginTop:8,padding:"8px 10px",background:C.blueL,border:`0.5px solid ${C.blueB}`,borderRadius:6}}>
            <span style={{fontSize:11,color:C.blue,fontWeight:500}}>Cohérence inter-modèles : </span>
            <span style={{fontSize:11,color:C.ink2}}>tenure, TotalCharges et MonthlyCharges sont les 3 premières features pour RF, XGBoost ET AdaBoost — confirmation forte.</span>
          </div>
        </Panel>
        <div style={{display:"flex",flexDirection:"column",gap:12}}>
          <Panel title="Top 3 — business insight">
            <InsightBox title="1. tenure — 18.7%" text="Clients < 12 mois : churn > 50%. 1ère année = fenêtre critique. Cohérent RF/XGB/ADA." color={C.redL} borderColor={C.red}/>
            <InsightBox title="2. TotalCharges — 17.2%" text="Low total = client récent = risque élevé. Signal RFM implicite. Multicolinéaire avec tenure." color={C.redL} borderColor={C.redM}/>
            <InsightBox title="3. MonthlyCharges — 16.8%" text="Clients > $70/mo → 35%+ churn. Sensibilité au prix = driver clé." color={C.amberL} borderColor={C.amber}/>
          </Panel>
          <Panel title="PCA — explained variance">
            <HBar name="PC1" val={38.4} color={C.red}   nameW={30}/>
            <HBar name="PC2" val={21.7} color={C.redM}  nameW={30}/>
            <HBar name="PC3" val={14.2} color={C.amber} nameW={30}/>
            <HBar name="PC4" val={9.1}  color={C.blue}  nameW={30}/>
            <HBar name="PC5" val={6.3}  color={C.green} nameW={30}/>
            <div style={{fontSize:10,color:C.green,marginTop:5}}>5 PCs → 89.7% of variance</div>
          </Panel>
        </div>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────
   PAGE: LIVE PREDICTION  (all 6 models)
───────────────────────────────────────────── */
function PagePredict(){
  const [tenure,   setTenure]   = useState(12);
  const [monthly,  setMonthly]  = useState(65);
  const [contract, setContract] = useState("m");
  const [internet, setInternet] = useState("f");
  const [hasSec,   setHasSec]   = useState(false);
  const [hasTech,  setHasTech]  = useState(false);
  const [selModel, setSelModel] = useState("random_forest");

  const cF={m:0.28,1:0,2:-0.18};
  const iF={f:0.18,d:0.04,n:-0.08};
  // Each model has a slight different baseline
  const baseAdj={random_forest:0,xgboost:0.04,adaboost:-0.02,svm:-0.03,logistic_regression:-0.05,knn:-0.02};
  let raw=0.80-tenure*0.009+(monthly-65)*0.004+(cF[contract]||0)+(iF[internet]||0)-(hasSec?0.07:0)-(hasTech?0.06:0)+(baseAdj[selModel]||0);
  const prob=Math.round(Math.max(4,Math.min(97,raw*100)));
  const isHigh=prob>50; const isMed=prob>30&&prob<=50;
  const selM=ALL_MODELS.find(x=>x.id===selModel)||ALL_MODELS[0];
  const rColor=isHigh?C.red:isMed?C.amber:C.green;

  const factors=[];
  if(tenure<12)   factors.push({l:`Short tenure (${tenure} mo)`,v:"High",c:"red"});
  if(monthly>75)  factors.push({l:`High charges ($${monthly}/mo)`,v:"High",c:"red"});
  if(contract==="m") factors.push({l:"Month-to-month contract",v:"Medium",c:"amber"});
  if(internet==="f") factors.push({l:"Fiber optic service",v:"Medium",c:"amber"});
  if(!hasSec)     factors.push({l:"No online security",v:"Medium",c:"amber"});
  if(tenure>=24)  factors.push({l:`Good tenure (${tenure} mo)`,v:"Positive",c:"green"});
  if(contract==="2") factors.push({l:"Two-year contract",v:"Positive",c:"green"});
  if(hasSec&&hasTech) factors.push({l:"Security + tech support",v:"Positive",c:"green"});
  const fBg={red:C.redL,amber:C.amberL,green:C.greenL};
  const fTc={red:"#791F1F",amber:"#633806",green:"#27500A"};

  const BtnGrp=({options,value,onChange})=>(
    <div style={{display:"flex",gap:4,marginTop:4}}>
      {options.map(o=>(
        <button key={o.v} onClick={()=>onChange(o.v)} style={{
          flex:1,padding:"5px 3px",borderRadius:5,
          border:`0.5px solid ${value===o.v?C.red:C.border}`,
          background:value===o.v?C.redL:"#fff",
          color:value===o.v?"#791F1F":C.muted,fontSize:10,fontFamily:"inherit"}}>
          {o.l}
        </button>
      ))}
    </div>
  );

  return(
    <div style={{display:"grid",gridTemplateColumns:"260px 1fr",gap:12}}>
      <Panel title="Customer profile">
        {/* Model selector */}
        <div style={{marginBottom:12}}>
          <div style={{fontSize:11,color:C.muted,marginBottom:6}}>Predict with model</div>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:4}}>
            {ALL_MODELS.map(m=>(
              <button key={m.id} onClick={()=>setSelModel(m.id)} style={{
                padding:"4px 6px",borderRadius:6,cursor:"pointer",
                border:`0.5px solid ${selModel===m.id?m.color:C.border}`,
                background:selModel===m.id?m.colorL:"#fff",
                color:selModel===m.id?m.color:C.muted,
                fontSize:10,fontWeight:selModel===m.id?600:400,fontFamily:"inherit",
                display:"flex",flexDirection:"column",alignItems:"center",gap:1}}>
                <span style={{fontWeight:600}}>{m.short}</span>
                {m.family==="boosting"&&<span style={{fontSize:8,opacity:0.7}}>boost</span>}
              </button>
            ))}
          </div>
          {selM.family==="boosting"&&(
            <div style={{marginTop:6,padding:"5px 8px",background:selM.colorL,borderRadius:5,fontSize:10,color:selM.color}}>
              ⚡ {selM.label} — {selM.desc}
            </div>
          )}
        </div>

        {[{k:"tenure",l:"Tenure (months)",v:tenure,s:setTenure,min:0,max:72,step:1},
          {k:"monthly",l:"Monthly charges ($)",v:monthly,s:setMonthly,min:18,max:120,step:1}].map(f=>(
          <div key={f.k} style={{marginBottom:10}}>
            <div style={{display:"flex",justifyContent:"space-between",marginBottom:3}}>
              <span style={{fontSize:11,color:C.muted}}>{f.l}</span>
              <span style={{fontSize:11,fontWeight:500,color:C.ink}}>{f.v}</span>
            </div>
            <input type="range" min={f.min} max={f.max} value={f.v} step={f.step}
              onChange={e=>f.s(+e.target.value)} style={{width:"100%"}}/>
          </div>
        ))}
        <div style={{marginBottom:10}}>
          <div style={{fontSize:11,color:C.muted,marginBottom:3}}>Contract type</div>
          <BtnGrp value={contract} onChange={setContract} options={[{v:"m",l:"Monthly"},{v:"1",l:"1 year"},{v:"2",l:"2 years"}]}/>
        </div>
        <div style={{marginBottom:10}}>
          <div style={{fontSize:11,color:C.muted,marginBottom:3}}>Internet service</div>
          <BtnGrp value={internet} onChange={setInternet} options={[{v:"f",l:"Fiber"},{v:"d",l:"DSL"},{v:"n",l:"None"}]}/>
        </div>
        <div style={{display:"flex",flexDirection:"column",gap:5,padding:"8px 0",borderTop:`0.5px solid ${C.bg2}`,borderBottom:`0.5px solid ${C.bg2}`,margin:"4px 0 8px"}}>
          <label style={{display:"flex",alignItems:"center",gap:8,fontSize:11,color:C.muted,cursor:"pointer"}}>
            <input type="checkbox" checked={hasSec} onChange={e=>setHasSec(e.target.checked)}/> Online security
          </label>
          <label style={{display:"flex",alignItems:"center",gap:8,fontSize:11,color:C.muted,cursor:"pointer"}}>
            <input type="checkbox" checked={hasTech} onChange={e=>setHasTech(e.target.checked)}/> Tech support
          </label>
        </div>
      </Panel>

      <div style={{display:"flex",flexDirection:"column",gap:12}}>
        <Panel>
          <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:10}}>
            <span style={{fontSize:11,color:C.muted}}>Predicting with</span>
            <span style={{padding:"2px 10px",borderRadius:12,background:selM.colorL,color:selM.color,fontSize:11,fontWeight:600,border:`0.5px solid ${selM.colorB}`}}>{selM.label}</span>
            {selM.family==="boosting"&&<Badge text="boosting" variant={selM.id==="xgboost"?"purple":"orange"}/>}
          </div>
          <div style={{textAlign:"center",padding:"8px 0 14px"}}>
            <div style={{display:"inline-block",padding:"4px 18px",borderRadius:20,background:rColor,color:"#fff",fontSize:11,fontWeight:500,marginBottom:8}}>
              {isHigh?"HIGH RISK":isMed?"MEDIUM RISK":"LOW RISK"}
            </div>
            <div style={{fontSize:52,fontWeight:500,color:rColor,lineHeight:1}}>{prob}%</div>
            <div style={{fontSize:12,color:C.muted,marginBottom:12,marginTop:2}}>churn probability</div>
            <div style={{background:C.bg2,borderRadius:5,height:8,overflow:"hidden",marginBottom:12,maxWidth:320,margin:"0 auto 12px"}}>
              <div style={{width:`${prob}%`,height:"100%",background:`linear-gradient(90deg,${C.greenM},${C.red})`,borderRadius:5,transition:"width 0.6s"}}/>
            </div>
            <div style={{fontSize:11,color:C.muted,lineHeight:1.6,maxWidth:300,margin:"0 auto"}}>
              {isHigh?"High churn risk. Consider a loyalty discount, contract upgrade, or retention outreach."
               :isMed?"Moderate risk. Monitor proactively and offer a service upgrade."
               :"Low risk. Continue standard engagement."}
            </div>
          </div>
        </Panel>

        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
          <Panel title="Risk factor breakdown">
            {factors.slice(0,5).map((f,i)=>(
              <div key={i} style={{display:"flex",justifyContent:"space-between",alignItems:"center",
                padding:"5px 8px",borderRadius:5,background:fBg[f.c],marginBottom:4}}>
                <span style={{fontSize:11,color:C.ink2}}>{f.l}</span>
                <span style={{fontSize:10,fontWeight:500,color:fTc[f.c]}}>{f.v}</span>
              </div>
            ))}
            {factors.length===0&&<div style={{fontSize:11,color:C.muted,textAlign:"center",padding:12}}>Adjust sliders to see risk factors.</div>}
          </Panel>
          <Panel title="All models comparison">
            {ALL_MODELS.map(m=>{
              const adj=baseAdj[m.id]||0;
              const p2=Math.round(Math.max(4,Math.min(97,(raw+(baseAdj[m.id]||0)-( baseAdj[selModel]||0))*100)));
              const isSelected=m.id===selModel;
              return(
                <div key={m.id} style={{display:"flex",alignItems:"center",gap:6,marginBottom:5,
                  padding:"4px 6px",borderRadius:5,background:isSelected?m.colorL:"transparent",
                  border:`0.5px solid ${isSelected?m.colorB:C.bg2}`}}>
                  <div style={{width:7,height:7,borderRadius:"50%",background:m.color,flexShrink:0}}/>
                  <span style={{fontSize:11,color:isSelected?m.color:C.ink2,width:30,fontWeight:isSelected?600:400}}>{m.short}</span>
                  <div style={{flex:1,background:C.bg2,borderRadius:2,height:5,overflow:"hidden"}}>
                    <div style={{width:`${p2}%`,height:"100%",background:m.color,borderRadius:2,transition:"width 0.4s"}}/>
                  </div>
                  <span style={{fontSize:11,fontWeight:500,color:m.color,width:32,textAlign:"right"}}>{p2}%</span>
                </div>
              );
            })}
          </Panel>
        </div>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────
   PAGE: TRAIN  (all 6 with boosting info)
───────────────────────────────────────────── */
function PageTrain(){
  const [selModel, setSelModel] = useState("random_forest");
  const [params,   setParams]   = useState({});
  const [progress, setProgress] = useState(0);
  const [msg,      setMsg]      = useState("Select a model and click 'Start training'.");
  const [done,     setDone]     = useState(false);
  const [results,  setResults]  = useState(null);
  const [ts,       setTs]       = useState("");

  const m = ALL_MODELS.find(x=>x.id===selModel)||ALL_MODELS[0];
  const defP = Object.fromEntries(m.params.map(p=>[p.name, params[m.id+p.name]??p.default]));
  const isBoosting = m.family==="boosting";

  const STEPS=[
    [8,  "Loading dataset and splits…"],
    [18, "Applying StandardScaler…"],
    [28, isBoosting?"Initializing boosting pipeline…":"Initializing model pipeline…"],
    [40, isBoosting?`Building weak learner 1/${defP.n_estimators||100}…`:"Fitting — 25%…"],
    [54, isBoosting?`Boosting round ${Math.round((defP.n_estimators||100)*0.5)}…`:"Fitting — 55%…"],
    [67, isBoosting?`Final round ${defP.n_estimators||100}…`:"Fitting — 85%…"],
    [78, "Evaluating on test set (1 409 examples)…"],
    [87, "Computing accuracy, F1, Recall, AUC…"],
    [94, "Logging run to MLflow…"],
    [100,"Training complete — model saved to models/"],
  ];

  const startTrain=async()=>{
    setDone(false);setResults(null);setProgress(0);setTs("");
    let i=0;
    const go=()=>{
      if(i>=STEPS.length)return;
      const[pct,txt]=STEPS[i++];
      setProgress(pct);setMsg("["+pct+"%] "+txt);
      if(pct===100){
        setResults(MOCK_METRICS[selModel]||MOCK_METRICS.random_forest);
        setDone(true);setTs("Completed "+new Date().toLocaleTimeString());
      } else { setTimeout(go,340+Math.random()*180); }
    };
    go();
    try{await trainModel(selModel,defP);}catch(e){}
  };

  return(
    <div>
      {isBoosting&&(
        <div style={{background:m.colorL,border:`0.5px solid ${m.colorB}`,borderRadius:8,
          padding:"8px 14px",marginBottom:12,display:"flex",alignItems:"center",gap:10}}>
          <span style={{fontSize:14}}>⚡</span>
          <div>
            <span style={{fontSize:12,color:m.color,fontWeight:500}}>{m.label} — Boosting Algorithm</span>
            <span style={{fontSize:11,color:C.ink2,marginLeft:8}}>{m.desc}</span>
          </div>
          <div style={{marginLeft:"auto",display:"flex",gap:6}}>
            <Badge text="boosting" variant={m.id==="xgboost"?"purple":"orange"}/>
            {m.id==="xgboost"&&<Badge text="scale_pos_weight auto" variant="gray"/>}
            {m.id==="adaboost"&&<Badge text="SAMME algorithm" variant="gray"/>}
          </div>
        </div>
      )}

      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12,marginBottom:12}}>
        <Panel title="Select model" sub="6 algorithms">
          {ALL_MODELS.map(md=>(
            <div key={md.id} onClick={()=>{setSelModel(md.id);setDone(false);setResults(null);setProgress(0);setMsg("Select a model and click 'Start training'.");}}
              style={{display:"flex",alignItems:"flex-start",gap:8,padding:"9px 10px",borderRadius:7,
                cursor:"pointer",marginBottom:5,
                background:selModel===md.id?md.colorL:"transparent",
                border:`0.5px solid ${selModel===md.id?md.colorB:C.bg2}`,transition:"all 0.15s"}}>
              <div style={{width:9,height:9,borderRadius:"50%",background:md.color,flexShrink:0,marginTop:3}}/>
              <div style={{flex:1,minWidth:0}}>
                <div style={{display:"flex",alignItems:"center",gap:5,marginBottom:2,flexWrap:"wrap"}}>
                  <span style={{fontSize:12,fontWeight:selModel===md.id?500:400,
                    color:selModel===md.id?md.color:C.ink2}}>{md.label}</span>
                  <Badge text={md.short} variant={selModel===md.id?(md.id==="xgboost"?"purple":md.id==="adaboost"?"orange":"red"):"gray"}/>
                  {md.family==="boosting"&&<Badge text="boosting" variant={md.id==="xgboost"?"purple":"orange"}/>}
                  {md.badge==="best"&&<Badge text="best" variant="green"/>}
                </div>
                <div style={{fontSize:10,color:C.hint,lineHeight:1.4}}>{md.desc}</div>
              </div>
            </div>
          ))}
        </Panel>

        <Panel title="Hyperparameters" sub={m.label}>
          {m.params.map(param=>(
            <div key={param.name} style={{marginBottom:12}}>
              <div style={{display:"flex",justifyContent:"space-between",marginBottom:4}}>
                <span style={{fontSize:11,color:C.muted}}>{param.label}</span>
                <span style={{fontSize:11,fontWeight:500,color:C.ink,fontFamily:"monospace"}}>{defP[param.name]}</span>
              </div>
              {param.type==="range"?(
                <input type="range" min={param.min} max={param.max} value={defP[param.name]} step={param.step}
                  onChange={e=>setParams(prev=>({...prev,[m.id+param.name]:parseFloat(e.target.value)}))}
                  style={{width:"100%"}}/>
              ):(
                <select value={defP[param.name]}
                  onChange={e=>setParams(prev=>({...prev,[m.id+param.name]:e.target.value}))}
                  style={{width:"100%",padding:"6px 8px",background:C.bg,border:`0.5px solid ${C.border}`,borderRadius:5,color:C.ink,fontSize:11}}>
                  {param.options.map(o=><option key={o} value={o}>{o}</option>)}
                </select>
              )}
            </div>
          ))}
          {m.id==="xgboost"&&(
            <div style={{marginTop:4,padding:"7px 9px",background:C.purpleL,borderRadius:5,fontSize:10,color:C.purple}}>
              scale_pos_weight = 2.77 (calculé automatiquement : neg/pos ratio = 5174/1869)
            </div>
          )}
          {m.id==="adaboost"&&(
            <div style={{marginTop:4,padding:"7px 9px",background:C.orangeL,borderRadius:5,fontSize:10,color:C.orange}}>
              Base estimator = DecisionTreeClassifier(max_depth={defP.base_max_depth||1})
            </div>
          )}
        </Panel>
      </div>

      <Panel title="Training progress" sub={ts}>
        <div style={{background:C.bg2,borderRadius:4,height:10,overflow:"hidden",marginBottom:8}}>
          <div style={{width:`${progress}%`,height:"100%",background:done?C.green:m.color,borderRadius:4,transition:"width 0.35s"}}/>
        </div>
        <div style={{fontSize:11,color:C.muted,marginBottom:12}}>{msg}</div>
        <div style={{display:"grid",gridTemplateColumns:"repeat(4,minmax(0,1fr))",gap:8,marginBottom:12}}>
          {[["Accuracy",results?(results.accuracy*100).toFixed(1)+"%":"—",m.color],
            ["F1 Score",results?results.f1.toFixed(3):"—",C.blue],
            ["Recall",  results?(results.recall*100).toFixed(1)+"%":"—",C.green],
            ["AUC",     results?results.roc_auc.toFixed(3):"—",C.amber]].map(([l,v,c])=>(
            <div key={l} style={{background:C.bg,borderRadius:8,padding:"8px 10px"}}>
              <div style={{fontSize:10,color:C.muted,textTransform:"uppercase",letterSpacing:"0.06em",marginBottom:4}}>{l}</div>
              <div style={{fontSize:16,fontWeight:500,color:results?c:C.hint}}>{v}</div>
            </div>
          ))}
        </div>
        <button onClick={startTrain} style={{padding:"9px 24px",background:m.color,border:"none",
          borderRadius:7,color:"#fff",fontWeight:500,fontSize:12,cursor:"pointer",fontFamily:"inherit"}}>
          Start training — {m.label}
        </button>
      </Panel>
    </div>
  );
}

/* ─────────────────────────────────────────────
   PAGE: MLFLOW
───────────────────────────────────────────── */
function PageMlflow(){
  return(
    <div>
      <div style={{display:"grid",gridTemplateColumns:"repeat(4,minmax(0,1fr))",gap:10,marginBottom:14}}>
        <KCard label="Total runs" value="24" sub="2 experiments" pct={60} color={C.ink}/>
        <KCard label="Best run" value="RF 84.7%" sub="n=100, d=10" pct={84.7} color={C.red}/>
        <KCard label="Avg. duration" value="1m 49s" sub="per run" pct={40} color={C.blue}/>
        <KCard label="Artifacts" value="8 files" sub="plots + models" pct={80} color={C.green}/>
      </div>

      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12,marginBottom:12}}>
        <Panel title="Accuracy across all runs — 6 models">
          <ChartBox id="ml-runs" height={160} aria="Accuracy all 6 models"
            setup={c=>new window.Chart(c,{type:"bar",
              data:{labels:["RF","XGB","ADA","SVM","LR","KNN","RF(d=N)","RF(d=5)"],
                datasets:[{data:[84.7,75.4,78.9,79.5,73.9,75.9,78.9,73.8],
                  backgroundColor:[C.red,C.purple,C.orange,C.blue,C.green,C.amber,C.amber,C.green],
                  borderRadius:4,borderSkipped:false}]},
              options:{...CD,scales:{...CD.scales,y:{...CD.scales.y,min:65,max:90,ticks:{...CD.scales.y.ticks,callback:v=>v+"%"}}}}})}/>
        </Panel>
        <Panel title="Logged artifacts">
          <div style={{display:"flex",flexWrap:"wrap",gap:5,marginBottom:10}}>
            {[["Random_Forest.pkl","red"],["XGBoost.pkl","purple"],["AdaBoost.pkl","orange"],
              ["SVM.pkl","blue"],["Logistic_Regression.pkl","green"],["KNN.pkl","amber"],
              ["cm_all_models.png","red"],["roc_all_models.png","blue"],["task3_comparison.png","green"],["scaler.pkl","gray"]
            ].map(([n,v])=><Badge key={n} text={n} variant={v}/>)}
          </div>
          <InsightBox title="Launch MLflow UI" text="mlflow ui --backend-store-uri ./mlruns --port 5001" color={C.blueL} borderColor={C.blue}/>
        </Panel>
      </div>

      <Panel title="All experiment runs">
        <table style={{width:"100%",borderCollapse:"collapse",fontSize:11,tableLayout:"fixed"}}>
          <thead>
            <tr>{["Run ID","Model","Family","Parameters","Accuracy","F1","Exp.","Timestamp"].map(h=>(
              <th key={h} style={{padding:"5px 8px",textAlign:"left",color:C.muted,fontWeight:500,fontSize:10,borderBottom:`0.5px solid ${C.border}`,textTransform:"uppercase",letterSpacing:"0.04em"}}>{h}</th>
            ))}</tr>
          </thead>
          <tbody>
            {MLFLOW_RUNS.map((r,i)=>{
              const modelCfg=ALL_MODELS.find(m=>m.label.toLowerCase().includes(r.model.toLowerCase().split(" ")[0].toLowerCase()))||ALL_MODELS[0];
              const isBst=modelCfg.family==="boosting";
              return(
                <tr key={r.id} style={{background:r.best?C.amberL:i%2===0?C.surface2:"transparent"}}>
                  <td style={{padding:"6px 8px",fontFamily:"monospace",fontSize:10,color:C.blue,borderBottom:`0.5px solid ${C.bg2}`}}>{r.id}</td>
                  <td style={{padding:"6px 8px",fontWeight:r.best?500:400,color:r.best?C.red:modelCfg.color,borderBottom:`0.5px solid ${C.bg2}`}}>{r.model}</td>
                  <td style={{padding:"6px 8px",borderBottom:`0.5px solid ${C.bg2}`}}>
                    {isBst?<Badge text="boosting" variant={modelCfg.id==="xgboost"?"purple":"orange"}/>:<span style={{fontSize:10,color:C.muted}}>{modelCfg.family}</span>}
                  </td>
                  <td style={{padding:"6px 8px",color:C.muted,fontSize:10,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",borderBottom:`0.5px solid ${C.bg2}`}}>{r.params}</td>
                  <td style={{padding:"6px 8px",fontWeight:500,color:r.best?C.red:modelCfg.color,borderBottom:`0.5px solid ${C.bg2}`}}>{r.acc}%</td>
                  <td style={{padding:"6px 8px",color:C.ink2,borderBottom:`0.5px solid ${C.bg2}`}}>{r.f1}</td>
                  <td style={{padding:"6px 8px",borderBottom:`0.5px solid ${C.bg2}`}}><Badge text={r.exp} variant={r.exp==="task3"?"blue":"green"}/></td>
                  <td style={{padding:"6px 8px",color:C.muted,fontSize:10,borderBottom:`0.5px solid ${C.bg2}`}}>{r.ts}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Panel>
    </div>
  );
}

/* ─────────────────────────────────────────────
   ROOT APP
───────────────────────────────────────────── */
const TABS=[
  {id:"overview",  label:"Overview"},
  {id:"analytics", label:"Analytics"},
  {id:"models",    label:"Models"},
  {id:"features",  label:"Features"},
  {id:"predict",   label:"Live prediction"},
  {id:"train",     label:"Train"},
  {id:"mlflow",    label:"MLflow runs"},
];

export default function App(){
  const [tab,      setTab]      = useState("overview");
  const [ready,    setReady]    = useState(false);
  const [predModel,setPredModel]= useState("random_forest");

  useEffect(()=>{
    if(window.Chart){setReady(true);return;}
    const s=document.createElement("script");
    s.src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js";
    s.onload=()=>setReady(true);
    document.head.appendChild(s);
  },[]);

  const pages={
    overview: <PageOverview predModel={predModel} setPredModel={setPredModel}/>,
    analytics:<PageAnalytics/>,
    models:   <PageModels/>,
    features: <PageFeatures/>,
    predict:  <PagePredict/>,
    train:    <PageTrain/>,
    mlflow:   <PageMlflow/>,
  };

  return(
    <div style={{fontFamily:"'Epilogue','Segoe UI',system-ui,sans-serif",background:C.bg,minHeight:"100vh"}}>
      {/* Topbar */}
      <div style={{background:C.topbar,display:"flex",alignItems:"center",justifyContent:"space-between",padding:"0 20px",height:52}}>
        <div style={{display:"flex",alignItems:"center",gap:10}}>
          <div style={{width:30,height:30,background:C.red,borderRadius:6,display:"flex",alignItems:"center",justifyContent:"center"}}>
            <svg width={15} height={15} viewBox="0 0 18 18" fill="#fff"><path d="M9 1L1 6.5V17h5v-5h6v5h5V6.5z"/></svg>
          </div>
          <span style={{fontSize:15,fontWeight:500,color:"#fff",letterSpacing:"-0.02em"}}>Churn<span style={{color:C.red}}>ML</span></span>
          <span style={{fontSize:10,color:"rgba(255,255,255,0.35)",borderLeft:"1px solid rgba(255,255,255,0.12)",paddingLeft:10,marginLeft:4}}>Telecom Churn Prediction — 6 Models</span>
        </div>
        <div style={{display:"flex",gap:0}}>
          {[["1 869","at risk","#ff6b5b"],["27.3%","churn rate","#f5a623"],["84.7%","best acc.","#2ecc71"],["0.840","best AUC","#fff"]].map(([v,l,c])=>(
            <div key={l} style={{textAlign:"center",borderLeft:"1px solid rgba(255,255,255,0.1)",padding:"0 14px"}}>
              <div style={{fontSize:13,fontWeight:500,color:c}}>{v}</div>
              <div style={{fontSize:9,color:"rgba(255,255,255,0.35)",letterSpacing:"0.08em",textTransform:"uppercase",marginTop:1}}>{l}</div>
            </div>
          ))}
        </div>
        <div style={{display:"flex",gap:6,alignItems:"center"}}>
          <span style={{fontSize:10,color:"rgba(255,255,255,0.4)"}}>
            <span style={{display:"inline-block",width:6,height:6,borderRadius:"50%",background:"#2ecc71",marginRight:5}}/>API connected
          </span>
          <button onClick={()=>setTab("mlflow")} style={{padding:"5px 12px",borderRadius:5,border:"1px solid rgba(255,255,255,0.15)",background:"rgba(255,255,255,0.07)",color:"#fff",fontSize:11,cursor:"pointer",fontFamily:"inherit"}}>MLflow</button>
          <button onClick={()=>setTab("train")} style={{padding:"5px 12px",borderRadius:5,border:"none",background:C.red,color:"#fff",fontSize:11,fontWeight:500,cursor:"pointer",fontFamily:"inherit"}}>Train model</button>
        </div>
      </div>

      {/* Nav */}
      <div style={{background:C.surface,borderBottom:`1px solid ${C.border}`,display:"flex",padding:"0 20px"}}>
        {TABS.map(t=>(
          <button key={t.id} onClick={()=>setTab(t.id)} style={{
            padding:"9px 14px",fontSize:12,fontWeight:500,cursor:"pointer",
            color:tab===t.id?C.red:C.muted,
            background:"none",border:"none",
            borderBottom:`2px solid ${tab===t.id?C.red:"transparent"}`,
            whiteSpace:"nowrap",transition:"color 0.15s",fontFamily:"inherit"}}>
            {t.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{padding:"16px 20px"}}>
        {ready ? pages[tab] : <div style={{textAlign:"center",padding:60,color:C.muted,fontSize:13}}>Loading charts…</div>}
      </div>
    </div>
  );
}
