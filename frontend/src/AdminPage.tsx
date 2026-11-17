import { useEffect, useState } from 'react';
import { AudioLines, ArrowLeft, CheckCircle2, KeyRound, RefreshCw, Save, Server, ShieldCheck, Volume2, BrainCircuit } from 'lucide-react';
import './admin.css';

type Config = {
  sarvam_api_key_set: boolean; bakbak_api_key_set: boolean; bakbak_voice_id_set: boolean;
  sarvam_llm_model: string; bakbak_tts_model: string; sarvam_stt_mode: string;
  llm_provider: string; tts_provider: string;
};
const empty: Config = {
  sarvam_api_key_set:false,bakbak_api_key_set:false,bakbak_voice_id_set:false,
  sarvam_llm_model:'sarvam-105b-conversations',bakbak_tts_model:'m1',
  sarvam_stt_mode:'verbose',llm_provider:'sarvam',tts_provider:'bakbak'
};
export default function AdminPage(){
  const [config,setConfig]=useState<Config>(empty);
  const [keys,setKeys]=useState({SARVAM_API_KEY:'',BAKBAK_API_KEY:'',BAKBAK_VOICE_ID:''});
  const [models,setModels]=useState({SARVAM_LLM_MODEL:empty.sarvam_llm_model,BAKBAK_TTS_MODEL:empty.bakbak_tts_model,SARVAM_STT_MODE:'verbose',AURA_LLM_PROVIDER:'sarvam',AURA_TTS_PROVIDER:'bakbak'});
  const [status,setStatus]=useState('Loading configuration…');
  const [busy,setBusy]=useState(false);
  const load=async()=>{
    setBusy(true);setStatus('Connecting to AURA backend…');
    try{
      const response=await fetch('/api/admin/config');
      const data=await response.json();
      if(!response.ok) throw new Error(data.detail||'Could not load admin configuration');
      setConfig(data);
      setModels({SARVAM_LLM_MODEL:data.sarvam_llm_model,SARVAM_STT_MODE:data.sarvam_stt_mode,BAKBAK_TTS_MODEL:data.bakbak_tts_model,AURA_LLM_PROVIDER:data.llm_provider,AURA_TTS_PROVIDER:data.tts_provider});
      setStatus('Configuration loaded. Secret values are never returned to the browser.');
    }catch(e){setStatus(e instanceof Error?e.message:'Connection failed');}
    finally{setBusy(false);}
  };
  useEffect(()=>{void load();},[]);
  const save=async()=>{
    setBusy(true);setStatus('Saving securely on the backend…');
    const payload={...models,...Object.fromEntries(Object.entries(keys).filter(([,v])=>v.trim()).map(([k,v])=>[k,v.trim()]))};
    try{
      const response=await fetch('/api/admin/config',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
      const data=await response.json();
      if(!response.ok) throw new Error(data.detail||'Save failed');
      setConfig(data.config);setKeys({SARVAM_API_KEY:'',BAKBAK_API_KEY:'',BAKBAK_VOICE_ID:''});
      setStatus('Saved. New voice sessions will use the updated configuration.');
    }catch(e){setStatus(e instanceof Error?e.message:'Save failed');}
    finally{setBusy(false);}
  };
  const field=(label:string,value:string,onChange:(v:string)=>void,placeholder:string,type='text')=><label className="admin-field"><span>{label}</span><input type={type} value={value} onChange={e=>onChange(e.target.value)} placeholder={placeholder} autoComplete="off"/></label>;
  return <div className="aura-app admin-app"><header className="admin-top"><a href="/"><ArrowLeft size={16}/> Back to Playground</a><span><AudioLines/> AURA CONTROL CENTER</span><b><i/> ADMIN</b></header>
    <main className="admin-main"><div className="admin-heading"><div><small>OWNER CONFIGURATION</small><h1>Control Center</h1><p>Manage voice providers and credentials from one place.</p></div><button onClick={()=>void load()} disabled={busy}><RefreshCw size={14}/> Refresh</button></div>
      <div className="admin-warning"><ShieldCheck/><div><b>Restricted admin endpoint</b><p>This page has no login. Backend changes are accepted only from IP addresses listed in AURA_ADMIN_ALLOWED_IPS. Secret values are write-only and stored on the server.</p></div></div>
      <section className="admin-card"><div className="admin-card-title"><KeyRound/><div><h2>Provider credentials</h2><p>Leave a secret field blank to keep its existing value.</p></div></div>
        {field('Sarvam API key',keys.SARVAM_API_KEY,v=>setKeys(s=>({...s,SARVAM_API_KEY:v})),config.sarvam_api_key_set?'Configured · enter a new key to rotate':'Enter Sarvam API key','password')}
        {field('BakBak API key',keys.BAKBAK_API_KEY,v=>setKeys(s=>({...s,BAKBAK_API_KEY:v})),config.bakbak_api_key_set?'Configured · enter a new key to rotate':'Enter BakBak API key','password')}
        {field('BakBak Voice ID',keys.BAKBAK_VOICE_ID,v=>setKeys(s=>({...s,BAKBAK_VOICE_ID:v})),config.bakbak_voice_id_set?'Configured · enter a new voice ID to change':'Enter voice ID')}
      </section>
      <section className="admin-card"><div className="admin-card-title"><BrainCircuit/><div><h2>Speech & language models</h2><p>STT mode and LLM model used for new sessions.</p></div></div>
        {field('STT mode',models.SARVAM_STT_MODE,v=>setModels(s=>({...s,SARVAM_STT_MODE:v})),'verbose')}
        {field('LLM model',models.SARVAM_LLM_MODEL,v=>setModels(s=>({...s,SARVAM_LLM_MODEL:v})),'sarvam-105b-conversations')}
        {field('LLM provider label',models.AURA_LLM_PROVIDER,v=>setModels(s=>({...s,AURA_LLM_PROVIDER:v})),'sarvam')}
      </section>
      <section className="admin-card"><div className="admin-card-title"><Volume2/><div><h2>Text-to-speech</h2><p>BakBak model configuration.</p></div></div>
        {field('TTS model',models.BAKBAK_TTS_MODEL,v=>setModels(s=>({...s,BAKBAK_TTS_MODEL:v})),'m1')}
        {field('TTS provider label',models.AURA_TTS_PROVIDER,v=>setModels(s=>({...s,AURA_TTS_PROVIDER:v})),'bakbak')}
      </section>
      <div className="admin-save"><span className="admin-status">{status}</span><button onClick={()=>void save()} disabled={busy}><Save size={15}/>{busy?'Please wait':'Save configuration'}</button></div>
      <div className="admin-foot"><Server/> Configuration applies to newly created voice sessions. Restart the API service only if you change server-level environment variables.</div>
    </main></div>;
}
