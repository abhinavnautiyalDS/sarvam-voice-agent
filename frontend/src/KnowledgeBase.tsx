import { useCallback, useEffect, useState } from 'react';
import { AlertCircle, CheckCircle2, ChevronRight, Database, FileText, Layers3, LoaderCircle, Network, RefreshCw, ToggleLeft, ToggleRight } from 'lucide-react';

type KBStatus = { enabled: boolean; documents: number; chunks: number; embedding_model: string; dimensions: number };
type Document = { id: string; title: string; source: string; created_at: string; chunk_count: number };
type Chunk = { id: number; chunk_index: number; content: string };

export default function KnowledgeBase() {
  const [status, setStatus] = useState<KBStatus | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selected, setSelected] = useState('');
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [loading, setLoading] = useState(true);
  const [chunkLoading, setChunkLoading] = useState(false);
  const [error, setError] = useState('');
  const [toggleBusy, setToggleBusy] = useState(false);

  const load = useCallback(async () => {
    setLoading(true); setError('');
    try {
      const [statusResponse, docsResponse] = await Promise.all([
        fetch('/api/knowledge/status'), fetch('/api/knowledge/documents')
      ]);
      if (!statusResponse.ok || !docsResponse.ok) throw new Error('Knowledge Base API is not available yet.');
      const statusData: KBStatus = await statusResponse.json();
      const docsData: { documents: Document[] } = await docsResponse.json();
      setStatus(statusData); setDocuments(docsData.documents || []);
      setSelected(current => current && docsData.documents.some(d => d.id === current) ? current : (docsData.documents[0]?.id || ''));
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not load Knowledge Base'); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { void load(); }, [load]);
  useEffect(() => {
    if (!selected) { setChunks([]); return; }
    let active = true;
    setChunkLoading(true);
    fetch(`/api/knowledge/documents/${encodeURIComponent(selected)}/chunks`)
      .then(async response => { if (!response.ok) throw new Error('Could not load document chunks'); return response.json(); })
      .then(data => { if (active) setChunks(data.chunks || []); })
      .catch(e => { if (active) setError(e instanceof Error ? e.message : 'Could not load chunks'); })
      .finally(() => { if (active) setChunkLoading(false); });
    return () => { active = false; };
  }, [selected]);

  const toggle = async () => {
    if (!status) return;
    setToggleBusy(true); setError('');
    try {
      const response = await fetch('/api/admin/knowledge/enabled', {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: !status.enabled })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Toggle requires an allowed owner IP.');
      setStatus(data);
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not update KB setting'); }
    finally { setToggleBusy(false); }
  };

  const activeDoc = documents.find(d => d.id === selected);
  return <section className="panel page-panel">
    <div className="section-head"><div><span className="eyebrow">CENTRAL KNOWLEDGE STORE</span><h3>Knowledge Base</h3><p>Browse owner-indexed documents and inspect the exact text chunks stored for retrieval.</p></div><span className="readonly">READ ONLY VIEW</span></div>
    <div className="kb-live-toolbar">
      <div className="kb-live-status"><Database/><span><b>{status ? `${status.documents} documents · ${status.chunks} chunks` : 'Knowledge store'}</b><small>{status ? `${status.embedding_model} · ${status.dimensions} dimensions` : 'Waiting for backend status'}</small></span></div>
      <div className="kb-enable"><span><b>Knowledge retrieval</b><small>{status?.enabled ? 'Enabled for voice sessions' : 'Disabled by default'}</small></span><button onClick={() => void toggle()} disabled={!status || toggleBusy} title="Toggle knowledge retrieval">{toggleBusy ? <LoaderCircle className="spin"/> : status?.enabled ? <ToggleRight/> : <ToggleLeft/>}<b>{status?.enabled ? 'ON' : 'OFF'}</b></button></div>
      <button className="kb-refresh" onClick={() => void load()} disabled={loading}><RefreshCw className={loading ? 'spin' : ''}/> Refresh</button>
    </div>
    {error && <div className="kb-error"><AlertCircle/>{error}</div>}
    <div className="kb-content-layout">
      <div className="kb-source-list"><div className="kb-subhead"><b><FileText/> Indexed documents</b><span>{documents.length} SOURCES</span></div>
        {loading ? <div className="kb-source-empty"><LoaderCircle className="spin"/><b>Loading knowledge store…</b></div> :
          documents.length ? <div className="kb-documents">{documents.map(doc => <button key={doc.id} className={selected === doc.id ? 'kb-document selected' : 'kb-document'} onClick={() => setSelected(doc.id)}><FileText/><span><b>{doc.title}</b><small>{doc.source} · {doc.chunk_count} chunks</small></span><ChevronRight/></button>)}</div> :
          <div className="kb-source-empty"><Layers3/><b>No indexed documents</b><p>Documents added by the project owner will appear here. Uploading and re-indexing are intentionally not available in this UI.</p></div>}
      </div>
      <div className="kb-document-view"><div className="kb-subhead"><b><FileText/> Stored content</b><span>{activeDoc ? `${chunks.length} CHUNKS` : 'SELECT A DOCUMENT'}</span></div>
        {!selected ? <div className="kb-document-empty"><Database/><h3>Knowledge content will appear here</h3><p>Select an indexed document to read its extracted text in the same chunk format used by the retrieval index.</p></div> :
          <div className="kb-chunks">{activeDoc && <div className="kb-doc-heading"><b>{activeDoc.title}</b><small>{activeDoc.source}</small></div>}
            {chunkLoading ? <div className="kb-source-empty"><LoaderCircle className="spin"/> Loading chunks…</div> :
              chunks.map(chunk => <article className="kb-chunk" key={chunk.id}><header><span>CHUNK {String(chunk.chunk_index + 1).padStart(3, '0')}</span><small>ID {chunk.id}</small></header><p>{chunk.content}</p></article>)}
            {!chunkLoading && !chunks.length && <p className="kb-muted">No chunks are stored for this document.</p>}
          </div>}
      </div>
    </div>
    <div className="kb-footnote"><CheckCircle2/><span>Browsing is read-only. No upload, attach, embedding, or re-index controls are exposed here.</span><span className="kb-faiss"><Network/> FAISS + SQLite</span></div>
  </section>;
}
