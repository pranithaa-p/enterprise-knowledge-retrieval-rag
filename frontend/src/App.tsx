
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import "./App.css";

type DocumentItem = {
  document_id: number;
  filename: string;
  [key: string]: unknown;
};

type SearchResult = {
  document_id: number;
  filename: string;
  chunk_id: number;
  text: string;
  rrf_score?: number;
  semantic_score?: number;
};

type HistoryItem = {
  query: string;
  time: string;
};

const API = "/api";

function renderInlineMarkdown(text: string) {
  const tokens = text.split(/(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|\[[^\]]+\]\([^)]+\))/g);

  return tokens.map((token, index) => {
    if (token.startsWith("`") && token.endsWith("`")) {
      return <code key={index}>{token.slice(1, -1)}</code>;
    }

    if (token.startsWith("**") && token.endsWith("**")) {
      return <strong key={index}>{token.slice(2, -2)}</strong>;
    }

    if (token.startsWith("*") && token.endsWith("*")) {
      return <em key={index}>{token.slice(1, -1)}</em>;
    }

    const link = token.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
    if (link) {
      return (
        <a key={index} href={link[2]} target="_blank" rel="noreferrer">
          {link[1]}
        </a>
      );
    }

    return <span key={index}>{token}</span>;
  });
}

function MarkdownAnswer({ content }: { content: string }) {
  const lines = content.replace(/\r\n/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  let paragraph: string[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;
  let codeLines: string[] = [];
  let inCode = false;

  const flushParagraph = () => {
    if (paragraph.length) {
      blocks.push(
        <p key={`paragraph-${blocks.length}`}>
          {renderInlineMarkdown(paragraph.join(" "))}
        </p>
      );
      paragraph = [];
    }
  };

  const flushList = () => {
    if (!list) return;
    const List = list.ordered ? "ol" : "ul";
    blocks.push(
      <List key={`list-${blocks.length}`}>
        {list.items.map((item, index) => (
          <li key={`${index}-${item}`}>{renderInlineMarkdown(item)}</li>
        ))}
      </List>
    );
    list = null;
  };

  lines.forEach((line) => {
    if (line.trim().startsWith("```")) {
      flushParagraph();
      flushList();
      if (inCode) {
        blocks.push(
          <pre key={`code-${blocks.length}`}>
            <code>{codeLines.join("\n")}</code>
          </pre>
        );
        codeLines = [];
        inCode = false;
      } else {
        inCode = true;
      }
      return;
    }

    if (inCode) {
      codeLines.push(line);
      return;
    }

    const heading = line.match(/^\s{0,3}(#{1,3})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      flushList();
      const Heading = `h${heading[1].length}` as "h1" | "h2" | "h3";
      blocks.push(
        <Heading key={`heading-${blocks.length}`}>
          {renderInlineMarkdown(heading[2])}
        </Heading>
      );
      return;
    }

    const bullet = line.match(/^\s*[-*+]\s+(.+)$/);
    const orderedItem = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (bullet || orderedItem) {
      flushParagraph();
      const ordered = Boolean(orderedItem);
      if (!list || list.ordered !== ordered) {
        flushList();
        list = { ordered, items: [] };
      }
      list.items.push((bullet ?? orderedItem)![1]);
      return;
    }

    if (!line.trim()) {
      flushParagraph();
      flushList();
      return;
    }

    flushList();
    paragraph.push(line.trim());
  });

  if (inCode) {
    blocks.push(
      <pre key={`code-${blocks.length}`}>
        <code>{codeLines.join("\n")}</code>
      </pre>
    );
  }
  flushParagraph();
  flushList();

  return <div className="ai-answer-content">{blocks}</div>;
}

async function apiRequest(path: string, options?: RequestInit) {
  const response = await fetch(`${API}${path}`, {
    ...options,
    headers: {
      Accept: "application/json",
      ...options?.headers,
    },
  });

  const responseText = await response.text();

  let data: any = null;

  try {
    data = responseText ? JSON.parse(responseText) : null;
  } catch {
    data = responseText;
  }

  if (!response.ok) {
    const detail = data?.detail ?? data?.message;

    let message: string;

    if (typeof detail === "string") {
      message = detail;
    } else if (Array.isArray(detail)) {
      message = detail
        .map((item) =>
          typeof item === "string"
            ? item
            : item?.msg ?? JSON.stringify(item)
        )
        .join("; ");
    } else if (detail != null) {
      message = JSON.stringify(detail);
    } else if (typeof data === "string" && data.trim()) {
      message = data;
    } else {
      message = `Request failed with status ${response.status}`;
    }

    throw new Error(message);
  }

  return data;
}

function App() {
  const [theme, setTheme] = useState<"light" | "dark">(
    () => (localStorage.getItem("ekrp-theme") as "light" | "dark") || "light"
  );
  const [page, setPage] = useState("Search");
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [aiAnswer, setAiAnswer] = useState("");
  const [history, setHistory] = useState<HistoryItem[]>(() => {
    try {
      return JSON.parse(localStorage.getItem("ekrp-history") || "[]");
    } catch {
      return [];
    }
  });
  const [selected, setSelected] = useState<SearchResult | null>(null);
  const [selectedDocument, setSelectedDocument] = useState<DocumentItem | null>(null);
  const [previewChunks, setPreviewChunks] = useState<any[]>([]);
  const [files, setFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [topK, setTopK] = useState(5);
  const [sourceFilter, setSourceFilter] = useState("All documents");
  const [minScore, setMinScore] = useState(0);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("ekrp-theme", theme);
  }, [theme]);

  async function loadDocuments() {
    try {
      const data = await apiRequest("/documents");
      const list = Array.isArray(data)
        ? data
        : data?.documents ?? data?.results ?? [];
      setDocuments(list);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load documents.");
    }
  }

  useEffect(() => {
    void loadDocuments();
  }, []);

  const filteredResults = useMemo(() => {
    return results.filter((item) => {
      const matchesDocument =
        sourceFilter === "All documents" || item.filename === sourceFilter;
      const matchesScore = (item.rrf_score ?? 0) >= minScore;
      return matchesDocument && matchesScore;
    });
  }, [results, sourceFilter, minScore]);

  async function uploadFiles() {
    if (!files.length) {
      setError("Choose at least one PDF or DOCX file.");
      return;
    }

    setBusy(true);
    setError("");
    setMessage("");

    try {
      const body = new FormData();
      files.forEach((file) => body.append("files", file));

      // Batch endpoint accepts multiple files in one request.
      const data = await apiRequest("/documents/batch-process", {
        method: "POST",
        body,
      });

      setMessage(
        data?.message ||
          `Upload request completed for ${files.length} file(s).`
      );
      setFiles([]);
      if (fileInput.current) fileInput.current.value = "";
      await loadDocuments();
    } catch (e) {
      setError(
        e instanceof Error
          ? `${e.message}. Check that POST /documents/batch-process exists in your FastAPI backend.`
          : "Upload failed."
      );
    } finally {
      setBusy(false);
    }
  }

  async function search() {
    const cleaned = query.trim();
    if (!cleaned) {
      setError("Enter a question or keywords to search.");
      return;
    }

    setBusy(true);
    setError("");
    setMessage("");
    setSelected(null);
    setAiAnswer("");

    try {
      // One hybrid endpoint combines keyword and semantic retrieval.
      const data = await apiRequest("/search/hybrid", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: cleaned, top_k: topK }),
      });

      const list: SearchResult[] = data?.results ?? [];
      setResults(list);
      setAiAnswer(
        typeof data?.ai_answer === "string" ? data.ai_answer : ""
      );
      setSourceFilter("All documents");
      setMinScore(0);

      const nextHistory = [
        { query: cleaned, time: new Date().toLocaleString() },
        ...history.filter((item) => item.query !== cleaned),
      ].slice(0, 20);

      setHistory(nextHistory);
      localStorage.setItem("ekrp-history", JSON.stringify(nextHistory));

      setMessage(
        data?.message ||
          (list.length
            ? `${list.length} source(s) found.`
            : "No matching sources were found.")
      );
      setPage("Search");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Search failed.");
    } finally {
      setBusy(false);
    }
  }

  async function openDocument(doc: DocumentItem) {
    setSelectedDocument(doc);
    setSelected(null);
    setPreviewChunks([]);
    setError("");

    try {
      const data = await apiRequest(
        `/documents/${doc.document_id}/chunks`
      );
      setPreviewChunks(
        Array.isArray(data) ? data : data?.chunks ?? data?.results ?? []
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load preview.");
    }
  }

  function chooseHistory(item: HistoryItem) {
    setQuery(item.query);
    setPage("Search");
  }

  function addFiles(incoming: FileList | File[]) {
    const allowed = Array.from(incoming).filter((file) =>
      /\.(pdf|docx)$/i.test(file.name)
    );

    setFiles((previous) => {
      const merged = [...previous];
      for (const file of allowed) {
        if (!merged.some((existing) =>
          existing.name === file.name && existing.size === file.size
        )) {
          merged.push(file);
        }
      }
      return merged;
    });

    setError("");
    if (!allowed.length) {
      setError("Only PDF and DOCX files are supported by this interface.");
    }
  }

  function closePreview() {
    setSelected(null);
    setSelectedDocument(null);
    setPreviewChunks([]);
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">K</div>
          <div>
            <strong>Knowledge Portal</strong>
            <span>Enterprise Search</span>
          </div>
        </div>

        <div className="side-label">WORKSPACE</div>
        {[
          ["Search", "⌕"],
          ["Documents", "▤"],
          ["History", "◷"],
          ["Settings", "⚙"],
        ].map(([name, symbol]) => (
          <button
            key={name}
            className={`nav-item ${page === name ? "active" : ""}`}
            onClick={() => {
              setPage(name);
              setError("");
            }}
          >
            <span className="nav-symbol">{symbol}</span>
            {name}
            {name === "Documents" && (
              <span className="nav-count">{documents.length}</span>
            )}
          </button>
        ))}

        <div className="sidebar-bottom">
          <div className="system-status">
            <span className="status-dot" />
            <div>
              <strong>Workspace</strong>
              <span>Local API connection</span>
            </div>
          </div>
          <button
            className="theme-toggle"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          >
            <span>{theme === "dark" ? "☀" : "☾"}</span>
            {theme === "dark" ? "Light appearance" : "Dark appearance"}
          </button>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumb">Workspace <span>/</span> {page}</div>
          <div className="topbar-right">
            <span className="connection"><span className="status-dot" /> API workspace</span>
            <button
              className="icon-button"
              title="Switch theme"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            >
              {theme === "dark" ? "☀" : "☾"}
            </button>
          </div>
        </header>

        <div className="page-content">
          {error && (
            <div className="notice error-notice">
              <span>{error}</span>
              <button onClick={() => setError("")}>×</button>
            </div>
          )}
          {message && (
            <div className="notice success-notice">
              <span>{message}</span>
              <button onClick={() => setMessage("")}>×</button>
            </div>
          )}

          {page === "Search" && (
            <>
              <div className="page-heading">
                <div>
                  <div className="eyebrow">KNOWLEDGE WORKSPACE</div>
                  <h1>Search your knowledge</h1>
                  <p>Find relevant information across your uploaded documents.</p>
                </div>
                <button className="secondary-button" onClick={() => setPage("Documents")}>
                  + Manage documents
                </button>
              </div>

              <section className="search-panel">
                <label className="field-label" htmlFor="search-query">
                  Ask a question or enter keywords
                </label>
                <div className="search-input-row">
                  <span className="search-icon">⌕</span>
                  <input
                    id="search-query"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !busy) void search();
                    }}
                    placeholder="e.g. What is the leave approval policy?"
                  />
                  <button className="primary-button" onClick={() => void search()} disabled={busy}>
                    {busy ? "Working…" : "Search"}
                    {!busy && <span>↗</span>}
                  </button>
                </div>
                <div className="search-footer">
                  <span><span className="mini-dot" /> Hybrid retrieval enabled</span>
                  <label>
                    Results
                    <select value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
                      <option value={3}>3</option>
                      <option value={5}>5</option>
                      <option value={10}>10</option>
                      <option value={20}>20</option>
                    </select>
                  </label>
                </div>
              </section>

              {aiAnswer && (
                <section className="ai-answer-panel">
                  <div className="section-heading">
                    <div>
                      <div className="eyebrow">AI ANSWER</div>
                      <h2>Answer from your documents</h2>
                    </div>
                  </div>
                  <MarkdownAnswer content={aiAnswer} />
                </section>
              )}

              <div className="section-heading results-heading">
                <div>
                  <h2>Supporting evidence</h2>
                  <p>Raw document excerpts used to generate the answer.</p>
                </div>
                <span className="result-count">{filteredResults.length} results</span>
              </div>

              {results.length > 0 && (
                <div className="filters-row">
                  <label>
                    Document
                    <select value={sourceFilter} onChange={(e) => setSourceFilter(e.target.value)}>
                      <option>All documents</option>
                      {[...new Set(results.map((item) => item.filename))].map((name) => (
                        <option key={name}>{name}</option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Minimum RRF score
                    <select value={minScore} onChange={(e) => setMinScore(Number(e.target.value))}>
                      <option value={0}>Any score</option>
                      <option value={0.01}>0.01+</option>
                      <option value={0.02}>0.02+</option>
                      <option value={0.03}>0.03+</option>
                    </select>
                  </label>
                  <button className="text-button" onClick={() => { setSourceFilter("All documents"); setMinScore(0); }}>
                    Reset filters
                  </button>
                </div>
              )}

              {filteredResults.length > 0 ? (
                <div className="results-list">
                  {filteredResults.map((item, index) => (
                    <article className="result-card" key={`${item.document_id}-${item.chunk_id}`}>
                      <div className="result-number">{String(index + 1).padStart(2, "0")}</div>
                      <div className="result-body">
                        <div className="source-meta">
                          <span className="file-chip">PDF / DOC</span>
                          <span>{item.filename}</span>
                          <span>·</span>
                          <span>Chunk {item.chunk_id}</span>
                        </div>
                        <p className="result-text">{item.text}</p>
                        <div className="result-footer">
                          <span>RRF {item.rrf_score?.toFixed(5) ?? "—"}</span>
                          <span>Semantic {item.semantic_score?.toFixed(3) ?? "—"}</span>
                        </div>
                      </div>
                      <button className="open-source" onClick={() => setSelected(item)}>
                        View source ↗
                      </button>
                    </article>
                  ))}
                </div>
              ) : (
                <div className="empty-state">
                  <div className="empty-icon">⌕</div>
                  <h3>{results.length ? "No sources match these filters" : "Your sources will appear here"}</h3>
                  <p>
                    {results.length
                      ? "Adjust the filters to see more results."
                      : "Search across your uploaded documents using a question or keywords."}
                  </p>
                </div>
              )}

            </>
          )}

          {page === "Documents" && (
            <>
              <div className="page-heading">
                <div>
                  <div className="eyebrow">DOCUMENT LIBRARY</div>
                  <h1>Manage documents</h1>
                  <p>Upload multiple files and search their extracted content together.</p>
                </div>
                <span className="result-count">{documents.length} indexed documents</span>
              </div>

              <section
                className={`upload-zone ${dragging ? "dragging" : ""}`}
                onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragging(false);
                  addFiles(e.dataTransfer.files);
                }}
              >
                <input
                  ref={fileInput}
                  type="file"
                  accept=".pdf,.docx"
                  multiple
                  hidden
                  onChange={(e) => {
                    if (e.target.files) addFiles(e.target.files);
                  }}
                />
                <div className="upload-icon">↑</div>
                <h2>Upload documents</h2>
                <p>Drag and drop files here, or choose them from your computer.</p>
                <p className="muted-small">PDF and DOCX · Multiple files supported</p>
                <button className="secondary-button" onClick={() => fileInput.current?.click()}>
                  Choose files
                </button>

                {files.length > 0 && (
                  <div className="selected-files">
                    <div className="selected-files-heading">{files.length} file(s) selected</div>
                    {files.map((file) => (
                      <div className="selected-file" key={`${file.name}-${file.size}`}>
                        <span>▤</span>
                        <span>{file.name}</span>
                        <small>{(file.size / 1024).toFixed(0)} KB</small>
                        <button onClick={() => setFiles((old) => old.filter((f) => f !== file))}>×</button>
                      </div>
                    ))}
                    <button className="primary-button upload-button" disabled={busy} onClick={() => void uploadFiles()}>
                      {busy ? "Processing files…" : `Upload and process ${files.length} file(s)`}
                    </button>
                  </div>
                )}
              </section>

              <div className="section-heading">
                <div>
                  <h2>Indexed documents</h2>
                  <p>Documents currently available to the search service.</p>
                </div>
                <button className="secondary-button" onClick={() => void loadDocuments()}>Refresh</button>
              </div>

              <div className="document-list">
                {documents.map((doc) => (
                  <div className="document-row" key={doc.document_id}>
                    <div className="document-icon">▤</div>
                    <div className="document-info">
                      <strong>{doc.filename}</strong>
                      <span>Document ID: {doc.document_id}</span>
                    </div>
                    <span className="indexed-label"><span className="status-dot" /> Available</span>
                    <button className="secondary-button small-button" onClick={() => void openDocument(doc)}>
                      Preview
                    </button>
                  </div>
                ))}
                {!documents.length && (
                  <div className="empty-state compact">
                    <h3>No documents listed</h3>
                    <p>Upload your first document to begin.</p>
                  </div>
                )}
              </div>
            </>
          )}

          {page === "History" && (
            <>
              <div className="page-heading">
                <div>
                  <div className="eyebrow">RECENT ACTIVITY</div>
                  <h1>Search history</h1>
                  <p>Return to a previous search without typing it again.</p>
                </div>
                <button className="secondary-button" onClick={() => {
                  setHistory([]);
                  localStorage.removeItem("ekrp-history");
                }}>Clear history</button>
              </div>
              <div className="document-list">
                {history.map((item, index) => (
                  <button className="history-row" key={`${item.query}-${index}`} onClick={() => chooseHistory(item)}>
                    <span className="history-icon">◷</span>
                    <span className="history-query">{item.query}<small>{item.time}</small></span>
                    <span className="history-open">Run search ↗</span>
                  </button>
                ))}
                {!history.length && <div className="empty-state"><h3>No recent searches</h3><p>Your searches will be saved in this browser.</p></div>}
              </div>
            </>
          )}

          {page === "Settings" && (
            <>
              <div className="page-heading">
                <div>
                  <div className="eyebrow">PREFERENCES</div>
                  <h1>Settings</h1>
                  <p>Configure your workspace appearance and search preferences.</p>
                </div>
              </div>
              <section className="settings-card">
                <div>
                  <h3>Appearance</h3>
                  <p>Choose the theme for your workspace.</p>
                </div>
                <div className="theme-options">
                  <button className={theme === "light" ? "theme-option selected" : "theme-option"} onClick={() => setTheme("light")}>☀ Light</button>
                  <button className={theme === "dark" ? "theme-option selected" : "theme-option"} onClick={() => setTheme("dark")}>☾ Dark</button>
                </div>
              </section>
              <section className="settings-card">
                <div>
                  <h3>Search service</h3>
                  <p>Search requests use the FastAPI hybrid retrieval endpoint.</p>
                  <code>POST /search/hybrid</code>
                </div>
              </section>
              <section className="settings-card">
                <div>
                  <h3>Document service</h3>
                  <p>Uploads use the batch processing endpoint.</p>
                  <code>POST /documents/batch-process</code>
                </div>
              </section>
            </>
          )}
        </div>
      </main>

      {(selected || selectedDocument) && (
        <div className="modal-backdrop" onClick={closePreview}>
          <section className="preview-modal" onClick={(e) => e.stopPropagation()}>
            <header className="preview-header">
              <div>
                <div className="eyebrow">SOURCE PREVIEW</div>
                <h2>{selected?.filename ?? selectedDocument?.filename}</h2>
              </div>
              <button className="icon-button" onClick={closePreview}>×</button>
            </header>

            {selected && (
              <>
                <div className="preview-meta">Document ID {selected.document_id} · Chunk {selected.chunk_id}</div>
                <div className="preview-text">{selected.text}</div>
              </>
            )}

            {selectedDocument && (
              <div className="preview-chunks">
                {previewChunks.length ? previewChunks.map((chunk, index) => (
                  <article className="chunk-preview" key={chunk.chunk_id ?? chunk.chunk_index ?? index}>
                    <strong>Chunk {chunk.chunk_id ?? chunk.chunk_index ?? index + 1}</strong>
                    <p>{chunk.text ?? JSON.stringify(chunk)}</p>
                  </article>
                )) : <p className="muted-small">No preview chunks available, or the document has no extracted text.</p>}
              </div>
            )}
          </section>
        </div>
      )}
    </div>
  );
}

export default App;
