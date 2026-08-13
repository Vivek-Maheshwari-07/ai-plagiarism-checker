import React from 'react';
import { CheckCircle2, ArrowRight, RefreshCw, FileText } from 'lucide-react';

export default function DocumentSummaryCard({ documentData, onReset }) {
  if (!documentData) return null;

  const { filename, source_type, word_count, character_count, page_count, text } = documentData;

  const textPreview = text.length > 350 ? text.substring(0, 350) + '...' : text;

  return (
    <div className="summary-container">
      <div className="summary-header-badge">
        <CheckCircle2 size={18} />
        Document processed successfully
      </div>

      <div className="metrics-grid">
        <div className="metric-card">
          <span className="metric-label">Filename</span>
          <span className="metric-value" style={{ fontSize: '1rem', wordBreak: 'break-all' }}>
            {filename}
          </span>
        </div>

        <div className="metric-card">
          <span className="metric-label">Input Source</span>
          <span className="metric-value" style={{ fontSize: '1.25rem', textTransform: 'uppercase' }}>
            {source_type}
          </span>
        </div>

        <div className="metric-card">
          <span className="metric-label">Words</span>
          <span className="metric-value">{word_count?.toLocaleString()}</span>
        </div>

        <div className="metric-card">
          <span className="metric-label">Characters</span>
          <span className="metric-value">{character_count?.toLocaleString()}</span>
        </div>

        <div className="metric-card">
          <span className="metric-label">Pages</span>
          <span className="metric-value">{page_count !== null && page_count !== undefined ? page_count : 'N/A'}</span>
        </div>
      </div>

      <div className="preview-box">
        <div className="preview-title">Extracted Text Preview</div>
        <div className="preview-content">{textPreview}</div>
      </div>

      <div className="actions-row">
        <button type="button" className="btn btn-secondary" onClick={onReset}>
          <RefreshCw size={16} />
          New Document
        </button>

        <button
          type="button"
          className="btn btn-primary"
          onClick={() => alert('Module 1 payload is ready! Module 2 Preprocessing will consume this document object.')}
        >
          Continue Analysis (Module 2)
          <ArrowRight size={18} />
        </button>
      </div>
    </div>
  );
}
