import React, { useState } from 'react';
import { Trash2, ArrowRight } from 'lucide-react';

export default function PasteTextInput({ onAnalyze }) {
  const [text, setText] = useState('');

  const charCount = text.length;
  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
  const isTooShort = text.length > 0 && text.trim().length < 50;

  const handleClear = () => {
    setText('');
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (text.trim().length >= 50) {
      onAnalyze(text);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <div className="text-area-container">
        <textarea
          className="custom-textarea"
          placeholder="Paste your assignment, research paper, or academic document here..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
      </div>

      <div className="input-stats-bar">
        <div className="stats-group">
          <div className="stat-item">
            Words: <span>{wordCount.toLocaleString()}</span>
          </div>
          <div className="stat-item">
            Characters: <span>{charCount.toLocaleString()}</span>
          </div>
        </div>

        {isTooShort && (
          <div className="min-length-warning">
            * Minimum 50 characters required (currently {charCount})
          </div>
        )}
      </div>

      <div className="actions-row">
        <button
          type="button"
          className="btn btn-secondary"
          onClick={handleClear}
          disabled={!text}
        >
          <Trash2 size={16} />
          Clear
        </button>

        <button
          type="submit"
          className="btn btn-primary"
          disabled={!text || text.trim().length < 50}
        >
          Analyze Document
          <ArrowRight size={18} />
        </button>
      </div>
    </form>
  );
}
