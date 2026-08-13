import React from 'react';
import { FileText, Upload } from 'lucide-react';

export default function InputMethodSelector({ activeTab, onTabChange }) {
  return (
    <div className="tab-container">
      <button
        type="button"
        className={`tab-button ${activeTab === 'text' ? 'active' : ''}`}
        onClick={() => onTabChange('text')}
      >
        <FileText size={18} />
        Paste Text
      </button>
      <button
        type="button"
        className={`tab-button ${activeTab === 'pdf' ? 'active' : ''}`}
        onClick={() => onTabChange('pdf')}
      >
        <Upload size={18} />
        Upload PDF
      </button>
    </div>
  );
}
