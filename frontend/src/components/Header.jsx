import React from 'react';
import { ShieldCheck } from 'lucide-react';

export default function Header() {
  return (
    <header className="navbar">
      <div className="navbar-inner">
        <div className="brand">
          <div className="brand-icon">
            <ShieldCheck size={20} />
          </div>
          <div className="brand-title">
            Veritas <span>AI</span>
          </div>
        </div>
        <div className="module-badge">
          Module 1: Document Extraction
        </div>
      </div>
    </header>
  );
}
