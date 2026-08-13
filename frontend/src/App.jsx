import React, { useState } from 'react';
import Header from './components/Header';
import InputMethodSelector from './components/InputMethodSelector';
import PasteTextInput from './components/PasteTextInput';
import PdfUploadInput from './components/PdfUploadInput';
import LoadingState from './components/LoadingState';
import DocumentSummaryCard from './components/DocumentSummaryCard';
import ErrorMessage from './components/ErrorMessage';
import { analyzeDocument } from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('text'); // 'text' | 'pdf'
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [extractedDocument, setExtractedDocument] = useState(null);

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    setError(null);
  };

  const handleAnalyzeText = async (text) => {
    setLoading(true);
    setError(null);
    setExtractedDocument(null);

    const response = await analyzeDocument({ text, filename: 'pasted_document.txt' });

    setLoading(false);
    if (response.success) {
      setExtractedDocument(response.document);
    } else {
      setError(response.error || 'An unexpected error occurred during document extraction.');
    }
  };

  const handleAnalyzePdf = async (file) => {
    setLoading(true);
    setError(null);
    setExtractedDocument(null);

    const response = await analyzeDocument({ file });

    setLoading(false);
    if (response.success) {
      setExtractedDocument(response.document);
    } else {
      setError(response.error || 'An unexpected error occurred during PDF extraction.');
    }
  };

  const handleReset = () => {
    setExtractedDocument(null);
    setError(null);
    setLoading(false);
  };

  return (
    <div>
      <Header />

      <main className="app-container">
        <div className="page-header">
          <h1 className="page-title">Analyze Document</h1>
          <p className="page-subtitle">
            Upload your academic research paper or assignment PDF, or paste text directly for automated extraction.
          </p>
        </div>

        <div className="main-card">
          {!extractedDocument && !loading && (
            <InputMethodSelector
              activeTab={activeTab}
              onTabChange={handleTabChange}
            />
          )}

          <ErrorMessage message={error} />

          {loading ? (
            <LoadingState sourceType={activeTab} />
          ) : extractedDocument ? (
            <DocumentSummaryCard
              documentData={extractedDocument}
              onReset={handleReset}
            />
          ) : activeTab === 'text' ? (
            <PasteTextInput onAnalyze={handleAnalyzeText} />
          ) : (
            <PdfUploadInput
              onAnalyze={handleAnalyzePdf}
              onError={(err) => setError(err)}
            />
          )}
        </div>
      </main>

      <footer className="footer">
        Veritas AI — Document Input & Extraction Module (TCS × CHARUSAT Hackathon Use Case 29)
      </footer>
    </div>
  );
}
