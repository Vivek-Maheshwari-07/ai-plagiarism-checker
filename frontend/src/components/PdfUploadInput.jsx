import React, { useState, useRef } from 'react';
import { FileUp, File, X, ArrowRight, UploadCloud } from 'lucide-react';

export default function PdfUploadInput({ onAnalyze, onError }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDragActive, setIsDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const validateAndSetFile = (file) => {
    if (!file) return;

    // Check extension / format
    if (!file.name.toLowerCase().endsWith('.pdf') && file.type !== 'application/pdf') {
      onError('Please upload a PDF file.');
      return;
    }

    // Check size (10 MB = 10 * 1024 * 1024 bytes)
    if (file.size > 10 * 1024 * 1024) {
      onError('File size must be less than 10 MB.');
      return;
    }

    onError(null); // Clear previous error
    setSelectedFile(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (selectedFile) {
      onAnalyze(selectedFile);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="file"
        ref={fileInputRef}
        accept=".pdf,application/pdf"
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />

      {!selectedFile ? (
        <div
          className={`dropzone ${isDragActive ? 'active' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <div className="dropzone-icon">
            <UploadCloud size={32} />
          </div>
          <div className="dropzone-title">Drag & drop your PDF here</div>
          <div className="dropzone-or">or</div>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={(e) => {
              e.stopPropagation();
              fileInputRef.current?.click();
            }}
          >
            <FileUp size={16} />
            Browse Files
          </button>
          <div className="dropzone-hint">Maximum file size: 10 MB (.pdf documents only)</div>
        </div>
      ) : (
        <div className="file-card">
          <div className="file-info">
            <div className="file-icon">
              <File size={24} />
            </div>
            <div className="file-details">
              <div className="file-name">{selectedFile.name}</div>
              <div className="file-size">{formatFileSize(selectedFile.size)}</div>
            </div>
          </div>

          <button
            type="button"
            className="btn-danger-text"
            onClick={handleRemoveFile}
          >
            Remove
          </button>
        </div>
      )}

      <div className="actions-row">
        <div />
        <button
          type="submit"
          className="btn btn-primary"
          disabled={!selectedFile}
        >
          Analyze Document
          <ArrowRight size={18} />
        </button>
      </div>
    </form>
  );
}
