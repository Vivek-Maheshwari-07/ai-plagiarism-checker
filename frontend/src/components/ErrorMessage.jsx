import React from 'react';
import { AlertTriangle } from 'lucide-react';

export default function ErrorMessage({ message }) {
  if (!message) return null;

  return (
    <div className="alert-error">
      <AlertTriangle size={20} className="alert-icon" />
      <div>{message}</div>
    </div>
  );
}
