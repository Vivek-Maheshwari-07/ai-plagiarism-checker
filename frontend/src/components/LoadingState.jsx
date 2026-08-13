import React, { useEffect, useState } from 'react';

export default function LoadingState({ sourceType }) {
  const [stepIndex, setStepIndex] = useState(0);

  const pdfSteps = [
    'Reading PDF...',
    'Extracting text...',
    'Preparing document...'
  ];

  const textSteps = [
    'Extracting document...',
    'Preparing text object...'
  ];

  const steps = sourceType === 'pdf' ? pdfSteps : textSteps;

  useEffect(() => {
    const timer = setInterval(() => {
      setStepIndex((prev) => (prev < steps.length - 1 ? prev + 1 : prev));
    }, 400);

    return () => clearInterval(timer);
  }, [steps.length]);

  return (
    <div className="loading-box">
      <div className="spinner"></div>
      <div className="loading-status">{steps[stepIndex]}</div>
      <div className="loading-subtext">Veritas AI is preparing your document payload...</div>
    </div>
  );
}
