/**
 * Veritas AI API Service
 * Handles communications with the FastAPI backend.
 */

const API_BASE_URL = '/api';

/**
 * Analyzes document input (either plain text or PDF file)
 * @param {Object} payload - { text, file, filename }
 * @returns {Promise<Object>} API response payload
 */
export async function analyzeDocument({ text, file, filename }) {
  try {
    let response;

    if (file) {
      // PDF File Upload (Multipart form-data)
      const formData = new FormData();
      formData.append('file', file);

      response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        body: formData,
      });
    } else {
      // Plain Text Submission (JSON)
      response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: text || '',
          filename: filename || 'pasted_document.txt',
        }),
      });
    }

    const data = await response.json();

    if (!response.ok && !data.error) {
      return {
        success: false,
        error: `Server responded with status ${response.status}`,
      };
    }

    return data;
  } catch (err) {
    console.error('API Integration Error:', err);
    return {
      success: false,
      error: 'Unable to connect to the analysis server. Please make sure the FastAPI backend is running.',
    };
  }
}
