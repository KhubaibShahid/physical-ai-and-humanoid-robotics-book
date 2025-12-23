import React, { useState } from 'react';
import styles from './CodeExample.module.css';

/**
 * CodeExample component for displaying code snippets with syntax highlighting,
 * language labels, and a copy-to-clipboard button.
 *
 * @param {Object} props
 * @param {string} props.language - Programming language (e.g., 'python', 'cpp', 'yaml')
 * @param {string} props.title - Optional title for the code example
 * @param {string} props.description - Optional description of what the code does
 * @param {string} props.code - The code content to display
 * @param {string} props.filename - Optional filename to display
 * @param {boolean} props.showLineNumbers - Whether to show line numbers (default: false)
 */
export default function CodeExample({
  language,
  title,
  description,
  code,
  filename,
  showLineNumbers = false,
  children
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    const textToCopy = code || children?.props?.children || '';
    try {
      await navigator.clipboard.writeText(textToCopy);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy code:', err);
    }
  };

  return (
    <div className={styles.codeExample}>
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          {title && <h4 className={styles.title}>{title}</h4>}
          {filename && <span className={styles.filename}>{filename}</span>}
          {language && <span className={styles.language}>{language}</span>}
        </div>
        <button
          className={styles.copyButton}
          onClick={handleCopy}
          aria-label="Copy code to clipboard"
        >
          {copied ? '✓ Copied!' : '📋 Copy'}
        </button>
      </div>
      {description && <p className={styles.description}>{description}</p>}
      <div className={styles.codeContainer}>
        {children || <pre><code>{code}</code></pre>}
      </div>
    </div>
  );
}
