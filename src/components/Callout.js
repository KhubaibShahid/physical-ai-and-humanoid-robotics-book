import React from 'react';
import styles from './Callout.module.css';

/**
 * Callout component for displaying informational boxes, warnings, tips, and important notes.
 *
 * @param {Object} props
 * @param {string} props.type - Callout type: 'info', 'warning', 'tip', 'danger', 'success'
 * @param {string} props.title - Optional title for the callout
 * @param {React.ReactNode} props.children - Callout content
 * @param {boolean} props.icon - Whether to show an icon (default: true)
 */
export default function Callout({ type = 'info', title, children, icon = true }) {
  const icons = {
    info: 'ℹ️',
    warning: '⚠️',
    tip: '💡',
    danger: '🚨',
    success: '✅',
  };

  const defaultTitles = {
    info: 'Information',
    warning: 'Warning',
    tip: 'Tip',
    danger: 'Important',
    success: 'Success',
  };

  const displayTitle = title || defaultTitles[type];
  const displayIcon = icon ? icons[type] : null;

  return (
    <div className={`${styles.callout} ${styles[type]}`}>
      <div className={styles.header}>
        {displayIcon && <span className={styles.icon}>{displayIcon}</span>}
        <strong className={styles.title}>{displayTitle}</strong>
      </div>
      <div className={styles.content}>{children}</div>
    </div>
  );
}
