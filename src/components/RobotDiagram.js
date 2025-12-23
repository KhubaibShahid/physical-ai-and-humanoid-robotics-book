import React from 'react';
import styles from './RobotDiagram.module.css';

/**
 * RobotDiagram component for visualizing robot architectures, control flows,
 * and system diagrams in the Physical AI & Humanoid Robotics book.
 *
 * @param {Object} props
 * @param {string} props.title - Diagram title
 * @param {string} props.description - Optional description text
 * @param {React.ReactNode} props.children - Diagram content (SVG, Mermaid, or image)
 * @param {string} props.caption - Optional caption for the diagram
 */
export default function RobotDiagram({ title, description, children, caption }) {
  return (
    <div className={styles.robotDiagram}>
      {title && <h3 className={styles.title}>{title}</h3>}
      {description && <p className={styles.description}>{description}</p>}
      <div className={styles.diagramContainer}>
        {children}
      </div>
      {caption && (
        <p className={styles.caption}>
          <em>Figure: {caption}</em>
        </p>
      )}
    </div>
  );
}
