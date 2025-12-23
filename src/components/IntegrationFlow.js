import React from 'react';
import styles from './IntegrationFlow.module.css';

/**
 * IntegrationFlow component for visualizing how different modules and systems
 * integrate together in the robotics stack (ROS 2, Simulation, Isaac, VLA).
 *
 * @param {Object} props
 * @param {string} props.title - Flow diagram title
 * @param {Array<Object>} props.steps - Array of integration steps
 * @param {string} props.steps[].module - Module name (e.g., "ROS 2", "Gazebo")
 * @param {string} props.steps[].description - What this module does
 * @param {string} props.steps[].icon - Optional emoji or icon
 * @param {string} props.direction - Flow direction: 'vertical' or 'horizontal' (default: 'vertical')
 */
export default function IntegrationFlow({ title, steps = [], direction = 'vertical' }) {
  return (
    <div className={styles.integrationFlow}>
      {title && <h3 className={styles.title}>{title}</h3>}
      <div className={`${styles.flowContainer} ${styles[direction]}`}>
        {steps.map((step, index) => (
          <React.Fragment key={index}>
            <div className={styles.step}>
              {step.icon && <div className={styles.icon}>{step.icon}</div>}
              <div className={styles.module}>{step.module}</div>
              <div className={styles.description}>{step.description}</div>
            </div>
            {index < steps.length - 1 && (
              <div className={styles.arrow}>
                {direction === 'vertical' ? '↓' : '→'}
              </div>
            )}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}
