import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Read the Book 📚
          </Link>
          <Link
            className="button button--primary button--lg"
            to="/docs/module-01-ros2/overview">
            Start Learning 🚀
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="A comprehensive guide to building autonomous humanoid robots">
      <HomepageHeader />
      <main>
        <section className={styles.features}>
          <div className="container">
            <div className="row">
              <div className="col col--4">
                <div className="text--center padding-horiz--md">
                  <h3>🤖 Physical AI</h3>
                  <p>Learn how to build intelligent systems that bridge the gap between digital AI and physical robots.</p>
                </div>
              </div>
              <div className="col col--4">
                <div className="text--center padding-horiz--md">
                  <h3> humanoid</h3>
                  <p>Master the art of creating robots that can walk, manipulate objects, and interact with the world.</p>
                </div>
              </div>
              <div className="col col--4">
                <div className="text--center padding-horiz--md">
                  <h3>🚀 Robotics</h3>
                  <p>Build complete robotic systems with ROS 2, computer vision, and AI integration.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className={styles.highlights}>
          <div className="container padding-vert--lg">
            <div className="row">
              <div className="col col--6">
                <h2>What You'll Learn</h2>
                <ul>
                  <li>ROS 2 fundamentals and advanced robotics concepts</li>
                  <li>Vision-Language-Action (VLA) systems for intelligent robots</li>
                  <li>Computer vision, machine learning, and AI integration</li>
                  <li>Humanoid robot design, control, and programming</li>
                  <li>Real-world deployment and testing strategies</li>
                </ul>
              </div>
              <div className="col col--6">
                <h2>Why This Book?</h2>
                <ul>
                  <li>Practical, hands-on approach with real examples</li>
                  <li>Industry-standard tools and best practices</li>
                  <li>Comprehensive coverage from basics to advanced topics</li>
                  <li>Focus on cutting-edge technologies like VLA systems</li>
                  <li>Ready-to-use code examples and projects</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        <section className={styles.modules}>
          <div className="container padding-vert--lg">
            <h2 style={{textAlign: 'center', marginBottom: '2rem'}}>Book Modules</h2>
            <div className="row">
              <div className="col col--3">
                <div className="text--center padding-horiz--md">
                  <h4>Module 1: ROS 2</h4>
                  <p>Foundation concepts and practical ROS 2 development</p>
                </div>
              </div>
              <div className="col col--3">
                <div className="text--center padding-horiz--md">
                  <h4>Module 2: Computer Vision</h4>
                  <p>Image processing, object detection, and visual perception</p>
                </div>
              </div>
              <div className="col col--3">
                <div className="text--center padding-horiz--md">
                  <h4>Module 3: Control Systems</h4>
                  <p>Robot kinematics, dynamics, and motion control</p>
                </div>
              </div>
              <div className="col col--3">
                <div className="text--center padding-horiz--md">
                  <h4>Module 4: VLA Systems</h4>
                  <p>Vision-Language-Action integration for intelligent robots</p>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}