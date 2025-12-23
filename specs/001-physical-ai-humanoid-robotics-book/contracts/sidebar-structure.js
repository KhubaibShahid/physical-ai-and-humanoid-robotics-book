// Docusaurus Sidebar Configuration
// Physical AI & Humanoid Robotics Book
// Generated: 2025-12-22

/**
 * Creating a sidebar enables you to:
 * - create an ordered group of docs
 * - render a sidebar for each doc of that group
 * - provide next/previous navigation
 *
 * The sidebars can be generated from the filesystem, or explicitly defined here.
 *
 * Create as many sidebars as you want.
 */

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  // Main tutorial sidebar
  tutorialSidebar: [
    {
      type: 'doc',
      id: 'intro',
      label: 'Introduction',
    },

    // Module 1: ROS 2 (Robotic Nervous System)
    {
      type: 'category',
      label: 'Module 1: ROS 2',
      link: {
        type: 'generated-index',
        title: 'Module 1: ROS 2 (Robotic Nervous System)',
        description: 'Learn robot software communication and control with ROS 2. Understand nodes, topics, services, and URDF for humanoid robot control.',
        keywords: ['ros2', 'robotics', 'nodes', 'topics', 'urdf'],
      },
      items: [
        'module-01-ros2/01-overview',
        'module-01-ros2/02-architecture',
        'module-01-ros2/03-tooling',
        'module-01-ros2/04-integration',
        'module-01-ros2/05-summary',
      ],
    },

    // Module 2: Digital Twin (Gazebo & Unity)
    {
      type: 'category',
      label: 'Module 2: Digital Twin',
      link: {
        type: 'generated-index',
        title: 'Module 2: Digital Twin (Gazebo & Unity)',
        description: 'Master robot simulation for safe testing and rapid prototyping. Learn physics simulation, environment creation, and sensor modeling in Gazebo and Unity.',
        keywords: ['simulation', 'gazebo', 'unity', 'physics', 'digital-twin'],
      },
      items: [
        'module-02-digital-twin/01-overview',
        'module-02-digital-twin/02-architecture',
        'module-02-digital-twin/03-tooling',
        'module-02-digital-twin/04-integration',
        'module-02-digital-twin/05-summary',
      ],
    },

    // Module 3: AI-Robot Brain (NVIDIA Isaac)
    {
      type: 'category',
      label: 'Module 3: AI Brain',
      link: {
        type: 'generated-index',
        title: 'Module 3: AI-Robot Brain (NVIDIA Isaac)',
        description: 'Understand AI-driven autonomy with NVIDIA Isaac. Explore perception pipelines, visual SLAM, navigation algorithms, and path planning for intelligent robots.',
        keywords: ['nvidia-isaac', 'perception', 'slam', 'navigation', 'ai'],
      },
      items: [
        'module-03-ai-brain/01-overview',
        'module-03-ai-brain/02-architecture',
        'module-03-ai-brain/03-tooling',
        'module-03-ai-brain/04-integration',
        'module-03-ai-brain/05-summary',
      ],
    },

    // Module 4: Vision-Language-Action (VLA)
    {
      type: 'category',
      label: 'Module 4: VLA',
      link: {
        type: 'generated-index',
        title: 'Module 4: Vision-Language-Action',
        description: 'Build end-to-end autonomous behavior with vision-language-action models. Integrate voice commands, LLM planning, and robotic execution in a capstone humanoid project.',
        keywords: ['vla', 'llm', 'voice-control', 'autonomous', 'capstone'],
      },
      items: [
        'module-04-vla/01-overview',
        'module-04-vla/02-architecture',
        'module-04-vla/03-tooling',
        'module-04-vla/04-integration',
        'module-04-vla/05-summary',
        'module-04-vla/06-capstone',
      ],
    },

    // Appendix
    {
      type: 'category',
      label: 'Appendix',
      link: {
        type: 'generated-index',
        title: 'Appendix',
        description: 'Supporting resources including glossary, external references, and troubleshooting guides.',
      },
      items: [
        'appendix/glossary',
        'appendix/resources',
        'appendix/troubleshooting',
      ],
    },
  ],
};

module.exports = sidebars;
