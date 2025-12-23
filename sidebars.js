// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
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
        'module-01-ros2/overview',
        'module-01-ros2/architecture',
        'module-01-ros2/tooling',
        'module-01-ros2/integration',
        'module-01-ros2/summary',
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
        'module-02-digital-twin/overview',
        'module-02-digital-twin/architecture',
        'module-02-digital-twin/tooling',
        'module-02-digital-twin/integration',
        'module-02-digital-twin/summary',
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
        'module-03-ai-brain/overview',
        'module-03-ai-brain/architecture',
        'module-03-ai-brain/tooling',
        'module-03-ai-brain/integration',
        'module-03-ai-brain/summary',
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
        'module-04-vla/overview',
        'module-04-vla/architecture',
        'module-04-vla/tooling',
        'module-04-vla/integration',
        'module-04-vla/summary',
        'module-04-vla/capstone',
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
