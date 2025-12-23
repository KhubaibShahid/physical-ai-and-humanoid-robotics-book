# Logo and Brand Assets

This directory contains the logo and branding assets for the Physical AI & Humanoid Robotics book.

## Files

### Primary Logo
- **robot-logo.svg** (2.5 KB)
  - Full-color green robot logo
  - Used in navbar and main branding
  - Scalable vector format
  - Colors: #22c55e (main green), #4ade80 (accent), #16a34a (dark green)

### Favicon
- **favicon.svg** (1.1 KB)
  - Simplified robot icon optimized for small sizes
  - Used as browser favicon
  - Circular design with white robot on green background

### Social Media Card
- **social-card.svg** (4+ KB)
  - 1200x630px Open Graph image
  - Used when sharing on social media (Twitter, LinkedIn, Facebook)
  - Includes title, subtitle, and module highlights
  - Dark background with green accents

## Design Specifications

### Color Palette
- Primary Green: `#22c55e` (rgb(34, 197, 94))
- Light Green: `#4ade80` (rgb(74, 222, 128))
- Dark Green: `#16a34a` (rgb(22, 163, 74))
- Accent Light: `#86efac` (rgb(134, 239, 172))
- Background Dark: `#0f172a` (rgb(15, 23, 42))
- Text Gray: `#94a3b8` (rgb(148, 163, 184))

### Logo Usage Guidelines
- The robot logo represents AI, automation, and humanoid robotics
- Green color symbolizes growth, innovation, and technology
- Maintain aspect ratio when scaling
- Use on light or dark backgrounds (high contrast)
- Minimum size: 32x32px for favicon, 48x48px for logo

## SVG Benefits
- Infinitely scalable without quality loss
- Small file sizes
- Crisp rendering on all screens (including retina displays)
- Easy to modify colors and elements
- Supported by all modern browsers

## File Locations
```
static/img/
├── robot-logo.svg      → Navbar logo
├── favicon.svg         → Browser tab icon
└── social-card.svg     → Social media sharing image
```

## Docusaurus Configuration
These assets are referenced in `docusaurus.config.js`:
```javascript
favicon: 'img/favicon.svg',           // Browser icon
logo: { src: 'img/robot-logo.svg' }, // Navbar logo
image: 'img/social-card.svg',         // Social sharing
```

---

**Note**: All logos are original designs created specifically for this project. The green robot symbolizes the intersection of AI intelligence and physical robotics.
