# Research: Docusaurus Home Page with Book and Chatbot Buttons

## Decision: Home Page Structure
**Rationale**: The home page will be implemented as `src/pages/index.mdx` which is the standard Docusaurus approach for creating a homepage. This follows Docusaurus conventions and will be automatically routed to the root URL.

## Decision: Page Content Organization
**Rationale**: The page will follow the required sections from the specification:
- Hero title section with "Physical AI & Humanoid Robotics"
- Short description explaining the book
- Paragraph about Physical AI and embodied intelligence
- Four-module overview with one-line descriptions each
- Two primary CTA buttons: "Read the Book" (→ Module 1) and "Ask the Chatbot" (→ placeholder)

## Decision: Button Implementation
**Rationale**: Using Docusaurus's built-in support for buttons through MDX/JSX components. The "Read the Book" button will link to the first module using Docusaurus's standard link syntax (likely `/docs/module-1/` or similar). The "Ask the Chatbot" button will link to a placeholder page or potentially an external URL for the future RAG chatbot.

## Decision: Responsive Design
**Rationale**: Docusaurus provides built-in responsive design capabilities. The layout will use standard Docusaurus components and CSS classes to ensure it works across mobile, tablet, and desktop devices as required by the specification.

## Decision: Styling Approach
**Rationale**: Using Docusaurus's built-in styling system with minimal custom CSS. This ensures compatibility with the theme and follows the "minimal, clean, non-marketing tone" requirement from the specification.

## Alternatives Considered:

### Alternative 1: Different Homepage Location
- **Alternative**: Using `docs/intro.md` as the homepage
- **Rejected**: `src/pages/index.mdx` is the standard Docusaurus approach for homepages and provides more flexibility for custom components

### Alternative 2: Custom React Component vs MDX
- **Alternative**: Creating a full React component instead of MDX
- **Rejected**: MDX is specifically required by the constitution and specification, and provides the right balance of content and functionality

### Alternative 3: Different Button Framework
- **Alternative**: Using a different button component library
- **Rejected**: Docusaurus's built-in button components are sufficient and maintain consistency with the theme

## Technical Implementation Details:

### Docusaurus Homepage Creation:
- Create `src/pages/index.mdx` file
- Use Docusaurus's built-in components like `<Hero>`, `<Button>`, and layout elements
- Follow MDX syntax for embedding React components within Markdown

### Link Structure:
- "Read the Book" button will use Docusaurus `<Link>` component to navigate to Module 1
- "Ask the Chatbot" button will link to a placeholder page (e.g., `/chatbot/`) or external URL

### Module Overview Display:
- Use a simple list or grid layout to display the 4 modules
- Each module gets a one-line description as specified