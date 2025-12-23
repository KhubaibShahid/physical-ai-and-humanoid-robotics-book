# Data Model: Docusaurus Home Page

## Entities

### Home Page Content
- **Name**: Home Page Content
- **Description**: Represents the introductory content for the book, including hero section, book description, and module overview
- **Fields**:
  - heroTitle: string (required) - The main title displayed on the homepage
  - heroDescription: string (required) - Brief description explaining the book
  - bookDescription: string (required) - Paragraph explaining what the book covers regarding Physical AI and embodied intelligence
  - modules: Module[] (required) - Array of 4 modules with descriptions
- **Validation Rules**:
  - heroTitle must be 1-100 characters
  - heroDescription must be 10-500 characters
  - bookDescription must be 50-1000 characters
  - modules array must contain exactly 4 elements

### Module
- **Name**: Module
- **Description**: Represents a single module in the book's 4-module structure
- **Fields**:
  - id: string (required) - Unique identifier for the module
  - title: string (required) - Title of the module
  - description: string (required) - One-line description of the module
  - link: string (required) - URL path to the module content
- **Validation Rules**:
  - id must be unique within the modules array
  - title must be 1-100 characters
  - description must be 1-200 characters
  - link must be a valid relative URL

### Navigation Buttons
- **Name**: Navigation Buttons
- **Description**: Represents the two primary CTA buttons for accessing the book content and chatbot feature
- **Fields**:
  - readBookButton: Button (required) - Button that links to Module 1
  - askChatbotButton: Button (required) - Button that links to the future RAG chatbot placeholder
- **Validation Rules**:
  - Both buttons must have valid link targets
  - Button text must be non-empty

### Button
- **Name**: Button
- **Description**: Represents a single button with text and link
- **Fields**:
  - text: string (required) - The display text for the button
  - link: string (required) - The URL the button links to
  - variant: string (optional) - Style variant (e.g., primary, secondary)
- **Validation Rules**:
  - text must be 1-50 characters
  - link must be a valid URL (relative or absolute)