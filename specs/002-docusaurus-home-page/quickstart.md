# Quickstart Guide: Docusaurus Home Page

## Overview
This guide explains how to set up and customize the Docusaurus home page for the Physical AI & Humanoid Robotics book.

## Prerequisites
- Node.js 18 or higher
- npm or yarn package manager
- Git

## Setup Instructions

### 1. Clone and Install
```bash
git clone <repository-url>
cd <repository-name>
npm install
```

### 2. Development Server
```bash
npm start
```
This will start the Docusaurus development server and open the site in your browser at http://localhost:3000

### 3. Home Page Location
The home page is located at:
```
src/pages/index.mdx
```

### 4. Making Changes
1. Edit `src/pages/index.mdx` to modify the home page content
2. The page will automatically reload in the development server
3. To change module information, update the relevant sections in the MDX file

### 5. Adding/Updating Modules
To update the 4-module overview:
1. Edit the modules section in `src/pages/index.mdx`
2. Ensure each module has:
   - Title
   - One-line description
   - Valid link to the module content

### 6. Button Links
- "Read the Book" button: Update the link to point to the correct first module path
- "Ask the Chatbot" button: Update the link to point to the future chatbot implementation

## Building for Production
```bash
npm run build
```
This creates a static site in the `build/` directory that can be deployed to GitHub Pages.

## Deployment to GitHub Pages
The site is configured for GitHub Pages deployment. After building, the site can be served from the `gh-pages` branch or from the `docs/` folder of the `main` branch depending on your GitHub Pages settings.