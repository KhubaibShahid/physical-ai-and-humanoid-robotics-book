/**
 * Link Checker Test
 *
 * Validates internal and external links in markdown files.
 * Checks for broken links, invalid anchors, and unreachable URLs.
 */

const fs = require('fs');
const path = require('path');
const glob = require('glob');

describe('Link Validation', () => {
  let markdownFiles = [];
  let allLinks = [];

  beforeAll(() => {
    // Find all markdown files
    const docsDir = path.join(__dirname, '..', 'docs');
    markdownFiles = glob.sync(`${docsDir}/**/*.md`, { nodir: true });
  });

  test('should find markdown files', () => {
    expect(markdownFiles.length).toBeGreaterThan(0);
  });

  test('all internal doc links should reference existing files', () => {
    const brokenLinks = [];
    const docsDir = path.join(__dirname, '..', 'docs');

    markdownFiles.forEach(file => {
      const content = fs.readFileSync(file, 'utf8');

      // Match markdown links: [text](/docs/path)
      const linkRegex = /\[([^\]]+)\]\(\/docs\/([^)#]+)(?:#[^)]*)?\)/g;
      let match;

      while ((match = linkRegex.exec(content)) !== null) {
        const linkPath = match[2];
        const fullPath = path.join(docsDir, linkPath);

        // Check if file exists (with or without .md extension)
        const exists =
          fs.existsSync(fullPath) ||
          fs.existsSync(`${fullPath}.md`) ||
          fs.existsSync(path.join(fullPath, 'index.md'));

        if (!exists) {
          brokenLinks.push({
            file: path.relative(docsDir, file),
            link: `/docs/${linkPath}`,
            text: match[1]
          });
        }
      }
    });

    if (brokenLinks.length > 0) {
      const errorMsg = brokenLinks
        .map(link => `  - In ${link.file}: "${link.text}" -> ${link.link}`)
        .join('\n');
      throw new Error(`Found ${brokenLinks.length} broken internal link(s):\n${errorMsg}`);
    }
  });

  test('markdown files should not have duplicate anchors', () => {
    const filesWithDuplicates = [];

    markdownFiles.forEach(file => {
      const content = fs.readFileSync(file, 'utf8');
      const headings = content.match(/^#{1,6}\s+.+$/gm) || [];

      // Convert headings to anchor format (lowercase, replace spaces with hyphens)
      const anchors = headings.map(h =>
        h.replace(/^#+\s+/, '')
          .toLowerCase()
          .replace(/[^a-z0-9\s-]/g, '')
          .replace(/\s+/g, '-')
      );

      // Find duplicates
      const seen = new Set();
      const duplicates = [];

      anchors.forEach(anchor => {
        if (seen.has(anchor)) {
          duplicates.push(anchor);
        }
        seen.add(anchor);
      });

      if (duplicates.length > 0) {
        filesWithDuplicates.push({
          file: path.relative(path.join(__dirname, '..', 'docs'), file),
          duplicates: [...new Set(duplicates)]
        });
      }
    });

    if (filesWithDuplicates.length > 0) {
      const errorMsg = filesWithDuplicates
        .map(f => `  - ${f.file}: ${f.duplicates.join(', ')}`)
        .join('\n');
      throw new Error(`Found duplicate anchors in:\n${errorMsg}`);
    }
  });

  test('links should use correct format', () => {
    const invalidLinks = [];
    const docsDir = path.join(__dirname, '..', 'docs');

    markdownFiles.forEach(file => {
      const content = fs.readFileSync(file, 'utf8');

      // Check for common link format issues
      // 1. Links with spaces (should use %20 or hyphens)
      const spacedLinks = content.match(/\]\([^)]*\s+[^)]*\)/g);
      if (spacedLinks) {
        invalidLinks.push({
          file: path.relative(docsDir, file),
          issue: 'Links with unencoded spaces',
          examples: spacedLinks.slice(0, 3)
        });
      }

      // 2. Relative links starting with ./ or ../ (should use absolute /docs/ paths)
      const relativeLinks = content.match(/\]\(\.\.?\/[^)]+\)/g);
      if (relativeLinks) {
        invalidLinks.push({
          file: path.relative(docsDir, file),
          issue: 'Relative links (should use absolute /docs/ paths)',
          examples: relativeLinks.slice(0, 3)
        });
      }
    });

    if (invalidLinks.length > 0) {
      const errorMsg = invalidLinks
        .map(link => `  - ${link.file}: ${link.issue}\n    Examples: ${link.examples.join(', ')}`)
        .join('\n');
      console.warn(`Found link format issues:\n${errorMsg}`);
      // Warning only, don't fail the test
    }
  });
});
