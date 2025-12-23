/**
 * MDX Syntax Validation Test
 *
 * Validates MDX and markdown syntax in documentation files.
 * Checks for common syntax errors, malformed frontmatter, and invalid MDX.
 */

const fs = require('fs');
const path = require('path');
const glob = require('glob');
const yaml = require('js-yaml');

describe('MDX Syntax Validation', () => {
  let mdxFiles = [];

  beforeAll(() => {
    // Find all markdown/MDX files
    const docsDir = path.join(__dirname, '..', 'docs');
    mdxFiles = glob.sync(`${docsDir}/**/*.{md,mdx}`, { nodir: true });
  });

  test('should find MDX/markdown files', () => {
    expect(mdxFiles.length).toBeGreaterThan(0);
  });

  test('all files should have valid frontmatter', () => {
    const invalidFiles = [];

    mdxFiles.forEach(file => {
      const content = fs.readFileSync(file, 'utf8');

      // Check for frontmatter delimiters
      if (!content.startsWith('---')) {
        invalidFiles.push({
          file: path.relative(path.join(__dirname, '..', 'docs'), file),
          error: 'Missing frontmatter opening delimiter'
        });
        return;
      }

      // Extract frontmatter
      const match = content.match(/^---\n([\s\S]*?)\n---/);
      if (!match) {
        invalidFiles.push({
          file: path.relative(path.join(__dirname, '..', 'docs'), file),
          error: 'Malformed frontmatter (missing closing delimiter)'
        });
        return;
      }

      // Validate YAML syntax
      try {
        const frontmatter = yaml.load(match[1]);

        // Check required fields
        if (!frontmatter.title) {
          invalidFiles.push({
            file: path.relative(path.join(__dirname, '..', 'docs'), file),
            error: 'Missing required field: title'
          });
        }
      } catch (error) {
        invalidFiles.push({
          file: path.relative(path.join(__dirname, '..', 'docs'), file),
          error: `Invalid YAML: ${error.message}`
        });
      }
    });

    if (invalidFiles.length > 0) {
      const errorMsg = invalidFiles
        .map(f => `  - ${f.file}: ${f.error}`)
        .join('\n');
      fail(`Found ${invalidFiles.length} file(s) with invalid frontmatter:\n${errorMsg}`);
    }
  });

  test('markdown should not have common syntax errors', () => {
    const syntaxErrors = [];

    mdxFiles.forEach(file => {
      const content = fs.readFileSync(file, 'utf8');
      const fileName = path.relative(path.join(__dirname, '..', 'docs'), file);

      // Remove frontmatter for content checks
      const contentWithoutFrontmatter = content.replace(/^---\n[\s\S]*?\n---\n/, '');

      // Check for unclosed code blocks
      const codeBlockMatches = contentWithoutFrontmatter.match(/```/g);
      if (codeBlockMatches && codeBlockMatches.length % 2 !== 0) {
        syntaxErrors.push({
          file: fileName,
          error: 'Unclosed code block (odd number of ```)'
        });
      }

      // Check for unmatched JSX tags (basic check)
      const jsxOpenTags = contentWithoutFrontmatter.match(/<[A-Z][a-zA-Z]*[^/>]*>/g) || [];
      const jsxCloseTags = contentWithoutFrontmatter.match(/<\/[A-Z][a-zA-Z]*>/g) || [];
      const jsxSelfClosing = contentWithoutFrontmatter.match(/<[A-Z][a-zA-Z]*[^>]*\/>/g) || [];

      const expectedCloseTags = jsxOpenTags.length - jsxSelfClosing.length;
      if (jsxCloseTags.length !== expectedCloseTags) {
        syntaxErrors.push({
          file: fileName,
          error: `Potential unmatched JSX tags (${jsxOpenTags.length} open, ${jsxCloseTags.length} close, ${jsxSelfClosing.length} self-closing)`
        });
      }

      // Check for broken image syntax
      const brokenImages = contentWithoutFrontmatter.match(/!\[[^\]]*\]\([^)]*\s+[^)]*\)/g);
      if (brokenImages) {
        syntaxErrors.push({
          file: fileName,
          error: 'Image syntax with spaces in URL (should use %20 or quotes)'
        });
      }

      // Check for empty headings
      const emptyHeadings = contentWithoutFrontmatter.match(/^#+\s*$/gm);
      if (emptyHeadings) {
        syntaxErrors.push({
          file: fileName,
          error: `Found ${emptyHeadings.length} empty heading(s)`
        });
      }
    });

    if (syntaxErrors.length > 0) {
      const errorMsg = syntaxErrors
        .map(e => `  - ${e.file}: ${e.error}`)
        .join('\n');
      fail(`Found ${syntaxErrors.length} syntax error(s):\n${errorMsg}`);
    }
  });

  test('code blocks should have language specified', () => {
    const missingLanguage = [];

    mdxFiles.forEach(file => {
      const content = fs.readFileSync(file, 'utf8');
      const fileName = path.relative(path.join(__dirname, '..', 'docs'), file);

      // Find code blocks without language specification
      const matches = content.match(/\n```\n/g);

      if (matches) {
        missingLanguage.push({
          file: fileName,
          count: matches.length
        });
      }
    });

    if (missingLanguage.length > 0) {
      const errorMsg = missingLanguage
        .map(f => `  - ${f.file}: ${f.count} code block(s) without language`)
        .join('\n');
      console.warn(`Warning: Code blocks without language specification:\n${errorMsg}`);
      // Warning only, don't fail the test
    }
  });

  test('files should not contain TODO or FIXME comments', () => {
    const todosFound = [];

    mdxFiles.forEach(file => {
      const content = fs.readFileSync(file, 'utf8');
      const fileName = path.relative(path.join(__dirname, '..', 'docs'), file);

      // Look for TODO or FIXME (case insensitive)
      const todos = content.match(/\b(TODO|FIXME|XXX|HACK)\b/gi);

      if (todos) {
        todosFound.push({
          file: fileName,
          count: todos.length
        });
      }
    });

    if (todosFound.length > 0) {
      const warningMsg = todosFound
        .map(f => `  - ${f.file}: ${f.count} TODO/FIXME comment(s)`)
        .join('\n');
      console.warn(`Warning: Found TODO/FIXME comments:\n${warningMsg}`);
      // Warning only, don't fail the test in development
    }
  });
});
