/**
 * Build Validation Test
 *
 * Ensures that the Docusaurus site builds successfully without errors.
 * This test runs `npm run build` and validates the build output.
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

describe('Build Validation', () => {
  test('npm run build should complete successfully', () => {
    try {
      // Run build command
      const output = execSync('npm run build', {
        encoding: 'utf8',
        stdio: 'pipe',
        cwd: path.join(__dirname, '..')
      });

      // Build should produce a build directory
      const buildDir = path.join(__dirname, '..', 'build');
      expect(fs.existsSync(buildDir)).toBe(true);

      // Build directory should contain index.html or docs folder
      const buildContents = fs.readdirSync(buildDir);
      expect(
        buildContents.includes('docs') ||
        buildContents.includes('404.html')
      ).toBe(true);

    } catch (error) {
      // If build fails, the test should fail with error details
      throw new Error(`Build failed with error: ${error.message}\n${error.stderr || error.stdout}`);
    }
  }, 120000); // 2 minute timeout for build

  test('build output should contain required files', () => {
    const buildDir = path.join(__dirname, '..', 'build');

    // Check for essential files
    expect(fs.existsSync(path.join(buildDir, 'sitemap.xml'))).toBe(true);
    expect(fs.existsSync(path.join(buildDir, '404.html'))).toBe(true);

    // Check for docs directory
    const docsDir = path.join(buildDir, 'docs');
    expect(fs.existsSync(docsDir)).toBe(true);

    // Check for assets
    const assetsDir = path.join(buildDir, 'assets');
    expect(fs.existsSync(assetsDir)).toBe(true);
  });

  test('build should not contain broken references', () => {
    const buildDir = path.join(__dirname, '..', 'build');

    // Read a sample HTML file and check for obvious issues
    const introPath = path.join(buildDir, 'docs', 'intro', 'index.html');

    if (fs.existsSync(introPath)) {
      const content = fs.readFileSync(introPath, 'utf8');

      // Should not contain unresolved placeholders
      expect(content).not.toMatch(/{{.*?}}/);
      expect(content).not.toMatch(/\[PLACEHOLDER\]/i);

      // Should contain expected content
      expect(content).toMatch(/Physical AI/);
    }
  });
});
