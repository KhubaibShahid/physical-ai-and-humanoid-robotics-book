---
sidebar_position: 3
title: GitHub Pages Deployment
description: Complete guide to deploying this Docusaurus site to GitHub Pages
---

# GitHub Pages Deployment Guide

This guide covers the complete process of deploying the Physical AI & Humanoid Robotics book to GitHub Pages.

## Prerequisites

- Node.js 20 or higher installed locally
- Git configured with GitHub authentication
- Repository pushed to GitHub

## Quick Start

If you just want to deploy quickly:

1. Enable GitHub Pages in repository settings
2. Push to `main` branch
3. Wait 3-5 minutes for automatic deployment

## Detailed Setup Guide

### Step 1: Enable GitHub Pages

1. **Navigate to Repository Settings**
   ```
   https://github.com/KhubaibShahid/physical-ai-and-humanoid-robotics-book/settings/pages
   ```

2. **Configure Build and Deployment**
   - Under "Build and deployment"
   - **Source**: Select **"GitHub Actions"** from dropdown
   - ⚠️ Do NOT select "Deploy from a branch"

3. **Save Configuration**
   - GitHub will show: "Your site is ready to be published"
   - After first successful deployment: "Your site is live at..."

### Step 2: Verify Workflow Permissions

1. **Go to Actions Settings**
   ```
   https://github.com/KhubaibShahid/physical-ai-and-humanoid-robotics-book/settings/actions
   ```

2. **Check Workflow Permissions**
   - Scroll to "Workflow permissions"
   - Select: **"Read and write permissions"**
   - Check: **"Allow GitHub Actions to create and approve pull requests"**
   - Click **Save**

### Step 3: Trigger Deployment

**Option A: Push to Main Branch**
```bash
git push origin main
```

**Option B: Manual Workflow Dispatch**
1. Go to Actions tab
2. Select "Build and Deploy to GitHub Pages"
3. Click "Run workflow" → "Run workflow"

**Option C: Re-run Failed Workflow**
1. Go to Actions tab
2. Click on the failed workflow run
3. Click "Re-run jobs" → "Re-run all jobs"

### Step 4: Monitor Deployment

1. **Go to Actions Tab**
   ```
   https://github.com/KhubaibShahid/physical-ai-and-humanoid-robotics-book/actions
   ```

2. **Watch Workflow Progress**
   - 🟡 Yellow dot = In progress
   - ✅ Green checkmark = Success
   - ❌ Red X = Failed

3. **Check Deployment Status**
   - Build job: Should complete in ~2-3 minutes
   - Deploy job: Should complete in ~1-2 minutes

### Step 5: Verify Site is Live

**Check URL:**
```
https://khubaibshahid.github.io/physical-ai-and-humanoid-robotics-book/
```

**Verify with curl:**
```bash
curl -I https://khubaibshahid.github.io/physical-ai-and-humanoid-robotics-book/
# Should return: HTTP/2 200
```

**Expected Pages:**
- Homepage: `/`
- Chatbot: `/chatbot`
- Book content: `/docs/intro`

## Common Issues and Solutions

### Issue 1: 404 "Site not found"

**Symptoms:**
- GitHub Pages shows "Site not found" page
- URL returns 404 error

**Causes:**
- GitHub Pages not enabled
- Source set to "Deploy from a branch" instead of "GitHub Actions"
- Deployment hasn't completed yet

**Solutions:**
1. Enable GitHub Pages in Settings → Pages
2. Change source to "GitHub Actions"
3. Wait for deployment to complete (3-5 minutes)
4. Clear browser cache and retry

### Issue 2: Node.js Version Error

**Error Message:**
```
Minimum Node.js version not met :(
You are using Node.js v18.x, Requirement: Node.js >=20.0
```

**Solution:**
Update GitHub Actions workflows to use Node.js 20:

```yaml
# .github/workflows/build-deploy.yml
# .github/workflows/content-validation.yml
- name: Setup Node.js
  uses: actions/setup-node@v4
  with:
    node-version: '20'  # ← Must be 20 or higher
    cache: 'npm'
```

### Issue 3: Deployment Permission Denied

**Error Message:**
```
Error: Creating Pages deployment failed
Error: HttpError: Not Found
Ensure GitHub Pages has been enabled
```

**Solutions:**
1. **Enable GitHub Pages** (Settings → Pages → Source: GitHub Actions)
2. **Check Workflow Permissions** (Settings → Actions → Read and write permissions)
3. **Verify Repository Visibility** (Private repos may need GitHub Pro for Pages)

### Issue 4: Configuration URL Mismatch

**Symptoms:**
- Site loads but assets are missing (404 for CSS/JS)
- Navigation doesn't work
- Blank page or broken styling

**Cause:**
Incorrect `url` or `organizationName` in `docusaurus.config.js`

**Solution:**
Update `docusaurus.config.js`:
```javascript
const config = {
  url: 'https://khubaibshahid.github.io',              // ← Your GitHub username
  baseUrl: '/physical-ai-and-humanoid-robotics-book/', // ← Repository name
  organizationName: 'KhubaibShahid',                   // ← Your GitHub username
  projectName: 'physical-ai-and-humanoid-robotics-book', // ← Repository name
};
```

### Issue 5: Build Succeeds but Deploy Fails

**Symptoms:**
- Build job: ✅ Success
- Deploy job: ❌ Failed

**Common Causes:**
1. GitHub Pages not enabled
2. Wrong workflow permissions
3. Repository visibility issues (private repo without GitHub Pro)

**Debug Steps:**
1. Check deploy job logs in Actions tab
2. Verify Pages is enabled (Settings → Pages)
3. Check workflow permissions (Settings → Actions)
4. Ensure repository is public OR you have GitHub Pro (for private repos)

### Issue 6: Broken Links After Deployment

**Symptoms:**
- Homepage works but internal links return 404
- Module navigation broken

**Causes:**
- Incorrect `baseUrl` in config
- Links don't include base path
- Trailing slash inconsistency

**Solutions:**
1. **Use Docusaurus Link component:**
   ```jsx
   import Link from '@docusaurus/Link';
   <Link to="/docs/intro">Read the Book</Link>
   ```

2. **Check baseUrl in config:**
   ```javascript
   baseUrl: '/physical-ai-and-humanoid-robotics-book/',
   trailingSlash: false,
   ```

3. **Run build locally to test:**
   ```bash
   npm run build
   npm run serve
   # Test at: http://localhost:3000/physical-ai-and-humanoid-robotics-book/
   ```

## Deployment Workflow Overview

### Automatic Deployment Trigger

The site automatically deploys when:
- Commits are pushed to `main` or `master` branch
- Workflow is manually triggered via Actions tab

### Workflow Steps

1. **Checkout Code**
   - Repository code is checked out to runner

2. **Setup Node.js**
   - Node.js 20 is installed
   - npm cache is configured

3. **Install Dependencies**
   - `npm ci` installs exact versions from `package-lock.json`

4. **Build Site**
   - `npm run build` generates static files in `build/` directory
   - Docusaurus compiles MDX to HTML
   - Assets are bundled and optimized

5. **Upload Artifact**
   - `build/` directory is packaged as GitHub Pages artifact

6. **Deploy to Pages**
   - Artifact is deployed to GitHub Pages service
   - Site becomes available at configured URL

### Build Output

The build process creates:
```
build/
├── index.html                    # Homepage
├── chatbot/
│   └── index.html               # Chatbot page
├── docs/
│   └── intro/
│       └── index.html           # Module 1 intro
├── assets/
│   ├── css/                     # Compiled CSS
│   └── js/                      # Bundled JavaScript
└── img/                         # Static images
```

## Configuration Files

### GitHub Actions Workflows

**Build and Deploy** (`.github/workflows/build-deploy.yml`):
- Triggers: Push to main/master, manual dispatch
- Jobs: Build → Deploy
- Node.js version: 20
- Deploy target: GitHub Pages

**Content Validation** (`.github/workflows/content-validation.yml`):
- Triggers: Pull requests, push to main/master
- Jobs: Build validation
- Checks: Build succeeds, no broken links (future)

### Docusaurus Configuration

**Key Settings** (`docusaurus.config.js`):
```javascript
url: 'https://khubaibshahid.github.io',
baseUrl: '/physical-ai-and-humanoid-robotics-book/',
organizationName: 'KhubaibShahid',
projectName: 'physical-ai-and-humanoid-robotics-book',
deploymentBranch: 'gh-pages',  // Auto-managed by GitHub Actions
trailingSlash: false,
onBrokenLinks: 'throw',        // Fail build on broken links
```

## Testing Before Deployment

### Local Build Test

```bash
# Build the site
npm run build

# Serve built site locally
npm run serve

# Visit http://localhost:3000/physical-ai-and-humanoid-robotics-book/
```

### Check for Issues

1. **Verify all pages load**
   - Homepage: `/`
   - Chatbot: `/chatbot`
   - Docs: `/docs/intro`

2. **Test navigation**
   - Click all buttons and links
   - Verify module navigation works

3. **Check styling**
   - Buttons have correct styling
   - No duplicate navbar/footer
   - Responsive design works

4. **Browser console**
   - Check for 404 errors
   - Verify no JavaScript errors

## Maintenance

### Updating Content

1. **Edit MDX files** in `docs/` or `src/pages/`
2. **Commit changes** to git
3. **Push to main** branch
4. **Automatic deployment** triggers
5. **Site updates** in 3-5 minutes

### Updating Dependencies

```bash
# Check for updates
npm outdated

# Update packages
npm update

# Test build
npm run build

# If successful, commit and push
git add package.json package-lock.json
git commit -m "update dependencies"
git push origin main
```

### Monitoring

**Check Deployment Status:**
```
https://github.com/KhubaibShahid/physical-ai-and-humanoid-robotics-book/actions
```

**View Site Analytics** (if enabled):
```
https://github.com/KhubaibShahid/physical-ai-and-humanoid-robotics-book/settings/pages
```

## Troubleshooting Commands

### Check Site Status
```bash
# Check if site is accessible
curl -I https://khubaibshahid.github.io/physical-ai-and-humanoid-robotics-book/

# Expected: HTTP/2 200
# If 404: Site not deployed yet or Pages not enabled
```

### Local Debug
```bash
# Clear build cache
rm -rf build/ .docusaurus/

# Fresh build
npm run build

# Serve locally
npm run serve
```

### GitHub CLI Debug
```bash
# Check if Pages is enabled
gh api repos/KhubaibShahid/physical-ai-and-humanoid-robotics-book/pages

# View latest workflow run
gh run list --limit 5

# View workflow logs
gh run view --log
```

## Support

If you encounter issues not covered in this guide:

1. **Check GitHub Actions logs** for detailed error messages
2. **Review Docusaurus documentation**: https://docusaurus.io/docs/deployment#deploying-to-github-pages
3. **GitHub Pages docs**: https://docs.github.com/en/pages
4. **Open an issue** in the repository

## Summary Checklist

Before deploying, ensure:

- ✅ Node.js 20+ installed
- ✅ GitHub Pages enabled (Source: GitHub Actions)
- ✅ Workflow permissions set to "Read and write"
- ✅ `docusaurus.config.js` has correct URL and organization
- ✅ Local build succeeds (`npm run build`)
- ✅ All links work in local serve (`npm run serve`)
- ✅ Changes committed and pushed to main branch

After deployment, verify:

- ✅ GitHub Actions workflow succeeds (green checkmarks)
- ✅ Site accessible at https://khubaibshahid.github.io/physical-ai-and-humanoid-robotics-book/
- ✅ All pages load correctly (homepage, chatbot, docs)
- ✅ Navigation works between pages
- ✅ Styling is correct (no duplicates, proper button styling)
