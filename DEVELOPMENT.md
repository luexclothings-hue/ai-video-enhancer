# Development Workflow Guide

## 🚀 Getting Started

This project uses trunk-based development with professional tooling for code quality and consistency.

### Prerequisites

- Node.js 20+
- Docker & Docker Compose
- Git

### Setup

```bash
# Install dependencies
npm install

# Start development environment
npm run dev
```

## 🔄 Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feat/your-feature-name
```

### 2. Make Changes

- Write code following TypeScript/JavaScript best practices
- Add tests for new functionality
- Update documentation as needed

### 3. Commit Changes

```bash
# Use conventional commits
npm run commit

# Or manually with proper format
git commit -m "feat: add new feature description"
```

### 4. Push and Create PR

```bash
git push origin feat/your-feature-name
```

Then create a Pull Request using the provided template.

## 🛠️ Available Scripts

- `npm run dev` - Start development environment with Docker
- `npm run lint` - Run ESLint
- `npm run lint:fix` - Fix ESLint issues
- `npm run format` - Format code with Prettier
- `npm run format:check` - Check formatting
- `npm run commit` - Interactive conventional commit
- `npm test` - Run tests
- `npm run build` - Build for production

## 📋 Commit Types

- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation changes
- `style` - Code formatting
- `refactor` - Code refactoring
- `perf` - Performance improvements
- `test` - Test updates
- `chore` - Maintenance tasks
- `ci` - CI/CD changes
- `build` - Build system changes

## 🔍 Code Quality

### Pre-commit Hooks

- **Lint-staged**: Runs ESLint and Prettier on staged files
- **Commitlint**: Validates commit message format

### CI/CD Pipeline

GitHub Actions automatically:

- Runs linting and formatting checks
- Performs TypeScript type checking
- Runs tests
- Builds the application
- Performs security audits

## 🧪 Testing

```bash
# Run all tests
npm test

# Run specific test
npm test -- auth.test.ts
```

## 📝 Code Style

- **ESLint**: Enforces code quality rules
- **Prettier**: Handles code formatting
- **TypeScript**: Strict type checking enabled
- **Conventional Commits**: Standardized commit messages

## 🚀 Deployment

See [GCP Deployment Guide](backend/GCP_DEPLOYMENT_GUIDE.md) for production deployment.

## 🆘 Troubleshooting

### Common Issues

**Lint errors:**

```bash
npm run lint:fix
```

**Format issues:**

```bash
npm run format
```

**Pre-commit hook failures:**

- Fix linting/formatting issues
- Ensure commit message follows conventional format
- Re-commit with fixes

### Getting Help

1. Check existing documentation
2. Review GitHub issues
3. Ask team members
4. Create new issue with proper template
