# AI Video Enhancer

Professional AI video enhancement service using Stream-DiffVSR with 720p optimization and subscription-based billing.

## 🚀 Quick Start

```bash
# Install dependencies
npm install

# Start development environment
npm run dev

# Run linting and formatting
npm run lint
npm run format

# Run tests
npm test
```

## 🏗️ Architecture

- **API Server**: Node.js/Fastify with TypeScript
- **GPU Worker**: Python Stream-DiffVSR processor
- **Database**: PostgreSQL with Prisma ORM
- **Queue**: Google Cloud Pub/Sub
- **Storage**: Google Cloud Storage

## 📋 Development Workflow

### Trunk-Based Development

- **Main branch**: `main` (production-ready)
- **Feature branches**: Short-lived, merge via PR
- **Commit format**: Conventional commits with Commitizen

### Making Changes

```bash
# Create feature branch
git checkout -b feat/your-feature-name

# Make changes and commit using conventional format
npm run commit

# Push and create PR
git push origin feat/your-feature-name
```

### Commit Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test updates
- `chore`: Maintenance tasks

## 🔧 Development Tools

- **ESLint**: Code linting with TypeScript support
- **Prettier**: Code formatting
- **Husky**: Git hooks for quality checks
- **Commitlint**: Conventional commit validation
- **Lint-staged**: Pre-commit linting
- **GitHub Actions**: CI/CD pipeline

## 📊 API Endpoints

- `POST /auth/register` - User registration
- `POST /auth/login` - Authentication
- `POST /videos/upload` - Video upload (width/height required)
- `GET /videos` - List user videos
- `GET /jobs/:id` - Check processing status

## 💳 Subscription Plans

| Plan    | Videos    | Duration | Resolution | Price  |
| ------- | --------- | -------- | ---------- | ------ |
| FREE    | 1         | 30s      | 720p       | Free   |
| CREATOR | Unlimited | 2min     | 720p       | $9.99  |
| PRO     | Unlimited | 5min     | 720p       | $29.99 |

## 🚀 Deployment

See [GCP Deployment Guide](backend/GCP_DEPLOYMENT_GUIDE.md) for production deployment instructions.

## 🧪 Testing

```bash
# Run all tests
npm test

# Run with coverage
npm run test:coverage

# Run specific test file
npm test -- auth.test.ts
```

## 📝 Scripts

- `npm run dev` - Start development environment
- `npm run build` - Build for production
- `npm run lint` - Run ESLint
- `npm run format` - Format code with Prettier
- `npm run commit` - Interactive commit with Commitizen
- `npm run type-check` - TypeScript type checking

## 🔒 Environment Variables

See `backend/apps/api/.env.example` for required environment variables.

## 📄 License

Proprietary - All rights reserved
