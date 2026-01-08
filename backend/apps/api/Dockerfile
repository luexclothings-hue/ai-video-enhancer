# syntax=docker/dockerfile:1

# ===========================
# Builder Stage
# ===========================
FROM node:20-alpine AS builder

# Enable Corepack and prepare pnpm
RUN corepack enable && corepack prepare pnpm@9.15.4 --activate

WORKDIR /app

# Install OpenSSL for Prisma
RUN apk add --no-cache openssl

# Copy package files first for better caching
COPY package.json pnpm-lock.yaml ./

# Install ALL dependencies (including devDependencies for build) using pnpm
RUN pnpm install --frozen-lockfile

# Copy prisma schema and generate client
COPY prisma ./prisma/
RUN pnpm exec prisma generate

# Copy source code and build
COPY . .
RUN pnpm run build

# ===========================
# Production Stage
# ===========================
FROM node:20-alpine

WORKDIR /app

# Install runtime dependencies
RUN apk add --no-cache openssl dumb-init

# Enable Corepack and prepare pnpm
RUN corepack enable && corepack prepare pnpm@9.15.4 --activate

# Create non-root user for security
RUN addgroup -g 1001 -S nodejs && adduser -S nodejs -u 1001

# Copy package files
COPY package.json pnpm-lock.yaml ./

# Install ONLY production dependencies using pnpm
RUN pnpm install --prod --frozen-lockfile

# Copy prisma schema and generate client
COPY prisma ./prisma/
RUN pnpm exec prisma generate

# Copy built application from builder stage
COPY --from=builder --chown=nodejs:nodejs /app/dist ./dist

# Set ownership to non-root user
RUN chown -R nodejs:nodejs /app

# Switch to non-root user
USER nodejs

# Expose application port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD node -e "require('http').get('http://localhost:3000/health', (r) => {process.exit(r.statusCode === 200 ? 0 : 1)})"

# Use dumb-init to handle signals properly
ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "dist/server.js"]