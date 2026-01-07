import { buildApp } from './app';
import { config } from './config';
import logger from './utils/logger';

async function start() {
  try {
    const app = await buildApp();

    await app.listen({
      port: config.PORT,
      host: config.HOST,
    });

    logger.info(`🚀 Server listening on http://${config.HOST}:${config.PORT}`);
    logger.info(`📚 API Documentation: http://${config.HOST}:${config.PORT}/documentation`);
  } catch (error) {
    logger.error({ error }, 'Failed to start server');
    process.exit(1);
  }
}

// Graceful shutdown
const gracefulShutdown = async (signal: string) => {
  logger.info(`Received ${signal}, shutting down gracefully...`);
  process.exit(0);
};

process.on('SIGINT', () => gracefulShutdown('SIGINT'));
process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));

start();
