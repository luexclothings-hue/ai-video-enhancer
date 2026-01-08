const fastify = require('fastify')({ logger: true });
require('dotenv').config();

const healthRoutes = require('./routes/health');

// Register plugins
fastify.register(require('@fastify/multipart'), {
    limits: {
        fileSize: 100 * 1024 * 1024 // 100MB limit
    }
});

// Register routes
fastify.register(healthRoutes);
fastify.register(require('./routes/videoRoutes'), { prefix: '/api/v1/videos' });

const start = async () => {
    try {
        const port = process.env.PORT || 3000;
        await fastify.listen({ port, host: '0.0.0.0' });
        console.log(`Server running at http://localhost:${port}`);
    } catch (err) {
        fastify.log.error(err);
        process.exit(1);
    }
};

start();
