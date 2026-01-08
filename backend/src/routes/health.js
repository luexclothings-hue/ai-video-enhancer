const db = require('../config/db');

async function routes(fastify, options) {
    fastify.get('/health', async (request, reply) => {
        try {
            const dbRes = await db.query('SELECT NOW()');
            return {
                status: 'ok',
                timestamp: new Date(),
                db_time: dbRes.rows[0].now
            };
        } catch (err) {
            fastify.log.error(err);
            reply.code(500).send({ status: 'error', message: 'Database connection failed' });
        }
    });
}

module.exports = routes;
