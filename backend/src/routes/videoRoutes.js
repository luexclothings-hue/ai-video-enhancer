const videoController = require('../controllers/videoController');

async function routes(fastify, options) {
    fastify.post('/upload', videoController.upload);
}

module.exports = routes;
