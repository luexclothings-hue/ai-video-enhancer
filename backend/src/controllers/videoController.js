const videoService = require('../services/videoService');

class VideoController {
    async upload(req, reply) {
        const data = await req.file();

        if (!data) {
            return reply.code(400).send({ error: 'No file uploaded' });
        }

        try {
            const job = await videoService.uploadVideo(data.file, data.filename, data.mimetype);
            return reply.code(201).send({
                success: true,
                job,
                message: 'Video uploaded and queue for processing'
            });
        } catch (error) {
            req.log.error(error);
            return reply.code(500).send({ error: 'Internal Server Error', details: error.message });
        }
    }
}

module.exports = new VideoController();
