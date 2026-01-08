# Simple API server to expose Stream-DiffVSR functionality
# This will be the main entry point for the worker service

from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "video-enhancer-worker"})

@app.route('/enhance', methods=['POST'])
def enhance_video():
    # TODO: Integrate Stream-DiffVSR here
    return jsonify({"message": "Video enhancement endpoint - coming soon"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)