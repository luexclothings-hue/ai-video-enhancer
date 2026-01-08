CREATE TABLE IF NOT EXISTS videos (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    gcs_raw_path TEXT NOT NULL,
    gcs_enhanced_path TEXT,
    status TEXT DEFAULT 'PENDING',
    progress INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);
