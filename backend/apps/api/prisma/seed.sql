-- Sample seed data for development/testing

-- Create a test user (password: "TestPassword123")
INSERT INTO users (id, email, password_hash, plan, minutes_used_this_month, billing_cycle_start, created_at, updated_at)
VALUES 
  ('550e8400-e29b-41d4-a716-446655440000', 'test@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8.SLUW.xkTZ4uy8e0SG', 'FREE', 0, NOW(), NOW(), NOW()),
  ('660e8400-e29b-41d4-a716-446655440000', 'creator@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8.SLUW.xkTZ4uy8e0SG', 'CREATOR', 0, NOW(), NOW(), NOW()),
  ('770e8400-e29b-41d4-a716-446655440000', 'pro@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8.SLUW.xkTZ4uy8e0SG', 'PRO', 0, NOW(), NOW(), NOW());

-- Note: All users have password "TestPassword123"
-- In production, users should create their own secure passwords

-- The password hash above is bcrypt hash of "TestPassword123" with 12 rounds
