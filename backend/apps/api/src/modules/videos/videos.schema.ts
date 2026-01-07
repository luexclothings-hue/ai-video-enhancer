import { z } from 'zod';

export const uploadVideoSchema = z.object({
  filename: z.string().min(1, 'Filename is required'),
  durationSeconds: z.number().positive('Duration must be positive'),
  width: z.number().positive('Width must be positive'),
  height: z.number().positive('Height must be positive'),
});

export type UploadVideoInput = z.infer<typeof uploadVideoSchema>;

export const videoResponseSchema = z.object({
  id: z.string(),
  originalFilename: z.string(),
  durationSeconds: z.number(),
  resolution: z.string().nullable(),
  storagePath: z.string(),
  outputStoragePath: z.string().nullable(),
  createdAt: z.date(),
});
