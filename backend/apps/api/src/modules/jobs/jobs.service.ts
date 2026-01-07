import { PrismaClient, JobStatus } from '@prisma/client';
import { NotFoundError } from '../../utils/errors';

export class JobService {
  constructor(private prisma: PrismaClient) {}

  /**
   * Create a new job
   */
  async createJob(userId: string, videoId: string) {
    const job = await this.prisma.job.create({
      data: {
        userId,
        videoId,
        status: 'UPLOADED',
        progressPercent: 0,
      },
    });

    return job;
  }

  /**
   * Update job status
   */
  async updateJobStatus(
    jobId: string,
    status: JobStatus,
    progressPercent?: number,
    errorMessage?: string
  ) {
    const updateData: {
      status: JobStatus;
      progressPercent?: number;
      errorMessage?: string;
      startedAt?: Date;
      completedAt?: Date;
    } = {
      status,
    };

    if (progressPercent !== undefined) {
      updateData.progressPercent = progressPercent;
    }

    if (errorMessage) {
      updateData.errorMessage = errorMessage;
    }

    if (status === 'PROCESSING' && !updateData.startedAt) {
      updateData.startedAt = new Date();
    }

    if (status === 'COMPLETED' || status === 'FAILED') {
      updateData.completedAt = new Date();
    }

    const job = await this.prisma.job.update({
      where: { id: jobId },
      data: updateData,
    });

    return job;
  }

  /**
   * Get user's jobs
   */
  async getUserJobs(userId: string, status?: JobStatus, limit: number = 50, offset: number = 0) {
    const where = status ? { userId, status } : { userId };

    const [jobs, total] = await Promise.all([
      this.prisma.job.findMany({
        where,
        include: {
          video: true,
        },
        orderBy: { createdAt: 'desc' },
        take: limit,
        skip: offset,
      }),
      this.prisma.job.count({ where }),
    ]);

    return { jobs, total };
  }

  /**
   * Get job by ID
   */
  async getJobById(jobId: string, userId: string) {
    const job = await this.prisma.job.findFirst({
      where: {
        id: jobId,
        userId,
      },
      include: {
        video: true,
      },
    });

    if (!job) {
      throw new NotFoundError('Job not found');
    }

    return job;
  }

  /**
   * Mark minutes as charged for a job (called by worker)
   */
  async chargeMinutes(jobId: string, minutesCharged: number) {
    const job = await this.prisma.job.findUnique({
      where: { id: jobId },
    });

    if (!job) {
      throw new NotFoundError('Job not found');
    }

    // Idempotent: only charge if not already charged
    if (job.minutesCharged !== null) {
      return job;
    }

    // Update job and user in a transaction
    const [updatedJob] = await this.prisma.$transaction([
      this.prisma.job.update({
        where: { id: jobId },
        data: { minutesCharged },
      }),
      this.prisma.user.update({
        where: { id: job.userId },
        data: {
          minutesUsedThisMonth: {
            increment: minutesCharged,
          },
        },
      }),
    ]);

    return updatedJob;
  }
}
