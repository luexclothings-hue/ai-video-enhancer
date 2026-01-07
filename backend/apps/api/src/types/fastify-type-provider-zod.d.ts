declare module 'fastify-type-provider-zod' {
  import { FastifyPluginCallback } from 'fastify';
  import { ZodTypeProvider } from 'fastify-type-provider-zod';

  const plugin: FastifyPluginCallback;
  export default plugin;
  export { ZodTypeProvider };
}
