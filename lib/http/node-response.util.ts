import type { ServerResponse } from 'node:http';
import { AiConfigurationError } from '../errors/ai.errors.js';

/** The parts of a Fastify reply used here. */
interface FastifyReplyLike {
  raw: ServerResponse;
  hijack(): void;
  getHeaders(): Record<string, string | number | string[] | undefined>;
}

function isFastifyReply(value: unknown): value is FastifyReplyLike {
  const reply = value as Partial<FastifyReplyLike> | null;
  return typeof reply?.hijack === 'function' && typeof reply.raw?.writeHead === 'function';
}

function isServerResponse(value: unknown): value is ServerResponse {
  const response = value as Partial<ServerResponse> | null;
  return typeof response?.writeHead === 'function' && typeof response.on === 'function';
}

/**
 * The Node response behind the platform's: Express's and the Node adapter's are Node responses; a Fastify reply is
 * hijacked when the body starts (`open()`), with the headers set on it so far (CORS) copied to its raw response.
 */
export function toNodeResponse(response: unknown): { res: ServerResponse; open(): void } {
  if (isFastifyReply(response)) {
    return {
      res: response.raw,
      open: () => {
        const headers = response.getHeaders();
        response.hijack();
        for (const [name, value] of Object.entries(headers)) {
          if (value !== undefined && !response.raw.hasHeader(name)) {
            response.raw.setHeader(name, value);
          }
        }
      },
    };
  }
  if (isServerResponse(response)) {
    return { res: response, open: () => {} };
  }
  throw new AiConfigurationError(
    'Pass the response that @Res() injects (Express, Fastify or the Node adapter) to pipeUIMessageStreamToResponse() ' +
      'and pipeTextStreamToResponse().',
  );
}

/**
 * Writes `response` to the Node response: status and headers, then the body as it streams, respecting backpressure.
 * A client that goes away cancels the body (which aborts the call that produces it).
 */
export async function writeWebResponse(response: Response, res: ServerResponse): Promise<void> {
  if (res.headersSent || res.destroyed) {
    await response.body?.cancel().catch(() => {});
    return;
  }

  try {
    const headers: Record<string, string | string[]> = {};
    response.headers.forEach((value, name) => {
      if (name !== 'set-cookie') {
        headers[name] = value;
      }
    });
    const cookies = response.headers.getSetCookie();
    if (cookies.length > 0) {
      headers['set-cookie'] = cookies;
    }
    for (const [name, value] of Object.entries(headers)) {
      res.setHeader(name, value);
    }
    res.writeHead(response.status);
    if (!response.body) {
      res.end();
      return;
    }
    // The client should see the stream open now, not with its first event.
    res.flushHeaders();
  } catch (error) {
    void response.body?.cancel().catch(() => {});
    throw error;
  }

  const reader = response.body!.getReader();
  let gone = false;
  let release: (() => void) | undefined;
  const onClose = () => {
    if (!res.writableFinished) {
      gone = true;
      release?.();
      reader.cancel().catch(() => {});
    }
  };
  res.once('close', onClose);
  try {
    for (;;) {
      const next = await reader.read();
      if (next.done || gone) {
        break;
      }
      if (!res.write(next.value)) {
        await new Promise<void>((resolve) => {
          release = resolve;
          res.once('drain', resolve);
        });
        res.off('drain', release!);
        release = undefined;
      }
    }
    if (gone) {
      // The client went away: the socket may be half-open still, and nothing more will be written to it.
      res.destroy();
    } else {
      res.end();
    }
  } catch {
    void reader.cancel().catch(() => {});
    // The body failed (a text stream whose call failed): the response ends early, as the client will see.
    res.destroy();
  } finally {
    res.off('close', onClose);
    if (release) {
      res.off('drain', release);
    }
    reader.releaseLock();
  }
}
