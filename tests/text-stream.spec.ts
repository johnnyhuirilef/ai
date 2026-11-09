/**
 * `streamText()` and HTTP response cleanup: the text and its promises, several readers, the UI message stream as a `Response`,
 * timeouts that stop at the first chunk, fallback, usage events, aborts and failures nobody reads.
 */
import { EventEmitter, once } from 'node:events';
import { createServer, get, type ServerResponse } from 'node:http';
import { writeWebResponse } from '../lib/http/node-response.util.js';
import { Injectable } from '@nestjs/common';
import type { TestingModule } from '@nestjs/testing';
import * as z from 'zod';
import { AgentFunction, AiConfigurationError, AiModelError, type AgentFunctionContext, type AiUsageEvent } from '../lib/index.js';
import { FakeModel, providerError } from '../lib/testing/index.js';
import { readUIMessageStream, type UIMessage } from 'ai';
import { createAi } from './support/module.js';
import { parseSse, typesOf } from './support/sse.js';

@Injectable()
class StockTools {
  @AgentFunction({ description: 'Stock of a product', inputSchema: z.object({ sku: z.string() }) })
  stock({ sku }: { sku: string }) {
    return { sku, inStock: 12 };
  }

  @AgentFunction({ description: 'Counts the warehouse, reporting its progress' })
  async count(_input: unknown, ctx: AgentFunctionContext) {
    await ctx.reportProgress(1, 2, 'Aisle 1');
    await ctx.reportProgress(2, 2, 'Aisle 2');
    return { counted: 2 };
  }
}

async function collect(iterable: AsyncIterable<string>): Promise<string[]> {
  const parts: string[] = [];
  for await (const part of iterable) {
    parts.push(part);
  }
  return parts;
}


function response() {
  return Object.assign(new EventEmitter(), {
    headersSent: false,
    destroyed: false,
    writableFinished: false,
    setHeader: vi.fn(),
    writeHead: vi.fn(),
    flushHeaders: vi.fn(),
    write: vi.fn(() => true),
    end: vi.fn(),
    destroy: vi.fn(),
  });
}

describe('streamText()', () => {
  let moduleRef: TestingModule | undefined;

  afterEach(async () => {
    await moduleRef?.close();
    moduleRef = undefined;
  });

  it('streams the text of every step, and settles its promises when it ends', async () => {
    const model = new FakeModel([
      { text: 'Let me check. ', toolCalls: [{ name: 'stock', input: { sku: 'SKU-1' } }] },
      (call) => `There are ${(call.toolResults[0]!.output as { inStock: number }).inStock} left.`,
    ]);
    const { moduleRef: ref, ai } = await createAi({ models: { default: model } }, { providers: [StockTools] });
    moduleRef = ref;

    const answer = ai.streamText({ prompt: 'Is SKU-1 in stock?', tools: ['stock'] });
    expect((await collect(answer.textStream)).join('')).toBe('Let me check. There are 12 left.');
    expect(await answer.text).toBe('There are 12 left.');
    expect(await answer.finishReason).toBe('stop');
    expect((await answer.usage).totalTokens).toBe(40);
    expect((await answer.steps).map((step) => step.toolResults.length)).toEqual([1, 0]);
    expect((await answer.responseMessages).map((message) => message.role)).toEqual(['assistant', 'tool', 'assistant']);
    expect(model.calls.every((call) => call.stream)).toBe(true);
  });

  it('gives every reader the whole stream, from its start', async () => {
    const { moduleRef: ref, ai } = await createAi({ models: { default: new FakeModel(['one two three']) } });
    moduleRef = ref;

    const answer = ai.streamText({ prompt: 'Count' });
    const first = collect(answer.textStream);
    await answer.text;
    expect(await first).toEqual(['one ', 'two ', 'three']);
    // A reader that comes after the end reads it all the same.
    expect(await collect(answer.textStream)).toEqual(['one ', 'two ', 'three']);

    const response = answer.toUIMessageStreamResponse({ headers: { 'x-extra': '1' } });
    expect(response.headers.get('content-type')).toBe('text/event-stream');
    expect(response.headers.get('x-extra')).toBe('1');
    expect(typesOf(parseSse(await response.text()))).toEqual([
      'start',
      'start-step',
      'text-start',
      'text-delta',
      'text-delta',
      'text-delta',
      'text-end',
      'finish-step',
      'finish',
      '[DONE]',
    ]);
  });

  it('times out an attempt until its first chunk only, and retries it', async () => {
    const model = new FakeModel([
      { text: 'late', delay: 1_000 },
      { text: 'slow but steady progress here', chunkDelay: 30 },
    ]);
    const { moduleRef: ref, ai } = await createAi({ models: { default: model }, defaults: { retry: { attempts: 2, backoff: { delay: 0 } } } });
    moduleRef = ref;

    // 200 ms to the first chunk (the second answer's comes after about 60 ms); the whole answer takes over 300 ms.
    const answer = ai.streamText({ prompt: 'Hi', timeout: 200 });
    expect(await answer.text).toBe('slow but steady progress here');
    expect((await answer.steps)[0]!.attempts).toBe(2);
  });

  it('falls back to another model before the first chunk, and reports each model call', async () => {
    const events: AiUsageEvent[] = [];
    const { moduleRef: ref, ai } = await createAi({
      models: { default: { model: new FakeModel(() => providerError(503)), fallback: 'backup' }, backup: new FakeModel(['from the backup']) },
      defaults: { retry: false },
      onUsage: (event) => void events.push(event),
    });
    moduleRef = ref;

    const answer = ai.streamText({ prompt: 'Hi', label: 'chat' });
    expect(await answer.text).toBe('from the backup');
    expect(events).toEqual([expect.objectContaining({ label: 'chat', stream: true, fallback: true, model: expect.objectContaining({ name: 'backup' }) })]);
  });

  it('rejects its promises and ends its readers with the error; the UI stream says only that one occurred', async () => {
    const { moduleRef: ref, ai } = await createAi({ models: { default: new FakeModel([providerError(500)]) }, defaults: { retry: false } });
    moduleRef = ref;

    const answer = ai.streamText({ prompt: 'Hi' });
    await expect(answer.text).rejects.toBeInstanceOf(AiModelError);
    await expect(collect(answer.textStream)).rejects.toBeInstanceOf(AiModelError);
    expect(parseSse(await answer.toUIMessageStreamResponse().text())).toEqual([{ type: 'error', errorText: 'An error occurred.' }, '[DONE]']);
    const explained = answer.toUIMessageStreamResponse({ errorText: (error) => (error instanceof AiModelError ? 'The assistant is unavailable.' : 'Oops') });
    expect(parseSse(await explained.text())[0]).toEqual({ type: 'error', errorText: 'The assistant is unavailable.' });
  });

  it('does not crash the process when nobody reads a stream that fails', async () => {
    const { moduleRef: ref, ai } = await createAi({ models: { default: new FakeModel([providerError(500)]) }, defaults: { retry: false } });
    moduleRef = ref;
    const answer = ai.streamText({ prompt: 'Hi' });
    await answer.result.catch(() => {});
    // vitest fails the run on an unhandled rejection: reaching here without one is the assertion.
    await new Promise((resolve) => setTimeout(resolve, 10));
  });

  it('aborts with abort(), and with the caller\'s signal', async () => {
    const model = new FakeModel(() => ({ text: 'a long story told slowly', chunkDelay: 50 }));
    const { moduleRef: ref, ai } = await createAi({ models: { default: model } });
    moduleRef = ref;

    const answer = ai.streamText({ prompt: 'Tell me a story' });
    const reader = answer.textStream[Symbol.asyncIterator]();
    await reader.next();
    answer.abort();
    await expect(answer.text).rejects.toMatchObject({ name: 'AbortError' });
    await vi.waitFor(() => expect(model.lastCall!.options.abortSignal?.aborted).toBe(true));

    const controller = new AbortController();
    const second = ai.streamText({ prompt: 'Another', abortSignal: controller.signal });
    controller.abort(new Error('the user left'));
    await expect(second.text).rejects.toThrow('the user left');
  });

  it("streams a tool's progress as transient data chunks, which useChat() hands to onData, not to the message", async () => {
    const model = new FakeModel([{ toolCalls: [{ name: 'count' }] }, 'Counted.']);
    const { moduleRef: ref, ai } = await createAi({ models: { default: model } }, { providers: [StockTools] });
    moduleRef = ref;

    const chunks = parseSse(await ai.streamText({ prompt: 'Count the stock', tools: ['count'] }).toUIMessageStreamResponse({ keepAlive: false }).text());
    const progress = chunks.filter((chunk) => (chunk as { type: string }).type === 'data-tool-progress');
    expect(progress).toEqual([
      { type: 'data-tool-progress', id: 'call-1', data: { toolCallId: 'call-1', toolName: 'count', progress: 1, total: 2, message: 'Aisle 1' }, transient: true },
      { type: 'data-tool-progress', id: 'call-1', data: { toolCallId: 'call-1', toolName: 'count', progress: 2, total: 2, message: 'Aisle 2' }, transient: true },
    ]);
    let message: UIMessage | undefined;
    const stream = new ReadableStream({
      start(controller) {
        chunks.filter((chunk) => chunk !== '[DONE]').forEach((chunk) => controller.enqueue(chunk));
        controller.close();
      },
    });
    for await (const state of readUIMessageStream({ stream: stream as never })) {
      message = state as UIMessage;
    }
    expect(message!.parts.map((part) => part.type)).not.toContain('data-tool-progress');

    // Without a stream, progress goes nowhere, and the tool runs all the same.
    const model2 = new FakeModel([{ toolCalls: [{ name: 'count' }] }, 'Counted.']);
    expect((await ai.generateText({ model: model2, prompt: 'Count', tools: ['count'] })).toolResults[0]!.output).toEqual({ counted: 2 });
  });

  it('throws mistakes in its options at once, before anything is sent', async () => {
    const model = new FakeModel([]);
    const { moduleRef: ref, ai } = await createAi({ models: { default: model } });
    moduleRef = ref;
    expect(() => ai.streamText({ prompt: 'Hi', tools: ['nope'] })).toThrow(AiConfigurationError);
    expect(() => ai.streamText({ model: 'nope', prompt: 'Hi' })).toThrow('There is no model named "nope"');
    expect(model.calls).toHaveLength(0);
  });
  it('cleans up when a real HTTP client disconnects during backpressure', async () => {
    const cancel = vi.fn();
    const body = new ReadableStream<Uint8Array>({
      pull(controller) { controller.enqueue(new Uint8Array(1024 * 1024)); },
      cancel,
    });
    let serverResponse: ServerResponse | undefined;
    let sending: Promise<void> | undefined;
    const server = createServer((_, res) => {
      serverResponse = res;
      sending = writeWebResponse(new Response(body), res);
    });
    server.listen(0, '127.0.0.1');
    await once(server, 'listening');
    const address = server.address() as { port: number };
    const request = get(`http://127.0.0.1:${address.port}`, (incoming) => incoming.pause());
    request.on('error', () => {});
    try {
      // A paused client eventually fills the socket buffers; no response methods are mocked.
      await vi.waitFor(() => {
        expect(serverResponse?.writableNeedDrain).toBe(true);
        expect(serverResponse?.listenerCount('drain')).toBeGreaterThan(0);
      });
      request.destroy();
      await sending;
      expect(cancel).toHaveBeenCalledOnce();
      expect(serverResponse!.listenerCount('drain')).toBe(0);
      expect(serverResponse!.listenerCount('close')).toBe(0);
      expect(body.locked).toBe(false);
    } finally {
      request.destroy();
      server.closeAllConnections();
      await new Promise<void>((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
    }
  });

  it('does not wait for cancellation before destroying a failed response', async () => {
    const cancel = vi.fn(() => new Promise<void>(() => {}));
    const body = new ReadableStream<Uint8Array>({
      start(controller) { controller.enqueue(new Uint8Array([1])); },
      cancel,
    });
    const res = response();
    res.write.mockImplementation(() => { throw new Error('write failed'); });
    await writeWebResponse(new Response(body), res as unknown as ServerResponse);
    expect(cancel).toHaveBeenCalledOnce();
    expect(res.destroy).toHaveBeenCalledOnce();
    expect(body.locked).toBe(false);
    expect(res.listenerCount('close')).toBe(0);
  });

  it('releases the reader and listeners after normal completion', async () => {
    const body = new ReadableStream<Uint8Array>({ start(controller) { controller.close(); } });
    const res = response();
    await writeWebResponse(new Response(body), res as unknown as ServerResponse);
    expect(body.locked).toBe(false);
    expect(res.listenerCount('close')).toBe(0);
    expect(res.end).toHaveBeenCalledOnce();
  });

  it('removes a pending drain listener and cancels the body on disconnect', async () => {
    const cancel = vi.fn();
    const body = new ReadableStream<Uint8Array>({
      start(controller) { controller.enqueue(new Uint8Array([1])); },
      cancel,
    });
    const res = response();
    res.write.mockReturnValue(false);
    const sending = writeWebResponse(new Response(body), res as unknown as ServerResponse);
    await vi.waitFor(() => expect(res.listenerCount('drain')).toBe(1));
    res.emit('close');
    await sending;
    expect(cancel).toHaveBeenCalledOnce();
    expect(body.locked).toBe(false);
    expect(res.listenerCount('close')).toBe(0);
    expect(res.listenerCount('drain')).toBe(0);
    expect(res.destroy).toHaveBeenCalledOnce();
  });

  it('releases the reader when the body fails', async () => {
    const body = new ReadableStream<Uint8Array>({ start(controller) { controller.error(new Error('read failed')); } });
    const res = response();
    await writeWebResponse(new Response(body), res as unknown as ServerResponse);
    expect(body.locked).toBe(false);
    expect(res.listenerCount('close')).toBe(0);
    expect(res.destroy).toHaveBeenCalledOnce();
  });

  it('cancels the body when response headers cannot be written', async () => {
    const cancel = vi.fn();
    const body = new ReadableStream<Uint8Array>({ cancel });
    const res = response();
    const error = new Error('headers failed');
    res.writeHead.mockImplementation(() => { throw error; });
    await expect(writeWebResponse(new Response(body), res as unknown as ServerResponse)).rejects.toBe(error);
    expect(cancel).toHaveBeenCalledOnce();
    expect(body.locked).toBe(false);
  });

  it.each(['open', 'create'] as const)('aborts production and removes the close listener when %s fails', async (stage) => {
    const model = new FakeModel(() => ({ text: 'a long story told slowly', chunkDelay: 50 }));
    const { moduleRef: ref, ai } = await createAi({ models: { default: model } });
    moduleRef = ref;
    const answer = ai.streamText({ prompt: 'Tell me a story' });
    const res = response();
    const error = new Error('setup failed');
    const target = stage === 'open'
      ? { raw: res, hijack() { throw error; }, getHeaders() { return {}; } }
      : res;
    if (stage === 'create') {
      vi.spyOn(answer, 'toUIMessageStreamResponse').mockImplementation(() => { throw error; });
    }
    await expect(answer.pipeUIMessageStreamToResponse(target)).rejects.toBe(error);
    expect(model.lastCall!.options.abortSignal?.aborted).toBe(true);
    expect(res.listenerCount('close')).toBe(0);
    await expect(answer.result).rejects.toBe(error);
  });
});
