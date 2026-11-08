import type { TestingModule } from '@nestjs/testing';
import { Test } from '@nestjs/testing';
import { WorkflowsModule, type WorkflowContext } from '@nestjs/workflows';
import {
  AiConfigurationError,
  AiModule,
  type AiCallDefaults,
} from '../lib/index.js';
import { FakeModel } from '../lib/testing/index.js';
import { AiAgents, AiWorkflowsModule } from '../lib/workflows/index.js';

describe('durable agent run options', () => {
  let app: TestingModule | undefined;

  afterEach(async () => {
    await app?.close();
    app = undefined;
  });

  async function boot(defaults: AiCallDefaults = {}) {
    const model = new FakeModel([]);
    app = await Test.createTestingModule({
      imports: [
        WorkflowsModule.forRoot({ worker: { enabled: false } }),
        AiModule.forRoot({ models: { default: model }, defaults }),
        AiWorkflowsModule,
      ],
    }).compile();
    app.useLogger(false);
    await app.init();
    const step = vi.fn().mockRejectedValue(new Error('Unexpected model step'));
    const executeChild = vi
      .fn()
      .mockRejectedValue(new Error('Unexpected child workflow'));
    const ctx = {
      workflowId: 'invalid-options',
      step,
      executeChild,
    } as unknown as WorkflowContext;
    return { agents: app.get(AiAgents), model, ctx, step, executeChild };
  }

  const invalidLimits = [0, -1, 1.5, NaN, Infinity, -Infinity];

  describe.each(['maxSteps', 'stepsPerInstance'] as const)('%s', (name) => {
    it.each(invalidLimits)(
      'refuses %s before running a model or child workflow',
      async (value) => {
        const { agents, model, ctx, step, executeChild } = await boot();
        const run = async () =>
          agents.run(ctx, { prompt: 'Hello', [name]: value });

        await expect(run()).rejects.toMatchObject({
          name: 'AiConfigurationError',
          message: `\`${name}\` must be a whole number of at least 1. Got ${String(value)}.`,
        });
        expect(step).not.toHaveBeenCalled();
        expect(executeChild).not.toHaveBeenCalled();
        expect(model.calls).toHaveLength(0);
      },
    );
  });

  it.each(invalidLimits)(
    'refuses defaults.maxSteps = %s before running',
    async (value) => {
      const { agents, model, ctx, step, executeChild } = await boot({
        maxSteps: value,
      });
      const run = async () => agents.run(ctx, { prompt: 'Hello' });

      await expect(run()).rejects.toBeInstanceOf(AiConfigurationError);
      expect(step).not.toHaveBeenCalled();
      expect(executeChild).not.toHaveBeenCalled();
      expect(model.calls).toHaveLength(0);
    },
  );
  describe.each(['maxTokens', 'maxCost'] as const)('budget.%s', (name) => {
    it.each([-1, NaN, Infinity, -Infinity])('rejects invalid limit %s before workflow steps', async (value) => {
      const { agents, model, ctx, step, executeChild } = await boot();
      expect(() => agents.run(ctx, { prompt: 'Hello', budget: { [name]: value } })).toThrow(AiConfigurationError);
      expect(step).not.toHaveBeenCalled();
      expect(executeChild).not.toHaveBeenCalled();
      expect(model.calls).toHaveLength(0);
    });

    it('rejects invalid module budget defaults', async () => {
      const { agents, ctx, step } = await boot({ budget: { [name]: NaN } });
      expect(() => agents.run(ctx, { prompt: 'Hello' })).toThrow(AiConfigurationError);
      expect(step).not.toHaveBeenCalled();
    });
  });

  it('rejects fractional token limits before workflow steps', async () => {
    const { agents, ctx, step } = await boot();
    expect(() => agents.run(ctx, { prompt: 'Hello', budget: { maxTokens: 1.5 } })).toThrow(AiConfigurationError);
    expect(step).not.toHaveBeenCalled();
  });

});
