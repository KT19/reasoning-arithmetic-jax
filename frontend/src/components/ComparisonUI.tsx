import { useState } from "react";
import type { StreamState } from "../types/stream_state";
import { streamModel } from "./streamModel";
import { Play, Terminal } from "lucide-react";
import { ModelPanel } from "./ModelPanel";

export const ComparisonUI = () => {
  const [prompt, setPrompt] = useState("Calculate (1 + 2) * 3");
  const [isRunning, setIsRunning] = useState(false);

  //States for both models
  const [sft, setSft] = useState<StreamState>({
    rawText: "",
    isDone: false,
    tokenCount: 0,
    startTime: null,
    endTime: null,
  });
  const [grpo, setGrpo] = useState<StreamState>({
    rawText: "",
    isDone: false,
    tokenCount: 0,
    startTime: null,
    endTime: null,
  });

  const handleCompare = () => {
    setIsRunning(true);
    //Execute in parallel
    Promise.all([
      streamModel(prompt, "sft", setSft),
      streamModel(prompt, "grpo", setGrpo),
    ]).finally(() => setIsRunning(false));
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 p-6 font-sans">
      <div className="max-w-7xl mx-auto">
        {/* Top Control Bar */}
        <header className="mb-8 flex flex-col gap-4 bg-gray-800 p-6 rounded-2xl border border-gray-700 shadow-xl">
          <div className="flex items-center gap-2 mb-2">
            <Terminal className="text-indigo-400" />
            <h1 className="text-xl font-bold tracking-tight">
              LLM Reasoning Bench: SFT vs GRPO
            </h1>
          </div>
          <div className="flex gap-4">
            <input
              className="flex-1 bg-gray-950 border border-gray-600 rounded-lg px-4 py-3 focus:ring-2 focus:ring-indigo-500 outline-none transition-all"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter arithmetic prompt..."
            />
            <button
              onClick={handleCompare}
              disabled={isRunning}
              className="bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-700 px-8 py-3 rounded-lg font-bold flex items-center gap-2 transition-colors"
            >
              <Play size={18} fill="currentColor" />
              {isRunning ? "Generating..." : "Run Comparison"}
            </button>
          </div>
        </header>

        {/* Comparison Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ModelPanel title="SFT (Supervised)" state={sft} color="blue" />
          <ModelPanel
            title="GRPO (Reinforcement)"
            state={grpo}
            color="emerald"
          />
        </div>
      </div>
    </div>
  );
};
