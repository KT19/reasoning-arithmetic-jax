import { useMemo } from "react";
import type { StreamState } from "../types/stream_state";
import { BarChart3, Brain, CheckCircle2, Cpu } from "lucide-react";

interface ModelPanelProp {
  title: string;
  state: StreamState;
  color: string;
}

export const ModelPanel = ({ title, state, color }: ModelPanelProp) => {
  const { thought, answer } = useMemo(() => {
    const thinkMatch = state.rawText.match(/<think>([\s\S]*?)<\/think>/);
    const t = thinkMatch ? thinkMatch[1].trim() : "";
    const a = state.rawText.replace(/<think>[\s\S]*?<\/think>/, "").trim();
    return { thought: t, answer: a };
  }, [state.rawText]);

  return (
    <div
      className={`flex flex-col h-175 bg-gray-800 rounded-2xl border border-gray-700 overflow-hidden shadow-2xl transition-all ${state.isDone ? "ring-2 ring-" + color + "-500" : ""}`}
    >
      <div className="p-4 bg-gray-800/50 border-b border-gray-700 flex justify-between items-center">
        <div className="flex items-center gap-2">
          <Cpu className={`text-${color}-400`} size={20} />
          <h2 className="font-bold text-lg">{title}</h2>
        </div>
        {state.isDone && (
          <CheckCircle2 className="text-emerald-400" size={20} />
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-5 space-y-4">
        {/* Reasoning Block */}
        {thought && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-xs font-bold text-gray-500 uppercase tracking-widest">
              <Brain size={14} /> Reasoning
            </div>
            <div className="bg-gray-900/50 border border-gray-700 rounded-xl p-4 text-gray-300 italic text-sm leading-relaxed">
              {thought}
            </div>
          </div>
        )}

        {/* Final Answer */}
        <div className="space-y-2">
          <div className="text-xs font-bold text-gray-500 uppercase tracking-widest">
            Output
          </div>
          <div className="bg-gray-950 border border-gray-800 rounded-xl p-4 font-mono text-indigo-300 whitespace-pre-wrap">
            {answer || (state.startTime && !state.isDone ? "Thinking..." : "")}
          </div>
        </div>
      </div>

      {/* Stats Footer */}
      {state.isDone && (
        <div className="p-4 bg-gray-900 border-t border-gray-700 grid grid-cols-2 gap-4">
          <div className="flex items-center gap-3">
            <BarChart3 className="text-gray-500" size={18} />
            <div>
              <p className="text-[10px] text-gray-500 uppercase font-bold">
                Total Tokens
              </p>
              <p className="font-mono text-lg">{state.tokenCount}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
