import type { StreamState } from "../types/stream_state";

const SERVER_URL = "http://localhost:8000/";

export const streamModel = async (prompt: string, modelType: "sft" | "grpo", setter: React.Dispatch<React.SetStateAction<StreamState>>) => {
    setter({rawText: "", isDone: false, tokenCount: 0, startTime: Date.now(), endTime: null});

    try {
        const response = await fetch(`${SERVER_URL}run`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({prompt, model_type: modelType}),
        });

        const reader = response.body?.getReader();
        const decoder = new TextDecoder();

        if(!reader) return;

        while(true) {
            const {value, done} = await reader.read();
            if(done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split("\n");

            for(const line of lines) {
                if(line.startsWith("data: ")) {
                    const dataStr = line.replace("data: ", "").trim();
                    if(dataStr == "[Done]") {
                        setter(prev => ({...prev, isDone: true, endTime: Date.now()}));
                        break;
                    }
                    try {
                        const {token} = JSON.parse(dataStr);
                        setter(prev => ({
                            ...prev,
                            rawText: token,
                            tokenCount: prev.tokenCount+1
                        }));
                    } catch(e) {
                        
                    }
                }
            }
        }
    } catch(err) {
        console.error(`Error streaming ${modelType}:`, err);
    }
};