interface ParsedResponse {
    thought: string;
    answer: string;
}

export const parseResponse = (text: string): ParsedResponse => {
    const thinkMatch = text.match(/<think>([\s\S]*?)<\/think>/);
    const thought = thinkMatch ? thinkMatch[1].trim() : "";
    const answer = text.replace(/<think>[\s\S]*?<\/think>/, "").trim();

    return {thought, answer};
}