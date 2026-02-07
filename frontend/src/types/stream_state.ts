export interface StreamState {
    rawText: string;
    isDone: boolean;
    tokenCount: number;
    startTime: number | null;
    endTime: number | null;
};