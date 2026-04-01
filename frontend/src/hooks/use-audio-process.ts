import { useMutation, useQuery } from "@tanstack/react-query";

export interface JobStatus {
  jobId: string;
  status: "pending" | "processing" | "completed" | "failed";
  stage: string | null;
  progress: number;
  error?: string | null;
  summary?: {
    segmentsRemoved: number;
    segmentsKept: number;
    durationSecs: number;
    removedSecs: number;
  } | null;
}

const CHUNK_SIZE = 512 * 1024;

async function uploadFileInChunks(
  file: File,
  removeFatiha: string,
  onProgress?: (pct: number) => void,
): Promise<string> {
  const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
  let uploadId: string | null = null;

  for (let i = 0; i < totalChunks; i++) {
    const start = i * CHUNK_SIZE;
    const end = Math.min(start + CHUNK_SIZE, file.size);
    const chunk = file.slice(start, end);

    const fd = new FormData();
    fd.append("chunk", chunk, file.name);
    fd.append("chunk_index", String(i));
    fd.append("total_chunks", String(totalChunks));
    if (uploadId) fd.append("upload_id", uploadId);
    if (i === totalChunks - 1) fd.append("remove_fatiha", removeFatiha);

    const res = await fetch("/api/upload-chunk", { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: "Chunk upload failed" }));
      throw new Error(err.error || "Chunk upload failed");
    }

    const data = await res.json();
    uploadId = data.uploadId;
    onProgress?.(Math.round(((i + 1) / totalChunks) * 100));

    if (data.jobId) return data.jobId as string;
  }

  throw new Error("Upload did not complete — no job ID returned");
}

export function useUploadAudioMutation(onUploadProgress?: (pct: number) => void) {
  return useMutation({
    mutationFn: async (data: {
      audio?: File;
      youtube_url?: string;
      remove_fatiha: string;
    }) => {
      if (data.audio) {
        const jobId = await uploadFileInChunks(
          data.audio,
          data.remove_fatiha,
          onUploadProgress,
        );
        return { jobId };
      }

      const fd = new FormData();
      if (data.youtube_url) fd.append("youtube_url", data.youtube_url);
      fd.append("remove_fatiha", data.remove_fatiha);

      const res = await fetch("/api/upload", { method: "POST", body: fd });
      if (!res.ok) {
        const error = await res.json().catch(() => ({ error: "Upload failed" }));
        throw new Error(error.error || "Failed to upload");
      }
      return res.json() as Promise<{ jobId: string }>;
    },
  });
}

export function useJobStatusQuery(jobId: string | null) {
  return useQuery({
    queryKey: ["/api/status", jobId],
    queryFn: async () => {
      if (!jobId) throw new Error("No job ID");
      const res = await fetch(`/api/status/${jobId}`);
      if (!res.ok) throw new Error("Failed to fetch job status");
      return res.json() as Promise<JobStatus>;
    },
    enabled: !!jobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === "completed" || status === "failed") return false;
      return 2000;
    },
  });
}
