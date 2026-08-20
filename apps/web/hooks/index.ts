"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { authApi } from "@/lib/api/auth";
import { chatApi } from "@/lib/api/chat";
import { modelsApi } from "@/lib/api/models";
import { datasetsApi } from "@/lib/api/datasets";
import { trainingApi } from "@/lib/api/training";
import { knowledgeApi } from "@/lib/api/knowledge";
import { agentsApi } from "@/lib/api/agents";

export function useMe() {
  return useQuery({ queryKey: ["auth", "me"], queryFn: authApi.me });
}

export function useConversations() {
  return useQuery({
    queryKey: ["conversations"],
    queryFn: chatApi.conversations,
  });
}

export function useConversation(id: string) {
  return useQuery({
    queryKey: ["conversations", id],
    queryFn: () => chatApi.conversation(id),
    enabled: !!id,
  });
}

export function useMessages(conversationId: string) {
  return useQuery({
    queryKey: ["messages", conversationId],
    queryFn: () => chatApi.messages(conversationId),
    enabled: !!conversationId,
  });
}

export function useModels() {
  return useQuery({ queryKey: ["models"], queryFn: modelsApi.list });
}

export function useModel(id: string) {
  return useQuery({
    queryKey: ["models", id],
    queryFn: () => modelsApi.get(id),
    enabled: !!id,
  });
}

export function useDatasets() {
  return useQuery({ queryKey: ["datasets"], queryFn: datasetsApi.list });
}

export function useTrainingJobs() {
  return useQuery({ queryKey: ["training"], queryFn: trainingApi.list });
}

export function useKnowledgeBases() {
  return useQuery({ queryKey: ["knowledge"], queryFn: knowledgeApi.list });
}

export function useAgents() {
  return useQuery({ queryKey: ["agents"], queryFn: agentsApi.list });
}

export function useSetActiveModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, active }: { id: string; active: boolean }) =>
      modelsApi.setActive(id, active),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["models"] }),
  });
}

export function useCreateConversation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ title, model }: { title: string; model: string }) =>
      chatApi.createConversation(title, model),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["conversations"] }),
  });
}