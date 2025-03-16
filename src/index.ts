/**
 * /home/funboy/eliza/packages/client-twitch/src/index.ts
 *
 * Twitch plugin for ElizaOS
 *
 * This plugin handles:
 *   - Refreshing the Twitch OAuth token (using secrets from the character file)
 *   - Validating Twitch credentials using Zod
 *   - Selecting the LLM model dynamically (using runtime settings)
 *   - Opening a WebSocket connection to Twitch EventSub and subscribing to channel.chat.message
 *   - Receiving notifications (filtering out messages from the bot)
 *   - Creating a user message memory and composing the conversation state once
 *   - Generating a final response by instructing the model to produce valid JSON output
 *   - Marking the user message as processed so it is removed from the context
 *   - Processing post-actions and evaluation
 *   - Sending the final reply to Twitch via the Helix Chat API (with required sender_id)
 */

import WebSocket from "ws";
import fetch, { Headers } from "node-fetch";
import JSON5 from "json5";
import fs from "fs/promises";
import path from "path";
import { z } from "zod";
import {
  elizaLogger,
  IAgentRuntime,
  stringToUuid,
  getEmbeddingZeroVector,
  composeContext,
  generateMessageResponse,
  ModelClass,
  Memory,
  Content,
  Client,
  ClientInstance,
} from "@elizaos/core";

/* ============================================================================
   1) Twitch Configuration Validation
   ----------------------------------------------------------------------------
   We validate essential Twitch credentials using Zod.
=========================================================================== */
const twitchEnvSchema = z.object({
  TWITCH_BOT_USER_ID: z.string().min(1),
  TWITCH_BOT_USERNAME: z.string().min(1),
  TWITCH_OAUTH_TOKEN: z.string().min(1),
  TWITCH_CLIENT_ID: z.string().min(1),
  TWITCH_CHANNEL_USER_ID: z.string().min(1),
});
type TwitchConfig = z.infer<typeof twitchEnvSchema>;

async function validateTwitchConfig(runtime: IAgentRuntime): Promise<TwitchConfig> {
  const cfg = {
    TWITCH_BOT_USER_ID: (runtime.getSetting("TWITCH_BOT_USER_ID") || "") as string,
    TWITCH_BOT_USERNAME: (runtime.getSetting("TWITCH_BOT_USERNAME") || "") as string,
    TWITCH_OAUTH_TOKEN: (runtime.getSetting("TWITCH_OAUTH_TOKEN") || "") as string,
    TWITCH_CLIENT_ID: (runtime.getSetting("TWITCH_CLIENT_ID") || "") as string,
    TWITCH_CHANNEL_USER_ID: (runtime.getSetting("TWITCH_CHANNEL_USER_ID") || "") as string,
  };
  elizaLogger.debug("[Twitch] validateTwitchConfig =>", cfg);
  return twitchEnvSchema.parse(cfg);
}

/* ============================================================================
   2) Twitch OAuth Token Refresh
   ----------------------------------------------------------------------------
   Refreshes the OAuth token using credentials stored in the character file.
=========================================================================== */
interface ExtendedCharacterSettings {
  secrets?: {
    TWITCH_CLIENT_ID?: string;
    TWITCH_CLIENT_SECRET?: string;
    TWITCH_REFRESH_TOKEN?: string;
    TWITCH_OAUTH_TOKEN?: string;
    TWITCH_CHANNEL_USER_ID?: string;
    TWITCH_BOT_USER_ID?: string;
  };
}
interface TwitchRefreshResponse {
  access_token?: string;
  refresh_token?: string;
  expires_in?: number;
}

async function refreshTwitchToken(runtime: IAgentRuntime): Promise<boolean> {
  try {
    const extCharacter = runtime.character as ExtendedCharacterSettings & { filePath?: string };
    const characterFilePath = extCharacter.filePath
      ? extCharacter.filePath
      : path.join(process.cwd(), "..", "characters", `${runtime.character.name}.character.json`);

    elizaLogger.debug("[Twitch] refreshTwitchToken: Loading character file from", characterFilePath);
    const raw = await fs.readFile(characterFilePath, "utf8");
    const config = JSON5.parse(raw);
    const settings: ExtendedCharacterSettings = config.settings || {};

    if (!settings.secrets) {
      elizaLogger.warn("[Twitch] refreshTwitchToken: No 'settings.secrets' found. Skipping token refresh.");
      return false;
    }

    const secrets = settings.secrets;
    const clientId = secrets.TWITCH_CLIENT_ID || "";
    const clientSecret = secrets.TWITCH_CLIENT_SECRET || "";
    const refreshTk = secrets.TWITCH_REFRESH_TOKEN || "";

    if (!clientId || !clientSecret || !refreshTk) {
      elizaLogger.warn("[Twitch] refreshTwitchToken: Missing clientId/clientSecret/refreshToken. Skipping refresh.");
      return false;
    }

    elizaLogger.info(`[Twitch] Refreshing token with clientId=${clientId}`);

    const bodyParams = new URLSearchParams({
      client_id: clientId,
      client_secret: clientSecret,
      grant_type: "refresh_token",
      refresh_token: refreshTk,
    });

    const res = await fetch("https://id.twitch.tv/oauth2/token", {
      method: "POST",
      headers: new Headers({ "Content-Type": "application/x-www-form-urlencoded" }),
      body: bodyParams,
    });

    const json = (await res.json().catch(() => ({}))) as TwitchRefreshResponse;
    if (!res.ok || !json.access_token || !json.refresh_token) {
      elizaLogger.error("[Twitch] Token refresh failed. Response:", json);
      return false;
    }

    elizaLogger.info(`[Twitch] Token refreshed successfully. New access_token=${json.access_token}`);
    settings.secrets.TWITCH_OAUTH_TOKEN = json.access_token;
    settings.secrets.TWITCH_REFRESH_TOKEN = json.refresh_token;

    await fs.writeFile(characterFilePath, JSON.stringify(config, null, 2), "utf8");
    return true;
  } catch (err) {
    elizaLogger.error("[Twitch] refreshTwitchToken error:", err);
    return false;
  }
}

/* ============================================================================
   3) Dynamic Model Selection
   ----------------------------------------------------------------------------
   Returns the selected model from runtime settings, defaulting to "deepseek-r1:14b".
=========================================================================== */
function getSelectedModel(runtime: IAgentRuntime): string {
  const fromSettings =
    (runtime.getSetting("model") as string | undefined) ||
    (runtime.getSetting("LARGE_OLLAMA_MODEL") as string | undefined) ||
    "deepseek-r1:14b";
  return fromSettings.toLowerCase();
}

/* ============================================================================
   4) Generate Response with Logs and 120s Timeout
   ----------------------------------------------------------------------------
   Wraps generateMessageResponse with detailed logging and a timeout.
=========================================================================== */
async function generateResponseWithLogs(
  runtime: IAgentRuntime,
  prompt: string,
  memory: Memory
): Promise<Content> {
  elizaLogger.debug("[Twitch] LLM prompt (truncated):", prompt.slice(0, 600) + (prompt.length > 600 ? "..." : ""));
  elizaLogger.debug("[Twitch] Memory info =>", {
    memoryId: memory.id,
    userId: memory.userId,
    roomId: memory.roomId,
  });

  const selectedModel = getSelectedModel(runtime);
  elizaLogger.debug("[Twitch] Will request LLM with model =>", selectedModel);

  const hypotheticalRequest = {
    modelProvider: runtime.modelProvider,
    model: selectedModel,
    context: prompt,
  };
  elizaLogger.debug("[Twitch] Hypothetical LLM request =>", hypotheticalRequest);

  const TIMEOUT_MS = 120000;
  const genMsgResponse = generateMessageResponse as unknown as (args: {
    runtime: IAgentRuntime;
    context: string;
    modelClass: ModelClass;
    model: string;
  }) => Promise<Content>;

  const responsePromise = genMsgResponse({
    runtime,
    context: prompt,
    modelClass: ModelClass.LARGE,
    model: selectedModel,
  });

  const timeoutPromise = new Promise<Content>((_, reject) =>
    setTimeout(() => reject(new Error(`LLM response timed out after ${TIMEOUT_MS}ms`)), TIMEOUT_MS)
  );

  let result: Content;
  try {
    result = await Promise.race([responsePromise, timeoutPromise]);
  } catch (err) {
    elizaLogger.error("[Twitch] generateResponseWithLogs => Timeout or error =>", err);
    return { text: "", source: "twitch" };
  }

  if (!result) {
    elizaLogger.error("[Twitch] LLM returned null or undefined");
    return { text: "", source: "twitch" };
  }

  elizaLogger.info("[Twitch] LLM output text =>", result.text);
  elizaLogger.debug("[Twitch] Full LLM response =>", result);
  return result;
}

/* ============================================================================
   5) Twitch Client Class
   ----------------------------------------------------------------------------
   Manages the Twitch connection via WebSocket, processes notifications, composes prompts,
   calls the LLM, marks messages as processed, and sends final replies via Twitch Helix Chat API.
=========================================================================== */
class TwitchClient {
  private runtime: IAgentRuntime;
  private ws: WebSocket | null = null;
  private sessionId: string | null = null;
  private lastProcessedMessageIds = new Set<string>();

  constructor(runtime: IAgentRuntime) {
    this.runtime = runtime;
    elizaLogger.debug("[TwitchClient] Constructor: loaded runtime");
  }

  public get name(): string {
    return "twitch";
  }

  /**
   * Starts the Twitch client: refreshes token, validates config and token, opens WebSocket,
   * and subscribes to channel.chat.message events.
   */
  public async start(): Promise<TwitchClient> {
    elizaLogger.info("[TwitchClient] Starting client...");

    await refreshTwitchToken(this.runtime);
    const cfg = await validateTwitchConfig(this.runtime);
    await this.validateToken(cfg.TWITCH_OAUTH_TOKEN);

    this.ws = new WebSocket("wss://eventsub.wss.twitch.tv/ws");

    this.ws.on("open", () => {
      elizaLogger.info("[TwitchClient] WebSocket connection opened");
    });

    this.ws.on("error", (err) => {
      elizaLogger.error("[TwitchClient] WebSocket error =>", err);
    });

    this.ws.on("close", (code, reason) => {
      elizaLogger.warn(`[TwitchClient] WebSocket closed => code=${code}, reason=${reason.toString()}`);
      this.ws = null;
    });

    this.ws.on("message", async (data: Buffer) => {
      const raw = data.toString();
      elizaLogger.debug("[TwitchClient] Received raw WS data =>", raw);
      await this.handleWebSocketMessage(raw);
    });

    return this;
  }

  /**
   * Stops the Twitch client by closing the WebSocket.
   */
  public async stop(): Promise<void> {
    elizaLogger.info("[TwitchClient] Stopping client...");
    if (this.ws) {
      this.ws.close();
      this.ws = null;
      elizaLogger.info("[TwitchClient] Client stopped – WebSocket closed");
    } else {
      elizaLogger.info("[TwitchClient] No active WebSocket connection to stop");
    }
  }

  /**
   * Validates the Twitch OAuth token using the /oauth2/validate endpoint.
   */
  private async validateToken(token: string): Promise<void> {
    elizaLogger.debug("[TwitchClient] Validating token via /oauth2/validate");
    const res = await fetch("https://id.twitch.tv/oauth2/validate", {
      method: "GET",
      headers: { Authorization: `OAuth ${token}` },
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      elizaLogger.error("[TwitchClient] Token validation failed =>", res.status, data);
      throw new Error("Twitch token invalid");
    }
    elizaLogger.info("[TwitchClient] OAuth token validated");
  }

  /**
   * Handles incoming WebSocket messages by parsing JSON and routing based on message_type.
   */
  private async handleWebSocketMessage(raw: string): Promise<void> {
    let data: any;
    try {
      data = JSON.parse(raw);
    } catch (err) {
      elizaLogger.error("[TwitchClient] JSON parse error =>", err);
      return;
    }
    const msgType = data.metadata?.message_type;
    elizaLogger.debug("[TwitchClient] WS message_type =>", msgType);

    if (msgType === "session_welcome") {
      this.sessionId = data.payload?.session?.id;
      elizaLogger.info(`[TwitchClient] session_welcome => sessionId=${this.sessionId}`);
      await this.subscribeToChat();
    } else if (msgType === "session_keepalive") {
      elizaLogger.debug("[TwitchClient] Received session_keepalive (heartbeat)");
    } else if (msgType === "notification") {
      elizaLogger.info("[TwitchClient] Received notification => handleNotification");
      await this.handleNotification(data).catch((err) => {
        elizaLogger.error("[TwitchClient] Error in handleNotification =>", err);
      });
    } else {
      elizaLogger.info("[TwitchClient] Unhandled WS message_type =>", msgType);
    }
  }

  /**
   * Subscribes to Twitch channel.chat.message events.
   */
  private async subscribeToChat(): Promise<void> {
    if (!this.sessionId) {
      throw new Error("[TwitchClient] subscribeToChat => no sessionId");
    }
    const cfg = await validateTwitchConfig(this.runtime);

    const body = {
      type: "channel.chat.message",
      version: "1",
      condition: {
        broadcaster_user_id: cfg.TWITCH_CHANNEL_USER_ID || "",
        user_id: cfg.TWITCH_BOT_USER_ID || "",
      },
      transport: {
        method: "websocket",
        session_id: this.sessionId,
      },
    };

    elizaLogger.info(
      `[TwitchClient] Subscribing to channel.chat.message (broadcaster=${cfg.TWITCH_CHANNEL_USER_ID}, bot=${cfg.TWITCH_BOT_USER_ID})`
    );
    elizaLogger.debug("[TwitchClient] Subscription body =>", body);

    const resp = await fetch("https://api.twitch.tv/helix/eventsub/subscriptions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${cfg.TWITCH_OAUTH_TOKEN}`,
        "Client-Id": cfg.TWITCH_CLIENT_ID,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const txt = await resp.text();
      elizaLogger.error("[TwitchClient] subscribeToChat error =>", txt);
      throw new Error("[TwitchClient] Failed to subscribe to chat messages");
    }

    elizaLogger.info("[TwitchClient] Subscribed to channel.chat.message");
  }

  /**
   * Processes a notification from Twitch:
   *   - Logs details, deduplicates by message_id,
   *   - Skips messages from the bot,
   *   - And forwards the user text to onUserMessage.
   */
  private async handleNotification(data: any): Promise<void> {
    const evt = data.payload?.event;
    const text = evt?.message?.text || "";
    const senderId = evt?.chatter_user_id || "unknown-user";
    const senderName = evt?.chatter_user_name || "UnknownUser";
    const messageId = evt?.message_id || "";

    elizaLogger.info("[TwitchClient] Notification =>", {
      messageId,
      senderId,
      senderName,
      text,
    });

    const cfg = await validateTwitchConfig(this.runtime);
    if (senderId === cfg.TWITCH_BOT_USER_ID) {
      elizaLogger.warn("[TwitchClient] ignoring message from the bot => loop prevention");
      return;
    }

    // Deduplicate by messageId
    if (messageId && this.lastProcessedMessageIds.has(messageId)) {
      elizaLogger.warn(`[TwitchClient] ignoring repeated message => id=${messageId}`);
      return;
    }
    if (messageId) {
      this.lastProcessedMessageIds.add(messageId);
      if (this.lastProcessedMessageIds.size > 100) {
        const firstKey = this.lastProcessedMessageIds.values().next().value;
        if (typeof firstKey === "string") {
          this.lastProcessedMessageIds.delete(firstKey);
        }
      }
    }

    elizaLogger.info(`[TwitchClient] Processing => ${senderName}: "${text}"`);
    await this.onUserMessage(senderId, text);
  }

  /**
   * Handles an incoming user message:
   * 1. Creates a memory for the user's message.
   * 2. Composes the state once from that message.
   * 3. Builds a final prompt that forces the LLM to output valid JSON.
   * 4. Generates a response from the LLM.
   * 5. Creates a memory for the agent's reply.
   * 6. Marks the user message as processed.
   * 7. Updates the conversation state to filter out the processed message.
   * 8. Processes actions, evaluates, and sends the final reply via Twitch API.
   */
  private async onUserMessage(senderId: string, text: string): Promise<void> {
    const cfg = await validateTwitchConfig(this.runtime);
    const roomId = stringToUuid(`twitch-${cfg.TWITCH_CHANNEL_USER_ID}`);

    elizaLogger.info(`[TwitchClient] Creating memory => user=${senderId}, text="${text}"`);
    // Create user message; add an extra property "processed" (initially undefined)
    const userMsg: Memory & { processed?: boolean } = {
      id: stringToUuid(`twitch-msg-${Date.now()}-${senderId}`),
      agentId: this.runtime.agentId,
      userId: stringToUuid(senderId),
      roomId,
      content: { text, source: "twitch" },
      createdAt: Date.now(),
      embedding: getEmbeddingZeroVector(),
    };
    await this.runtime.messageManager.addEmbeddingToMemory(userMsg);
    await this.runtime.messageManager.createMemory(userMsg);

    // Compose the state from the user message
    const state = await this.runtime.composeState(userMsg);

    // Build the final prompt with instructions to output valid JSON
    const finalPrompt = `User asked: "${text}"
Generate a short, final Twitch reply in valid JSON format.
The output must be a JSON object with a single key "text" whose value is your final answer.
Do not include any additional text or commentary.
Final Answer:`;
    elizaLogger.debug("[TwitchClient] Final prompt (truncated) =>", finalPrompt.slice(0, 400) + (finalPrompt.length > 400 ? "..." : ""));

    // Generate LLM response with a 120-second timeout
    const replyContent = await generateResponseWithLogs(this.runtime, finalPrompt, userMsg);
    if (!replyContent.text) {
      elizaLogger.warn("[TwitchClient] LLM returned empty text => skipping send");
      return;
    }
    elizaLogger.info("[TwitchClient] LLM reply =>", replyContent.text);

    // Create memory for the agent's reply
    const agentMsg: Memory = {
      id: stringToUuid(`twitch-reply-${Date.now()}`),
      agentId: this.runtime.agentId,
      userId: this.runtime.agentId,
      roomId,
      content: replyContent,
      createdAt: Date.now(),
      embedding: getEmbeddingZeroVector(),
    };
    await this.runtime.messageManager.createMemory(agentMsg);

    // Mark the user message as processed
    userMsg.processed = true;

    // Update the state and filter out the processed user message from recentMessages.
    const updatedState = await this.runtime.updateRecentMessageState(state);
    const filteredRecent = updatedState.recentMessages
      .split("\n")
      .filter(line => !line.includes(String(userMsg.id)))
      .join("\n");
    const filteredState = { ...updatedState, recentMessages: filteredRecent };

    // Process actions and run evaluation using the filtered state
    await this.runtime.processActions(userMsg, [agentMsg], filteredState, async (msgs) => {
      elizaLogger.debug("[TwitchClient] processActions => newMessages:", msgs);
      return [userMsg];
    });
    await this.runtime.evaluate(userMsg, filteredState);

    elizaLogger.info("[TwitchClient] *** SENDING MESSAGE *** =>", replyContent.text);
    await this.sendTwitchMessage(replyContent.text);
  }

  /**
   * Sends the final reply message to Twitch using the Helix Chat API.
   * Now includes the required 'sender_id' parameter.
   */
  private async sendTwitchMessage(text: string): Promise<void> {
    const cfg = await validateTwitchConfig(this.runtime);
    const url = `https://api.twitch.tv/helix/chat/messages?broadcaster_id=${cfg.TWITCH_CHANNEL_USER_ID}&moderator_id=${cfg.TWITCH_BOT_USER_ID}`;
    elizaLogger.info("[TwitchClient] Attempting to POST =>", url, {
      textPreview: text.slice(0, 80) + (text.length > 80 ? "..." : ""),
    });

    const res = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${cfg.TWITCH_OAUTH_TOKEN}`,
        "Client-Id": cfg.TWITCH_CLIENT_ID,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: text,
        sender_id: cfg.TWITCH_BOT_USER_ID // Required by Twitch
      }),
    });

    if (!res.ok) {
      const errTxt = await res.text();
      elizaLogger.error("[TwitchClient] sendTwitchMessage => error", errTxt);
    } else {
      elizaLogger.info(`[TwitchClient] Sent message => "${text}"`);
    }
  }
}

/* ============================================================================
   6) Plugin Interface for ElizaOS
   ----------------------------------------------------------------------------
   Exports the Twitch client plugin as an object conforming to the ElizaOS Client interface.
=========================================================================== */
export const TwitchClientInterface: Client = {
  name: "twitch",
  config: {},
  start: async (runtime: IAgentRuntime): Promise<ClientInstance> => {
    elizaLogger.info("[Twitch] Starting Twitch plugin...");
    const client = new TwitchClient(runtime);
    await client.start();
    elizaLogger.info(`[Twitch] Client started for character "${runtime.character.name}"`);
    return client as unknown as ClientInstance;
  },
} as Client;

// Provide a stop() method for the plugin interface.
(TwitchClientInterface as any).stop = async (_runtime: IAgentRuntime, client?: ClientInstance) => {
  elizaLogger.info("[Twitch] Attempting to stop TwitchClient...");
  if (client && typeof (client as any).stop === "function") {
    await (client as any).stop();
    elizaLogger.info("[Twitch] Client stopped");
  } else {
    elizaLogger.warn("[Twitch] No valid stop() method found on the client.");
  }
};

export default TwitchClientInterface;
