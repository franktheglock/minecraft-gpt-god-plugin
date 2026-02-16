package net.bigyous.gptgodmc.GPT;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService; 
import java.util.concurrent.Executors; 

import org.apache.http.HttpHeaders;
import org.apache.http.HttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.ContentType;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClientBuilder;
import org.apache.http.util.EntityUtils;
import org.bukkit.Bukkit;
import org.bukkit.configuration.file.FileConfiguration;
import org.bukkit.plugin.java.JavaPlugin;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import net.bigyous.gptgodmc.GPTGOD;
import net.bigyous.gptgodmc.GPT.Json.GptFunction;
import net.bigyous.gptgodmc.GPT.Json.GptModel;
import net.bigyous.gptgodmc.GPT.Json.GptRequest;
import net.bigyous.gptgodmc.GPT.Json.GptTool;
import net.bigyous.gptgodmc.GPT.Json.ModelSerializer;
import net.bigyous.gptgodmc.GPT.Json.ParameterExclusion;

public class GptAPI {
    private GsonBuilder gson = new GsonBuilder();
    private GptRequest body;
    private Map<String, Integer> messageMap = new HashMap<String, Integer>();
    private boolean isSending = false;
    private static ExecutorService pool = Executors.newCachedThreadPool();

    public GptAPI(GptModel model) {
        this.body = new GptRequest(model, GptActions.GetAllTools());
        gson.registerTypeAdapter(GptModel.class, new ModelSerializer());
        gson.setExclusionStrategies(new ParameterExclusion());
    }

    public GptAPI(GptModel model, GptTool[] customTools) {
        this.body = new GptRequest(model, customTools);
        gson.registerTypeAdapter(GptModel.class, new ModelSerializer());
        gson.setExclusionStrategies(new ParameterExclusion());
    }

    public GptAPI(GptRequest request) {
        this.body = request;
        gson.registerTypeAdapter(GptModel.class, new ModelSerializer());
        gson.setExclusionStrategies(new ParameterExclusion());
    }

    public GptAPI addContext(String context, String name) {
        if (this.messageMap.containsKey(name)) {
            this.body.replaceMessage(messageMap.get(name), context);
            return this;
        }
        this.body.addMessage("system", context);
        this.messageMap.put(name, this.body.getMessagesSize() - 1);
        return this;
    }

    public GptAPI addContext(String context, String name, int index) {
        if (this.messageMap.containsKey(name)) {
            this.body.replaceMessage(messageMap.get(name), context);
            return this;
        }
        this.body.addMessage("system", context);
        for (String key : messageMap.keySet()) {
            if (messageMap.get(key) == index) {
                messageMap.replace(key, index + 1);
            }
        }
        this.messageMap.put(name, index);
        return this;
    }

    public GptAPI setTools(GptTool[] tools) {
        this.body.setTools(tools);
        return this;
    }

    public GptAPI addLogs(String Logs, String name) {
        if (this.messageMap.containsKey(name)) {
            this.body.replaceMessage(messageMap.get(name), Logs);
            return this;
        }
        this.body.addMessage("user", Logs);
        this.messageMap.put(name, this.body.getMessagesSize() - 1);
        return this;
    }

    public GptAPI addLogs(String Logs, String name, int index) {
        if(this.body.getMessagesSize() <= index){
            addLogs(Logs, name);
            return this;
        }
        if (this.messageMap.containsKey(name)) {
            this.body.replaceMessage(messageMap.get(name), Logs);
            return this;
        }
        this.body.addMessage("user", Logs, index);
        for (String key : messageMap.keySet()) {
            if (messageMap.get(key) == index) {
                messageMap.replace(key, index + 1);
            }
        }
        this.messageMap.put(name, index);
        return this;
    }

    public GptAPI setToolChoice(Object tool_choice) {
        this.body.setTool_choice(tool_choice);
        return this;
    }

    public void removeLastMessage() {
        this.body.removeLastMessage();
    }

    public int getMaxTokens() {
        return body.getModel().getTokenLimit();
    }

    public String getModelName() {
        return body.getModel().getName();
    }

    private String providerName(FileConfiguration config){
        return config.getString("inference-provider", "openai").toLowerCase();
    }

    private boolean providerIsOpenAICompatible(String provider){
        return provider.equals("openai") || provider.equals("openrouter") || provider.equals("lmstudio") || provider.equals("generic") || provider.equals("nim");
    }

    private String resolveChatUrl(FileConfiguration config){
        String provider = providerName(config);
        return switch(provider){
            case "openrouter" -> config.getString("openRouterUrl", "https://api.openrouter.ai/v1") + "/chat/completions";
            case "ollama" -> config.getString("ollamaUrl", "http://localhost:11434") + "/api/generate";
            case "lmstudio" -> config.getString("lmstudioUrl", "http://127.0.0.1:8080") + "/v1/chat/completions";
            case "generic" -> config.getString("genericUrl", "https://api.your-provider.example") + "/v1/chat/completions";
            case "nim" -> config.getString("nimUrl", "https://integrate.api.nvidia.com/v1") + "/v1/chat/completions";
            default -> "https://api.openai.com/v1/chat/completions";
        };
    }

    private String resolveAuthHeader(FileConfiguration config){
        String provider = providerName(config);
        return switch(provider){
            case "openrouter" -> "Bearer " + config.getString("openRouterKey", "");
            case "openai" -> "Bearer " + config.getString("openAiKey", "");
            case "generic" -> "Bearer " + config.getString("genericKey", "");
            case "nim" -> "Bearer " + config.getString("nimKey", "");
            default -> null; // local providers typically don't use a bearer header
        };
    }

    private String messagesToPrompt(){
        // convert chat messages to a single textual prompt for providers that expect a prompt string
        StringBuilder prompt = new StringBuilder();
        try{
            for (var m : this.body.getMessages()){
                prompt.append(m.getRole()).append(": ").append(m.getContent()).append("\n");
            }
        } catch(Exception e){
            // fallback to serializing the whole body
            prompt.append(gson.create().toJson(this.body));
        }
        return prompt.toString();
    }

    private String extractTextFromProviderResponse(String raw){
        try{
            var json = gson.fromJson(raw, com.google.gson.JsonElement.class);
            if(json.isJsonObject()){
                var obj = json.getAsJsonObject();
                // common fields used by various local servers
                if(obj.has("text")) return obj.get("text").getAsString();
                if(obj.has("response")) return obj.get("response").getAsString();
                if(obj.has("result")) return obj.get("result").getAsString();
                // try a nested 'choices[0].message.content' like OpenAI
                if(obj.has("choices")){
                    var choices = obj.getAsJsonArray("choices");
                    if(choices.size() > 0){
                        var first = choices.get(0).getAsJsonObject();
                        if(first.has("message")){
                            var msg = first.getAsJsonObject("message");
                            if(msg.has("content")) return msg.get("content").getAsString();
                        }
                        if(first.has("text")) return first.get("text").getAsString();
                    }
                }
            }
        } catch(Exception ignored){ }
        // if we couldn't parse, return raw trimmed
        return raw == null ? "" : raw.trim();
    }

    public void send() {
        CloseableHttpClient client = HttpClientBuilder.create().build();
        pool.execute(() -> {
            this.isSending = true;
            FileConfiguration config = JavaPlugin.getPlugin(GPTGOD.class).getConfig();
            String provider = providerName(config);

            try {
                String url = resolveChatUrl(config);
                StringEntity data;
                HttpPost post = new HttpPost(url);

                if(providerIsOpenAICompatible(provider)){
                    var payload = new com.google.gson.JsonObject();
                    payload.add("messages", gson.toJsonTree(body.getMessages()));
                    payload.addProperty("model", body.getModel().getName());
                    // include tools if present
                    if(body.getTools() != null && body.getTools().length > 0){
                        payload.add("tools", gson.toJsonTree(body.getTools()));
                        if(body.getTool_choice() != null){
                            payload.add("tool_choice", gson.toJsonTree(body.getTool_choice()));
                        }
                    }
                    // NIM-specific extra_body
                    if(provider.equals("nim")){
                        String extra = config.getString("nimExtraBody", "{}");
                        try{
                            var extraJson = com.google.gson.JsonParser.parseString(extra).getAsJsonObject();
                            for(var entry : extraJson.entrySet()){
                                payload.add(entry.getKey(), entry.getValue());
                            }
                        } catch(Exception ignored){}
                    }
                    data = new StringEntity(payload.toString(), ContentType.APPLICATION_JSON);
                    String auth = resolveAuthHeader(config);
                    if(auth != null) post.setHeader(HttpHeaders.AUTHORIZATION, auth);
                } else {
                    // build a simple prompt-based request for local servers (ollama / lmstudio)
                    var map = new java.util.HashMap<String, Object>();
                    map.put("model", body.getModel().getName());
                    map.put("prompt", messagesToPrompt());
                    // don't include functions/tooling for non-OpenAI providers
                    data = new StringEntity(gson.create().toJson(map), ContentType.APPLICATION_JSON);
                }

                GPTGOD.LOGGER.info("POSTING (" + provider + ") " + gson.setPrettyPrinting().create().toJson(body));
                post.setEntity(data);
                GPTGOD.LOGGER.info("Making POST request to " + url);

                HttpResponse response = client.execute(post);
                String raw = new String(response.getEntity().getContent().readAllBytes());
                EntityUtils.consume(response.getEntity());

                if(providerIsOpenAICompatible(provider)){
                    GPTGOD.LOGGER.info("received response from " + provider + ": " + raw);
                    if (response.getStatusLine().getStatusCode() != 200) {
                        GPTGOD.LOGGER.warn("API call failed");
                        this.isSending = false;
                    }
                    GptActions.processResponse(raw);
                } else {
                    // convert local provider response into the OpenAI-like envelope expected by the rest of the plugin
                    String text = extractTextFromProviderResponse(raw);
                    var synthetic = new com.google.gson.JsonObject();
                    synthetic.addProperty("id", provider + "-" + System.currentTimeMillis());
                    synthetic.addProperty("object", "chat.completion");
                    synthetic.addProperty("created", (int) (System.currentTimeMillis() / 1000));
                    synthetic.addProperty("model", body.getModel().getName());
                    var choices = new com.google.gson.JsonArray();
                    var choice = new com.google.gson.JsonObject();
                    choice.addProperty("index", 0);
                    var message = new com.google.gson.JsonObject();
                    message.addProperty("role", "assistant");
                    message.addProperty("content", text);
                    choice.add("message", message);
                    choice.addProperty("finish_reason", "stop");
                    choices.add(choice);
                    synthetic.add("choices", choices);
                    String out = gson.create().toJson(synthetic);
                    GPTGOD.LOGGER.info("mapped " + provider + " response to OpenAI format: " + out);
                    GptActions.processResponse(out);
                }

                client.close();
                Bukkit.getScheduler().runTaskLater(JavaPlugin.getPlugin(GPTGOD.class), () -> {
                    this.isSending = false;
                }, 10);
            } catch (IOException e) {
                GPTGOD.LOGGER.error("There was an error making a request to GPT", e);
                this.isSending = false;
            }
        });
    }

    public void send(Map<String, GptFunction> functions) {
        CloseableHttpClient client = HttpClientBuilder.create().build();
        pool.execute(() -> {
            this.isSending = true;
            FileConfiguration config = JavaPlugin.getPlugin(GPTGOD.class).getConfig();
            String provider = providerName(config);

            try {
                String url = resolveChatUrl(config);
                StringEntity data;
                HttpPost post = new HttpPost(url);

                if(providerIsOpenAICompatible(provider)){
                    var payload = new com.google.gson.JsonObject();
                    payload.add("messages", gson.toJsonTree(body.getMessages()));
                    payload.addProperty("model", body.getModel().getName());
                    if(body.getTools() != null && body.getTools().length > 0){
                        payload.add("tools", gson.toJsonTree(body.getTools()));
                        if(body.getTool_choice() != null){
                            payload.add("tool_choice", gson.toJsonTree(body.getTool_choice()));
                        }
                    }
                    // NIM-specific extra_body
                    if(provider.equals("nim")){
                        String extra = config.getString("nimExtraBody", "{}");
                        try{
                            var extraJson = com.google.gson.JsonParser.parseString(extra).getAsJsonObject();
                            for(var entry : extraJson.entrySet()){
                                payload.add(entry.getKey(), entry.getValue());
                            }
                        } catch(Exception ignored){}
                    }
                    data = new StringEntity(payload.toString(), ContentType.APPLICATION_JSON);
                    String auth = resolveAuthHeader(config);
                    if(auth != null) post.setHeader(HttpHeaders.AUTHORIZATION, auth);
                } else {
                    // functions are not supported for non-OpenAI-compatible providers; fall back to prompt-only
                    var map = new java.util.HashMap<String, Object>();
                    map.put("model", body.getModel().getName());
                    map.put("prompt", messagesToPrompt());
                    data = new StringEntity(gson.create().toJson(map), ContentType.APPLICATION_JSON);
                }

                GPTGOD.LOGGER.info("POSTING (" + provider + ") " + gson.setPrettyPrinting().create().toJson(body));
                post.setEntity(data);
                GPTGOD.LOGGER.info("Making POST request to " + url);

                HttpResponse response = client.execute(post);
                String raw = new String(response.getEntity().getContent().readAllBytes());
                EntityUtils.consume(response.getEntity());

                if(providerIsOpenAICompatible(provider)){
                    GPTGOD.LOGGER.info("received response from " + provider + ": " + raw);
                    if (response.getStatusLine().getStatusCode() != 200) {
                        GPTGOD.LOGGER.warn("API call failed");
                        this.isSending = false;
                    }
                    GptActions.processResponse(raw, functions);
                } else {
                    String text = extractTextFromProviderResponse(raw);
                    var synthetic = new com.google.gson.JsonObject();
                    synthetic.addProperty("id", provider + "-" + System.currentTimeMillis());
                    synthetic.addProperty("object", "chat.completion");
                    synthetic.addProperty("created", (int) (System.currentTimeMillis() / 1000));
                    synthetic.addProperty("model", body.getModel().getName());
                    var choices = new com.google.gson.JsonArray();
                    var choice = new com.google.gson.JsonObject();
                    choice.addProperty("index", 0);
                    var message = new com.google.gson.JsonObject();
                    message.addProperty("role", "assistant");
                    message.addProperty("content", text);
                    choice.add("message", message);
                    choice.addProperty("finish_reason", "stop");
                    choices.add(choice);
                    synthetic.add("choices", choices);
                    String out = gson.create().toJson(synthetic);
                    GPTGOD.LOGGER.info("mapped " + provider + " response to OpenAI format: " + out);
                    GptActions.processResponse(out, functions);
                }

                client.close();
                Bukkit.getScheduler().runTaskLater(JavaPlugin.getPlugin(GPTGOD.class), () -> {
                    this.isSending = false;
                }, 20);
            } catch (IOException e) {
                GPTGOD.LOGGER.error("There was an error making a request to GPT", e);
                this.isSending = false;
            }
        });
    }

    public boolean isSending() {
        return isSending;
    }

    // DEBUG method
    public void checkRequestBody() {
        GPTGOD.LOGGER.info("POSTING " + gson.setPrettyPrinting().create().toJson(body));
    }
}
