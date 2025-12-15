import * as z from "zod";
import { createAgent, tool } from "langchain";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import dotenv from "dotenv";

dotenv.config();

const getWeather = tool(
  ({ city }) => `It's always sunny in ${city}!`,
  {
    name: "get_weather",
    description: "Get the weather for a given city",
    schema: z.object({
      city: z.string(),
    }),
  }
);

const model = new ChatGoogleGenerativeAI({
  model: "gemini-2.5-flash",
  apiKey: process.env.GOOGLE_API_KEY!,
});

const agent = createAgent({
  model, 
  tools: [getWeather],
});

const result = await agent.invoke({
  messages: [
    { role: "user", content: "What's the weather like in New York?" },
  ],
});

console.log(result);
