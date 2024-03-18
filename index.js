import express from "express"
import bodyParser from "body-parser"
import cors from "cors"

import { LlamaCpp } from "@langchain/community/llms/llama_cpp";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";

import {LlamaModel, LlamaContext, LlamaChatSession} from "node-llama-cpp";

const modelPath = "gguf_model_path";


class ChainModel {
  constructor(templates, model) {
    this.templates = templates;
    this.model = model;
  }

  async chat(prompt, res) {
    const template = ChatPromptTemplate.fromTemplate(this.templates.chat);
    const parser = new StringOutputParser();

    const chain = template.pipe(this.model).pipe(parser);

    const stream = await chain.stream({
      question: prompt,
    })

    for await (const chunk of stream) {
      res.write(chunk);
    }
  }
}

const initChainModel = () => {
  const templates = {
    chat: "suppose you are an AI assistant. Please answer the question: {question}. Answer:"
  };
  
  const llamaCpp = new LlamaCpp({ modelPath });
  return new ChainModel(templates, llamaCpp)
}

class RawModel {
  constructor(context) {
    this.context = context;
  }

  async chat(prompt, res) {
    const session = new LlamaChatSession({context: this.context})
    await session.prompt(`suppose you are an AI assistant. Please answer the question: ${prompt}. Answer:`, {
      onToken: (tokens) => {
        res.write(this.context.decode(tokens))
      }
    })
  }
}

const initRawModel = () => {
  const m = new LlamaModel({ modelPath })
  const context = new LlamaContext({ model: m })
  return new RawModel(context)
}

const model = initChainModel()
// const model = initRawModel()


const app = express();
app.use(bodyParser.json())
app.use(cors())

const port = 8000;

app.post('/inference', async (req, res) => {
  const body = req.body;

  await model.chat(body.prompt, res)
  res.end()
})

app.listen(port, () => {
  console.log(`server is running on port ${port}`);
});
