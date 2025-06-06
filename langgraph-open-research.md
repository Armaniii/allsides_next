Directory Structure:

└── ./
    ├── src
    │   └── open_deep_research
    │       ├── __init__.py
    │       ├── configuration.py
    │       ├── graph.ipynb
    │       ├── graph.py
    │       ├── prompts.py
    │       ├── state.py
    │       └── utils.py
    ├── .env.example
    ├── langgraph.json
    ├── LICENSE
    ├── pyproject.toml
    └── README.md



---
File: /src/open_deep_research/__init__.py
---

"""Planning, research, and report generation."""

__version__ = "0.0.10"


---
File: /src/open_deep_research/configuration.py
---

import os
from enum import Enum
from dataclasses import dataclass, fields
from typing import Any, Optional, Dict 

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from dataclasses import dataclass

DEFAULT_REPORT_STRUCTURE = """Use this structure to create a report on the user-provided topic:

1. Introduction (no research needed)
   - Brief overview of the topic area

2. Main Body Sections:
   - Each section should focus on a sub-topic of the user-provided topic
   
3. Conclusion
   - Aim for 1 structural element (either a list of table) that distills the main body sections 
   - Provide a concise summary of the report"""

class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    EXA = "exa"
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    LINKUP = "linkup"
    DUCKDUCKGO = "duckduckgo"
    GOOGLESEARCH = "googlesearch"

@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the chatbot."""
    report_structure: str = DEFAULT_REPORT_STRUCTURE # Defaults to the default report structure
    number_of_queries: int = 2 # Number of search queries to generate per iteration
    max_search_depth: int = 2 # Maximum number of reflection + search iterations
    planner_provider: str = "anthropic"  # Defaults to Anthropic as provider
    planner_model: str = "claude-3-7-sonnet-latest" # Defaults to claude-3-7-sonnet-latest
    writer_provider: str = "anthropic" # Defaults to Anthropic as provider
    writer_model: str = "claude-3-5-sonnet-latest" # Defaults to claude-3-5-sonnet-latest
    search_api: SearchAPI = SearchAPI.TAVILY # Default to TAVILY
    search_api_config: Optional[Dict[str, Any]] = None 

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})



---
File: /src/open_deep_research/graph.ipynb
---

{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m A new release of pip is available: \u001b[0m\u001b[31;49m23.2.1\u001b[0m\u001b[39;49m -> \u001b[0m\u001b[32;49m25.0.1\u001b[0m\n",
      "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m To update, run: \u001b[0m\u001b[32;49mpip install --upgrade pip\u001b[0m\n"
     ]
    }
   ],
   "source": [
    "! pip install -U -q open-deep-research"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.0.10\n"
     ]
    }
   ],
   "source": [
    "import open_deep_research   \n",
    "print(open_deep_research.__version__) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "from IPython.display import Image, display\n",
    "from langgraph.types import Command\n",
    "from langgraph.checkpoint.memory import MemorySaver\n",
    "from open_deep_research.graph import builder"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVB",
      "text/plain": [
       "<IPython.core.display.Image object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "memory = MemorySaver()\n",
    "graph = builder.compile(checkpointer=memory)\n",
    "display(Image(graph.get_graph(xray=1).draw_mermaid_png()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os, getpass\n",
    "\n",
    "def _set_env(var: str):\n",
    "    if not os.environ.get(var):\n",
    "        os.environ[var] = getpass.getpass(f\"{var}: \")\n",
    "\n",
    "# Set the API keys used for any model or search tool selections below, such as:\n",
    "_set_env(\"OPENAI_API_KEY\")\n",
    "_set_env(\"ANTHROPIC_API_KEY\")\n",
    "_set_env(\"TAVILY_API_KEY\")\n",
    "_set_env(\"GROQ_API_KEY\")\n",
    "_set_env(\"PERPLEXITY_API_KEY\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/markdown": [
       "Please provide feedback on the following report plan. \n",
       "                        \n",
       "\n",
       "Section: Introduction\n",
       "Description: Provides an overall introduction to the AI inference market and sets the context for the focus on Fireworks, Together.ai, and Groq.\n",
       "Research needed: No\n",
       "\n",
       "\n",
       "Section: AI Inference Market Overview\n",
       "Description: Discusses the general landscape of the AI inference market, including key trends, technologies, and challenges facing LLM API providers.\n",
       "Research needed: Yes\n",
       "\n",
       "\n",
       "Section: Fireworks AI Overview and Market Position\n",
       "Description: Explores Fireworks AI’s technology, including its proprietary FireAttention inference engine, performance metrics, and data privacy compliance, as well as its role in contemporary AI development platforms.\n",
       "Research needed: Yes\n",
       "\n",
       "\n",
       "Section: Together.ai Overview and Market Position\n",
       "Description: Covers Together.ai’s platform features, competitive advantages, and how it compares to other AI inference providers. Highlights its market positioning in delivering high-quality open-source models.\n",
       "Research needed: Yes\n",
       "\n",
       "\n",
       "Section: Groq Overview and Market Position\n",
       "Description: Examines Groq’s innovative offering with its Tensor Streaming Processor, focusing on high-performance, low-latency inference, and its tailored approach to AI accelerators for enterprise applications.\n",
       "Research needed: Yes\n",
       "\n",
       "\n",
       "Section: Comparative Analysis of Fireworks AI, Together.ai, and Groq\n",
       "Description: Provides a direct comparison between the three platforms in terms of performance, API integration, latency, scalability, and suitability for various AI use cases.\n",
       "Research needed: Yes\n",
       "\n",
       "\n",
       "Section: Conclusion\n",
       "Description: Summarizes the main findings from the report, distilling key takeaways into a concise list or table that emphasizes the strengths and unique market positions of each platform.\n",
       "Research needed: No\n",
       "\n",
       "\n",
       "                        \n",
       "Does the report plan meet your needs?\n",
       "Pass 'true' to approve the report plan.\n",
       "Or, provide feedback to regenerate the report plan:"
      ],
      "text/plain": [
       "<IPython.core.display.Markdown object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import uuid \n",
    "from IPython.display import Markdown\n",
    "\n",
    "REPORT_STRUCTURE = \"\"\"Use this structure to create a report on the user-provided topic:\n",
    "\n",
    "1. Introduction (no research needed)\n",
    "   - Brief overview of the topic area\n",
    "\n",
    "2. Main Body Sections:\n",
    "   - Each section should focus on a sub-topic of the user-provided topic\n",
    "   \n",
    "3. Conclusion\n",
    "   - Aim for 1 structural element (either a list of table) that distills the main body sections \n",
    "   - Provide a concise summary of the report\"\"\"\n",
    "\n",
    "# Claude 3.7 Sonnet for planning with perplexity search\n",
    "thread = {\"configurable\": {\"thread_id\": str(uuid.uuid4()),\n",
    "                           \"search_api\": \"perplexity\",\n",
    "                           \"planner_provider\": \"anthropic\",\n",
    "                           \"planner_model\": \"claude-3-7-sonnet-latest\",\n",
    "                           \"writer_provider\": \"anthropic\",\n",
    "                           \"writer_model\": \"claude-3-5-sonnet-latest\",\n",
    "                           \"max_search_depth\": 2,\n",
    "                           \"report_structure\": REPORT_STRUCTURE,\n",
    "                           }}\n",
    "\n",
    "# DeepSeek-R1-Distill-Llama-70B for planning and llama-3.3-70b-versatile for writing\n",
    "thread = {\"configurable\": {\"thread_id\": str(uuid.uuid4()),\n",
    "                           \"search_api\": \"tavily\",\n",
    "                           \"planner_provider\": \"groq\",\n",
    "                           \"planner_model\": \"deepseek-r1-distill-llama-70b\",\n",
    "                           \"writer_provider\": \"groq\",\n",
    "                           \"writer_model\": \"llama-3.3-70b-versatile\",\n",
    "                           \"report_structure\": REPORT_STRUCTURE,\n",
    "                           \"max_search_depth\": 1,}\n",
    "                           }\n",
    "\n",
    "# Fast config (less search depth) with o3-mini for planning and Claude 3.5 Sonnet for writing\n",
    "thread = {\"configurable\": {\"thread_id\": str(uuid.uuid4()),\n",
    "                           \"search_api\": \"tavily\",\n",
    "                           \"planner_provider\": \"openai\",\n",
    "                           \"planner_model\": \"o3-mini\",\n",
    "                           \"writer_provider\": \"anthropic\",\n",
    "                           \"writer_model\": \"claude-3-5-sonnet-latest\",\n",
    "                           \"max_search_depth\": 1,\n",
    "                           \"report_structure\": REPORT_STRUCTURE,\n",
    "                           }}\n",
    "\n",
    "# Create a topic\n",
    "topic = \"Overview of the AI inference market with focus on Fireworks, Together.ai, Groq\"\n",
    "\n",
    "# Run the graph until the interruption\n",
    "async for event in graph.astream({\"topic\":topic,}, thread, stream_mode=\"updates\"):\n",
    "    if '__interrupt__' in event:\n",
    "        interrupt_value = event['__interrupt__'][0].value\n",
    "        display(Markdown(interrupt_value))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/markdown": [
       "Please provide feedback on the following report plan. \n",
       "                        \n",
       "\n",
       "Section: Introduction\n",
       "Description: Provides a brief overview of the AI inference market, its relevance, and introduces the three key players: Fireworks, Together.ai, and Groq.\n",
       "Research needed: No\n",
       "\n",
       "\n",
       "Section: Fireworks AI Overview\n",
       "Description: Covers Fireworks AI’s offerings in the inference market, including key performance metrics, product features, model quality, speed, and latency. Emphasizes revenue estimates (ARR) as part of its market positioning.\n",
       "Research needed: Yes\n",
       "\n",
       "\n",
       "Section: Together.ai Overview\n",
       "Description: Explores Together.ai’s role within the AI inference space. Discusses product features, integration details, user reviews, and performance metrics along with ARR revenue estimates.\n",
       "Research needed: Yes\n",
       "\n",
       "\n",
       "Section: Groq AI Overview\n",
       "Description: Examines Groq’s AI inference solutions with a focus on benchmark performance, inference speed, and cost efficiency. Includes an analysis of ARR revenue and how its technical performance shapes its market position.\n",
       "Research needed: Yes\n",
       "\n",
       "\n",
       "Section: Market Comparison and Trends\n",
       "Description: Provides a comparative analysis highlighting the strengths, weaknesses, and differentiating factors of Fireworks, Together.ai, and Groq. Discusses trends like cost, performance metrics, and revenue (ARR) estimates.\n",
       "Research needed: Yes\n",
       "\n",
       "\n",
       "Section: Conclusion\n",
       "Description: Summarizes key findings from the individual sections and the comparative analysis. Offers a concise distillation of the market landscape using a summary table or list to highlight major points including ARR insights.\n",
       "Research needed: No\n",
       "\n",
       "\n",
       "                        \n",
       "Does the report plan meet your needs?\n",
       "Pass 'true' to approve the report plan.\n",
       "Or, provide feedback to regenerate the report plan:"
      ],
      "text/plain": [
       "<IPython.core.display.Markdown object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Pass feedback to update the report plan  \n",
    "async for event in graph.astream(Command(resume=\"Include individuals sections for Together.ai, Groq, and Fireworks with revenue estimates (ARR)\"), thread, stream_mode=\"updates\"):\n",
    "    if '__interrupt__' in event:\n",
    "        interrupt_value = event['__interrupt__'][0].value\n",
    "        display(Markdown(interrupt_value))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'human_feedback': None}\n",
      "\n",
      "\n",
      "{'build_section_with_web_research': {'completed_sections': [Section(name='Groq AI Overview', description='Examines Groq’s AI inference solutions with a focus on benchmark performance, inference speed, and cost efficiency. Includes an analysis of ARR revenue and how its technical performance shapes its market position.', research=True, content=\"## Groq AI Overview\\n\\nGroq has emerged as a significant player in the AI inference market with its innovative Language Processing Unit (LPU) technology. Independent benchmarks from ArtificialAnalysis.ai demonstrate Groq's impressive performance, with their Llama 2 Chat (70B) API achieving 241 tokens per second - more than double the speed of competing providers [1]. Their total response time for 100 output tokens is just 0.8 seconds, positioning them as a leader in inference speed [2].\\n\\nThe company's core technology, the Tensor Streaming Processor (TSP), delivers 500-700 tokens per second on large language models, representing a 5-10x improvement over Nvidia's latest data center GPUs [3]. This performance advantage has positioned Groq as a compelling alternative in the inference market, particularly for startups seeking cost-effective solutions [4].\\n\\nWith approximately 300 employees, 60% of whom are software engineers, Groq has evolved beyond hardware to become a comprehensive AI solutions provider [5]. The company is strategically positioned to capture a share of the growing inference chip market, which is projected to reach $48 billion by 2027 [6].\\n\\n### Sources\\n[1] https://groq.com/artificialanalysis-ai-llm-benchmark-doubles-axis-to-fit-new-groq-lpu-inference-engine-performance-results/\\n[2] https://groq.com/inference/\\n[3] https://sacra.com/c/groq/\\n[4] https://venturebeat.com/ai/ai-chip-race-groq-ceo-takes-on-nvidia-claims-most-startups-will-use-speedy-lpus-by-end-of-2024/\\n[5] https://www.businessinsider.com/groq-nvidia-software-advantage-cuda-moat-challenge-inference-2024-12?op=1\\n[6] https://generativeaitech.substack.com/p/groq-and-its-impact-on-ai-inference\")]}}\n",
      "\n",
      "\n",
      "{'build_section_with_web_research': {'completed_sections': [Section(name='Market Comparison and Trends', description='Provides a comparative analysis highlighting the strengths, weaknesses, and differentiating factors of Fireworks, Together.ai, and Groq. Discusses trends like cost, performance metrics, and revenue (ARR) estimates.', research=True, content=\"## Market Comparison and Trends\\n\\nThe AI inference market shows distinct positioning among key players Fireworks, Together.ai, and Groq. Fireworks AI differentiates itself through superior speed, leveraging its proprietary FireAttention inference engine for text, image, and audio processing, while maintaining HIPAA and SOC2 compliance for data privacy [1]. Together.ai has demonstrated remarkable growth, reaching an estimated $130M in annualized recurring revenue (ARR) in 2024, representing a 400% year-over-year increase [2].\\n\\nBoth Fireworks and Together.ai focus on providing high-quality open-source models that can compete with proprietary alternatives [1]. Pricing structures vary across providers, with model costs ranging from $0.20 to $3.00 per million tokens for different model sizes and capabilities [3]. Together.ai's business model benefits from GPU price commoditization, allowing them to maintain competitive token pricing while focusing on developer experience and reliable inference across diverse open-source models [2].\\n\\nGroq distinguishes itself by offering quick access to various open-source AI models from major providers like Google, Meta, OpenAI, and Mistral through the OpenRouter API platform [4]. The market shows a trend toward comprehensive platform offerings, with providers competing on factors such as model variety, inference speed, and specialized features for different use cases.\\n\\n### Sources\\n[1] Top 10 AI Inference Platforms in 2025: Comparing LLM API Providers: https://www.helicone.ai/blog/llm-api-providers\\n[2] Together AI revenue, valuation & growth rate | Sacra: https://sacra.com/c/together-ai/\\n[3] Models Leaderboard : Comparison of AI Models & API Providers - Groq AI: https://groq-ai.com/models/\\n[4] Pricing: Compare Groq API Pricing With Other API Providers: https://groq-ai.com/pricing/\")]}}\n",
      "\n",
      "\n",
      "{'build_section_with_web_research': {'completed_sections': [Section(name='Together.ai Overview', description='Explores Together.ai’s role within the AI inference space. Discusses product features, integration details, user reviews, and performance metrics along with ARR revenue estimates.', research=True, content=\"## Together.ai Overview\\n\\nTogether.ai is a comprehensive AI acceleration platform that has quickly emerged as a significant player in the AI inference space since its founding in 2022 [1]. The platform enables developers to access over 200 AI models, offering high-performance inference capabilities with optimized infrastructure [2].\\n\\nThe company's Inference Engine, built on NVIDIA Tensor Core GPUs, delivers impressive performance metrics, achieving 117 tokens per second on Llama-2-70B-Chat models [3]. Their model offerings include high-quality options like Llama 3.3 70B Turbo and Llama 3.1 405B Turbo, with some models achieving sub-100ms latency [4].\\n\\nTogether.ai has demonstrated remarkable growth, with estimates suggesting $130M in annualized recurring revenue (ARR) in 2024, representing a 400% year-over-year increase [5]. The platform has received positive user feedback, maintaining a 4.8/5.0 rating based on 162 user reviews [6].\\n\\nThe company differentiates itself through competitive pricing and technical innovations, incorporating advanced features like token caching, load balancing, and model quantization [2]. Users particularly praise the platform's straightforward API, reliable performance, and competitive pricing compared to alternatives [7].\\n\\n### Sources\\n[1] https://siliconvalleyjournals.com/company/together-ai/\\n[2] https://www.keywordsai.co/blog/top-10-llm-api-providers\\n[3] https://www.together.ai/blog/together-inference-engine-v1\\n[4] https://artificialanalysis.ai/providers/togetherai\\n[5] https://sacra.com/c/together-ai/\\n[6] https://www.featuredcustomers.com/vendor/together-ai\\n[7] https://www.trustpilot.com/review/together.ai\")]}}\n",
      "\n",
      "\n",
      "{'build_section_with_web_research': {'completed_sections': [Section(name='Fireworks AI Overview', description='Covers Fireworks AI’s offerings in the inference market, including key performance metrics, product features, model quality, speed, and latency. Emphasizes revenue estimates (ARR) as part of its market positioning.', research=True, content='## Fireworks AI Overview\\n\\nFireworks AI has established itself as a significant player in the AI inference market, offering a comprehensive platform for deploying and managing AI models. The company provides access to various high-performance models, including Llama 3.1 405B and Qwen2.5 Coder 32B, which represent their highest quality offerings [1]. Their platform demonstrates impressive performance metrics, with some models achieving speeds up to 300 tokens per second for serverless deployments [2].\\n\\nThe company has gained substantial market traction, recently securing $52 million in Series B funding that values the company at $552 million [3]. With an annual revenue of $6 million and a team of 60 employees, Fireworks AI serves notable clients including DoorDash, Quora, and Upwork [4, 5].\\n\\nTheir platform differentiates itself through specialized features like FP8 quantization for large models and continuous batching capabilities [6]. Fireworks claims to offer approximately 3x faster speeds compared to competitors like Hugging Face TGI when using the same GPU configuration [2]. The company focuses on providing smaller, production-grade models that can be deployed privately and securely, rather than generic mega models [5].\\n\\n### Sources\\n[1] Fireworks: Models Intelligence, Performance & Price: https://artificialanalysis.ai/providers/fireworks\\n[2] Fireworks Platform Spring 2024 Updates: https://fireworks.ai/blog/spring-update-faster-models-dedicated-deployments-postpaid-pricing\\n[3] Fireworks AI Raises $52M in Series B Funding: https://siliconvalleyjournals.com/fireworks-ai-raises-52m-in-series-b-funding-to-expand-genai-inference-platform/\\n[4] Fireworks AI: Contact Details, Revenue, Funding: https://siliconvalleyjournals.com/company/fireworks-ai/\\n[5] Fireworks AI Valued at $552 Million After New Funding Round: https://www.pymnts.com/news/investment-tracker/2024/fireworks-ai-valued-552-million-dollars-after-new-funding-round/\\n[6] Inference performance - Fireworks AI Docs: https://docs.fireworks.ai/faq/models/inference/performance')]}}\n",
      "\n",
      "\n",
      "{'gather_completed_sections': {'report_sections_from_research': \"\\n============================================================\\nSection 1: Fireworks AI Overview\\n============================================================\\nDescription:\\nCovers Fireworks AI’s offerings in the inference market, including key performance metrics, product features, model quality, speed, and latency. Emphasizes revenue estimates (ARR) as part of its market positioning.\\nRequires Research: \\nTrue\\n\\nContent:\\n## Fireworks AI Overview\\n\\nFireworks AI has established itself as a significant player in the AI inference market, offering a comprehensive platform for deploying and managing AI models. The company provides access to various high-performance models, including Llama 3.1 405B and Qwen2.5 Coder 32B, which represent their highest quality offerings [1]. Their platform demonstrates impressive performance metrics, with some models achieving speeds up to 300 tokens per second for serverless deployments [2].\\n\\nThe company has gained substantial market traction, recently securing $52 million in Series B funding that values the company at $552 million [3]. With an annual revenue of $6 million and a team of 60 employees, Fireworks AI serves notable clients including DoorDash, Quora, and Upwork [4, 5].\\n\\nTheir platform differentiates itself through specialized features like FP8 quantization for large models and continuous batching capabilities [6]. Fireworks claims to offer approximately 3x faster speeds compared to competitors like Hugging Face TGI when using the same GPU configuration [2]. The company focuses on providing smaller, production-grade models that can be deployed privately and securely, rather than generic mega models [5].\\n\\n### Sources\\n[1] Fireworks: Models Intelligence, Performance & Price: https://artificialanalysis.ai/providers/fireworks\\n[2] Fireworks Platform Spring 2024 Updates: https://fireworks.ai/blog/spring-update-faster-models-dedicated-deployments-postpaid-pricing\\n[3] Fireworks AI Raises $52M in Series B Funding: https://siliconvalleyjournals.com/fireworks-ai-raises-52m-in-series-b-funding-to-expand-genai-inference-platform/\\n[4] Fireworks AI: Contact Details, Revenue, Funding: https://siliconvalleyjournals.com/company/fireworks-ai/\\n[5] Fireworks AI Valued at $552 Million After New Funding Round: https://www.pymnts.com/news/investment-tracker/2024/fireworks-ai-valued-552-million-dollars-after-new-funding-round/\\n[6] Inference performance - Fireworks AI Docs: https://docs.fireworks.ai/faq/models/inference/performance\\n\\n\\n============================================================\\nSection 2: Together.ai Overview\\n============================================================\\nDescription:\\nExplores Together.ai’s role within the AI inference space. Discusses product features, integration details, user reviews, and performance metrics along with ARR revenue estimates.\\nRequires Research: \\nTrue\\n\\nContent:\\n## Together.ai Overview\\n\\nTogether.ai is a comprehensive AI acceleration platform that has quickly emerged as a significant player in the AI inference space since its founding in 2022 [1]. The platform enables developers to access over 200 AI models, offering high-performance inference capabilities with optimized infrastructure [2].\\n\\nThe company's Inference Engine, built on NVIDIA Tensor Core GPUs, delivers impressive performance metrics, achieving 117 tokens per second on Llama-2-70B-Chat models [3]. Their model offerings include high-quality options like Llama 3.3 70B Turbo and Llama 3.1 405B Turbo, with some models achieving sub-100ms latency [4].\\n\\nTogether.ai has demonstrated remarkable growth, with estimates suggesting $130M in annualized recurring revenue (ARR) in 2024, representing a 400% year-over-year increase [5]. The platform has received positive user feedback, maintaining a 4.8/5.0 rating based on 162 user reviews [6].\\n\\nThe company differentiates itself through competitive pricing and technical innovations, incorporating advanced features like token caching, load balancing, and model quantization [2]. Users particularly praise the platform's straightforward API, reliable performance, and competitive pricing compared to alternatives [7].\\n\\n### Sources\\n[1] https://siliconvalleyjournals.com/company/together-ai/\\n[2] https://www.keywordsai.co/blog/top-10-llm-api-providers\\n[3] https://www.together.ai/blog/together-inference-engine-v1\\n[4] https://artificialanalysis.ai/providers/togetherai\\n[5] https://sacra.com/c/together-ai/\\n[6] https://www.featuredcustomers.com/vendor/together-ai\\n[7] https://www.trustpilot.com/review/together.ai\\n\\n\\n============================================================\\nSection 3: Groq AI Overview\\n============================================================\\nDescription:\\nExamines Groq’s AI inference solutions with a focus on benchmark performance, inference speed, and cost efficiency. Includes an analysis of ARR revenue and how its technical performance shapes its market position.\\nRequires Research: \\nTrue\\n\\nContent:\\n## Groq AI Overview\\n\\nGroq has emerged as a significant player in the AI inference market with its innovative Language Processing Unit (LPU) technology. Independent benchmarks from ArtificialAnalysis.ai demonstrate Groq's impressive performance, with their Llama 2 Chat (70B) API achieving 241 tokens per second - more than double the speed of competing providers [1]. Their total response time for 100 output tokens is just 0.8 seconds, positioning them as a leader in inference speed [2].\\n\\nThe company's core technology, the Tensor Streaming Processor (TSP), delivers 500-700 tokens per second on large language models, representing a 5-10x improvement over Nvidia's latest data center GPUs [3]. This performance advantage has positioned Groq as a compelling alternative in the inference market, particularly for startups seeking cost-effective solutions [4].\\n\\nWith approximately 300 employees, 60% of whom are software engineers, Groq has evolved beyond hardware to become a comprehensive AI solutions provider [5]. The company is strategically positioned to capture a share of the growing inference chip market, which is projected to reach $48 billion by 2027 [6].\\n\\n### Sources\\n[1] https://groq.com/artificialanalysis-ai-llm-benchmark-doubles-axis-to-fit-new-groq-lpu-inference-engine-performance-results/\\n[2] https://groq.com/inference/\\n[3] https://sacra.com/c/groq/\\n[4] https://venturebeat.com/ai/ai-chip-race-groq-ceo-takes-on-nvidia-claims-most-startups-will-use-speedy-lpus-by-end-of-2024/\\n[5] https://www.businessinsider.com/groq-nvidia-software-advantage-cuda-moat-challenge-inference-2024-12?op=1\\n[6] https://generativeaitech.substack.com/p/groq-and-its-impact-on-ai-inference\\n\\n\\n============================================================\\nSection 4: Market Comparison and Trends\\n============================================================\\nDescription:\\nProvides a comparative analysis highlighting the strengths, weaknesses, and differentiating factors of Fireworks, Together.ai, and Groq. Discusses trends like cost, performance metrics, and revenue (ARR) estimates.\\nRequires Research: \\nTrue\\n\\nContent:\\n## Market Comparison and Trends\\n\\nThe AI inference market shows distinct positioning among key players Fireworks, Together.ai, and Groq. Fireworks AI differentiates itself through superior speed, leveraging its proprietary FireAttention inference engine for text, image, and audio processing, while maintaining HIPAA and SOC2 compliance for data privacy [1]. Together.ai has demonstrated remarkable growth, reaching an estimated $130M in annualized recurring revenue (ARR) in 2024, representing a 400% year-over-year increase [2].\\n\\nBoth Fireworks and Together.ai focus on providing high-quality open-source models that can compete with proprietary alternatives [1]. Pricing structures vary across providers, with model costs ranging from $0.20 to $3.00 per million tokens for different model sizes and capabilities [3]. Together.ai's business model benefits from GPU price commoditization, allowing them to maintain competitive token pricing while focusing on developer experience and reliable inference across diverse open-source models [2].\\n\\nGroq distinguishes itself by offering quick access to various open-source AI models from major providers like Google, Meta, OpenAI, and Mistral through the OpenRouter API platform [4]. The market shows a trend toward comprehensive platform offerings, with providers competing on factors such as model variety, inference speed, and specialized features for different use cases.\\n\\n### Sources\\n[1] Top 10 AI Inference Platforms in 2025: Comparing LLM API Providers: https://www.helicone.ai/blog/llm-api-providers\\n[2] Together AI revenue, valuation & growth rate | Sacra: https://sacra.com/c/together-ai/\\n[3] Models Leaderboard : Comparison of AI Models & API Providers - Groq AI: https://groq-ai.com/models/\\n[4] Pricing: Compare Groq API Pricing With Other API Providers: https://groq-ai.com/pricing/\\n\\n\"}}\n",
      "\n",
      "\n",
      "{'write_final_sections': {'completed_sections': [Section(name='Introduction', description='Provides a brief overview of the AI inference market, its relevance, and introduces the three key players: Fireworks, Together.ai, and Groq.', research=False, content=\"# AI Inference Market: Analysis of Fireworks, Together.ai, and Groq\\n\\nThe AI inference market is experiencing rapid evolution as companies compete to provide faster, more efficient solutions for deploying and managing AI models. This analysis examines three key players reshaping the landscape: Fireworks, Together.ai, and Groq. Each brings distinct advantages - Fireworks with its specialized features and security focus, Together.ai with its comprehensive platform and impressive growth trajectory, and Groq with its groundbreaking Language Processing Unit technology. As organizations increasingly rely on AI inference capabilities, understanding these providers' strengths and market positions becomes crucial for making informed deployment decisions.\")]}}\n",
      "\n",
      "\n",
      "{'write_final_sections': {'completed_sections': [Section(name='Conclusion', description='Summarizes key findings from the individual sections and the comparative analysis. Offers a concise distillation of the market landscape using a summary table or list to highlight major points including ARR insights.', research=False, content='## Conclusion\\n\\nThe AI inference market shows clear differentiation among key players, with each provider carving out unique value propositions. Together.ai leads in revenue growth with $130M ARR, while Fireworks AI demonstrates strong potential with its recent $552M valuation. Groq distinguishes itself through superior inference speeds, achieving up to 700 tokens per second with its LPU technology.\\n\\n| Metric | Fireworks | Together.ai | Groq |\\n|--------|-----------|-------------|------|\\n| ARR | $6M | $130M | Not disclosed |\\n| Speed (tokens/sec) | Up to 300 | Up to 117 | 500-700 |\\n| Key Differentiator | FP8 quantization | 200+ model access | LPU technology |\\n| Notable Feature | 3x faster than competitors | Token caching | Sub-second response |\\n| Target Market | Enterprise security | Developer platforms | Startup solutions |\\n\\nThe market trends indicate a shift toward comprehensive platform offerings that balance speed, cost-efficiency, and model variety. As the inference chip market approaches $48B by 2027, providers focusing on specialized hardware solutions and optimized infrastructure will likely gain competitive advantages. Success will depend on maintaining the delicate balance between performance, pricing, and platform features while addressing growing enterprise demands for security and reliability.')]}}\n",
      "\n",
      "\n",
      "{'compile_final_report': {'final_report': \"# AI Inference Market: Analysis of Fireworks, Together.ai, and Groq\\n\\nThe AI inference market is experiencing rapid evolution as companies compete to provide faster, more efficient solutions for deploying and managing AI models. This analysis examines three key players reshaping the landscape: Fireworks, Together.ai, and Groq. Each brings distinct advantages - Fireworks with its specialized features and security focus, Together.ai with its comprehensive platform and impressive growth trajectory, and Groq with its groundbreaking Language Processing Unit technology. As organizations increasingly rely on AI inference capabilities, understanding these providers' strengths and market positions becomes crucial for making informed deployment decisions.\\n\\n## Fireworks AI Overview\\n\\nFireworks AI has established itself as a significant player in the AI inference market, offering a comprehensive platform for deploying and managing AI models. The company provides access to various high-performance models, including Llama 3.1 405B and Qwen2.5 Coder 32B, which represent their highest quality offerings [1]. Their platform demonstrates impressive performance metrics, with some models achieving speeds up to 300 tokens per second for serverless deployments [2].\\n\\nThe company has gained substantial market traction, recently securing $52 million in Series B funding that values the company at $552 million [3]. With an annual revenue of $6 million and a team of 60 employees, Fireworks AI serves notable clients including DoorDash, Quora, and Upwork [4, 5].\\n\\nTheir platform differentiates itself through specialized features like FP8 quantization for large models and continuous batching capabilities [6]. Fireworks claims to offer approximately 3x faster speeds compared to competitors like Hugging Face TGI when using the same GPU configuration [2]. The company focuses on providing smaller, production-grade models that can be deployed privately and securely, rather than generic mega models [5].\\n\\n### Sources\\n[1] Fireworks: Models Intelligence, Performance & Price: https://artificialanalysis.ai/providers/fireworks\\n[2] Fireworks Platform Spring 2024 Updates: https://fireworks.ai/blog/spring-update-faster-models-dedicated-deployments-postpaid-pricing\\n[3] Fireworks AI Raises $52M in Series B Funding: https://siliconvalleyjournals.com/fireworks-ai-raises-52m-in-series-b-funding-to-expand-genai-inference-platform/\\n[4] Fireworks AI: Contact Details, Revenue, Funding: https://siliconvalleyjournals.com/company/fireworks-ai/\\n[5] Fireworks AI Valued at $552 Million After New Funding Round: https://www.pymnts.com/news/investment-tracker/2024/fireworks-ai-valued-552-million-dollars-after-new-funding-round/\\n[6] Inference performance - Fireworks AI Docs: https://docs.fireworks.ai/faq/models/inference/performance\\n\\n## Together.ai Overview\\n\\nTogether.ai is a comprehensive AI acceleration platform that has quickly emerged as a significant player in the AI inference space since its founding in 2022 [1]. The platform enables developers to access over 200 AI models, offering high-performance inference capabilities with optimized infrastructure [2].\\n\\nThe company's Inference Engine, built on NVIDIA Tensor Core GPUs, delivers impressive performance metrics, achieving 117 tokens per second on Llama-2-70B-Chat models [3]. Their model offerings include high-quality options like Llama 3.3 70B Turbo and Llama 3.1 405B Turbo, with some models achieving sub-100ms latency [4].\\n\\nTogether.ai has demonstrated remarkable growth, with estimates suggesting $130M in annualized recurring revenue (ARR) in 2024, representing a 400% year-over-year increase [5]. The platform has received positive user feedback, maintaining a 4.8/5.0 rating based on 162 user reviews [6].\\n\\nThe company differentiates itself through competitive pricing and technical innovations, incorporating advanced features like token caching, load balancing, and model quantization [2]. Users particularly praise the platform's straightforward API, reliable performance, and competitive pricing compared to alternatives [7].\\n\\n### Sources\\n[1] https://siliconvalleyjournals.com/company/together-ai/\\n[2] https://www.keywordsai.co/blog/top-10-llm-api-providers\\n[3] https://www.together.ai/blog/together-inference-engine-v1\\n[4] https://artificialanalysis.ai/providers/togetherai\\n[5] https://sacra.com/c/together-ai/\\n[6] https://www.featuredcustomers.com/vendor/together-ai\\n[7] https://www.trustpilot.com/review/together.ai\\n\\n## Groq AI Overview\\n\\nGroq has emerged as a significant player in the AI inference market with its innovative Language Processing Unit (LPU) technology. Independent benchmarks from ArtificialAnalysis.ai demonstrate Groq's impressive performance, with their Llama 2 Chat (70B) API achieving 241 tokens per second - more than double the speed of competing providers [1]. Their total response time for 100 output tokens is just 0.8 seconds, positioning them as a leader in inference speed [2].\\n\\nThe company's core technology, the Tensor Streaming Processor (TSP), delivers 500-700 tokens per second on large language models, representing a 5-10x improvement over Nvidia's latest data center GPUs [3]. This performance advantage has positioned Groq as a compelling alternative in the inference market, particularly for startups seeking cost-effective solutions [4].\\n\\nWith approximately 300 employees, 60% of whom are software engineers, Groq has evolved beyond hardware to become a comprehensive AI solutions provider [5]. The company is strategically positioned to capture a share of the growing inference chip market, which is projected to reach $48 billion by 2027 [6].\\n\\n### Sources\\n[1] https://groq.com/artificialanalysis-ai-llm-benchmark-doubles-axis-to-fit-new-groq-lpu-inference-engine-performance-results/\\n[2] https://groq.com/inference/\\n[3] https://sacra.com/c/groq/\\n[4] https://venturebeat.com/ai/ai-chip-race-groq-ceo-takes-on-nvidia-claims-most-startups-will-use-speedy-lpus-by-end-of-2024/\\n[5] https://www.businessinsider.com/groq-nvidia-software-advantage-cuda-moat-challenge-inference-2024-12?op=1\\n[6] https://generativeaitech.substack.com/p/groq-and-its-impact-on-ai-inference\\n\\n## Market Comparison and Trends\\n\\nThe AI inference market shows distinct positioning among key players Fireworks, Together.ai, and Groq. Fireworks AI differentiates itself through superior speed, leveraging its proprietary FireAttention inference engine for text, image, and audio processing, while maintaining HIPAA and SOC2 compliance for data privacy [1]. Together.ai has demonstrated remarkable growth, reaching an estimated $130M in annualized recurring revenue (ARR) in 2024, representing a 400% year-over-year increase [2].\\n\\nBoth Fireworks and Together.ai focus on providing high-quality open-source models that can compete with proprietary alternatives [1]. Pricing structures vary across providers, with model costs ranging from $0.20 to $3.00 per million tokens for different model sizes and capabilities [3]. Together.ai's business model benefits from GPU price commoditization, allowing them to maintain competitive token pricing while focusing on developer experience and reliable inference across diverse open-source models [2].\\n\\nGroq distinguishes itself by offering quick access to various open-source AI models from major providers like Google, Meta, OpenAI, and Mistral through the OpenRouter API platform [4]. The market shows a trend toward comprehensive platform offerings, with providers competing on factors such as model variety, inference speed, and specialized features for different use cases.\\n\\n### Sources\\n[1] Top 10 AI Inference Platforms in 2025: Comparing LLM API Providers: https://www.helicone.ai/blog/llm-api-providers\\n[2] Together AI revenue, valuation & growth rate | Sacra: https://sacra.com/c/together-ai/\\n[3] Models Leaderboard : Comparison of AI Models & API Providers - Groq AI: https://groq-ai.com/models/\\n[4] Pricing: Compare Groq API Pricing With Other API Providers: https://groq-ai.com/pricing/\\n\\n## Conclusion\\n\\nThe AI inference market shows clear differentiation among key players, with each provider carving out unique value propositions. Together.ai leads in revenue growth with $130M ARR, while Fireworks AI demonstrates strong potential with its recent $552M valuation. Groq distinguishes itself through superior inference speeds, achieving up to 700 tokens per second with its LPU technology.\\n\\n| Metric | Fireworks | Together.ai | Groq |\\n|--------|-----------|-------------|------|\\n| ARR | $6M | $130M | Not disclosed |\\n| Speed (tokens/sec) | Up to 300 | Up to 117 | 500-700 |\\n| Key Differentiator | FP8 quantization | 200+ model access | LPU technology |\\n| Notable Feature | 3x faster than competitors | Token caching | Sub-second response |\\n| Target Market | Enterprise security | Developer platforms | Startup solutions |\\n\\nThe market trends indicate a shift toward comprehensive platform offerings that balance speed, cost-efficiency, and model variety. As the inference chip market approaches $48B by 2027, providers focusing on specialized hardware solutions and optimized infrastructure will likely gain competitive advantages. Success will depend on maintaining the delicate balance between performance, pricing, and platform features while addressing growing enterprise demands for security and reliability.\"}}\n",
      "\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Pass True to approve the report plan \n",
    "async for event in graph.astream(Command(resume=True), thread, stream_mode=\"updates\"):\n",
    "    print(event)\n",
    "    print(\"\\n\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/markdown": [
       "# AI Inference Market: Analysis of Fireworks, Together.ai, and Groq\n",
       "\n",
       "The AI inference market is experiencing rapid evolution as companies compete to provide faster, more efficient solutions for deploying and managing AI models. This analysis examines three key players reshaping the landscape: Fireworks, Together.ai, and Groq. Each brings distinct advantages - Fireworks with its specialized features and security focus, Together.ai with its comprehensive platform and impressive growth trajectory, and Groq with its groundbreaking Language Processing Unit technology. As organizations increasingly rely on AI inference capabilities, understanding these providers' strengths and market positions becomes crucial for making informed deployment decisions.\n",
       "\n",
       "## Fireworks AI Overview\n",
       "\n",
       "Fireworks AI has established itself as a significant player in the AI inference market, offering a comprehensive platform for deploying and managing AI models. The company provides access to various high-performance models, including Llama 3.1 405B and Qwen2.5 Coder 32B, which represent their highest quality offerings [1]. Their platform demonstrates impressive performance metrics, with some models achieving speeds up to 300 tokens per second for serverless deployments [2].\n",
       "\n",
       "The company has gained substantial market traction, recently securing $52 million in Series B funding that values the company at $552 million [3]. With an annual revenue of $6 million and a team of 60 employees, Fireworks AI serves notable clients including DoorDash, Quora, and Upwork [4, 5].\n",
       "\n",
       "Their platform differentiates itself through specialized features like FP8 quantization for large models and continuous batching capabilities [6]. Fireworks claims to offer approximately 3x faster speeds compared to competitors like Hugging Face TGI when using the same GPU configuration [2]. The company focuses on providing smaller, production-grade models that can be deployed privately and securely, rather than generic mega models [5].\n",
       "\n",
       "### Sources\n",
       "[1] Fireworks: Models Intelligence, Performance & Price: https://artificialanalysis.ai/providers/fireworks\n",
       "[2] Fireworks Platform Spring 2024 Updates: https://fireworks.ai/blog/spring-update-faster-models-dedicated-deployments-postpaid-pricing\n",
       "[3] Fireworks AI Raises $52M in Series B Funding: https://siliconvalleyjournals.com/fireworks-ai-raises-52m-in-series-b-funding-to-expand-genai-inference-platform/\n",
       "[4] Fireworks AI: Contact Details, Revenue, Funding: https://siliconvalleyjournals.com/company/fireworks-ai/\n",
       "[5] Fireworks AI Valued at $552 Million After New Funding Round: https://www.pymnts.com/news/investment-tracker/2024/fireworks-ai-valued-552-million-dollars-after-new-funding-round/\n",
       "[6] Inference performance - Fireworks AI Docs: https://docs.fireworks.ai/faq/models/inference/performance\n",
       "\n",
       "## Together.ai Overview\n",
       "\n",
       "Together.ai is a comprehensive AI acceleration platform that has quickly emerged as a significant player in the AI inference space since its founding in 2022 [1]. The platform enables developers to access over 200 AI models, offering high-performance inference capabilities with optimized infrastructure [2].\n",
       "\n",
       "The company's Inference Engine, built on NVIDIA Tensor Core GPUs, delivers impressive performance metrics, achieving 117 tokens per second on Llama-2-70B-Chat models [3]. Their model offerings include high-quality options like Llama 3.3 70B Turbo and Llama 3.1 405B Turbo, with some models achieving sub-100ms latency [4].\n",
       "\n",
       "Together.ai has demonstrated remarkable growth, with estimates suggesting $130M in annualized recurring revenue (ARR) in 2024, representing a 400% year-over-year increase [5]. The platform has received positive user feedback, maintaining a 4.8/5.0 rating based on 162 user reviews [6].\n",
       "\n",
       "The company differentiates itself through competitive pricing and technical innovations, incorporating advanced features like token caching, load balancing, and model quantization [2]. Users particularly praise the platform's straightforward API, reliable performance, and competitive pricing compared to alternatives [7].\n",
       "\n",
       "### Sources\n",
       "[1] https://siliconvalleyjournals.com/company/together-ai/\n",
       "[2] https://www.keywordsai.co/blog/top-10-llm-api-providers\n",
       "[3] https://www.together.ai/blog/together-inference-engine-v1\n",
       "[4] https://artificialanalysis.ai/providers/togetherai\n",
       "[5] https://sacra.com/c/together-ai/\n",
       "[6] https://www.featuredcustomers.com/vendor/together-ai\n",
       "[7] https://www.trustpilot.com/review/together.ai\n",
       "\n",
       "## Groq AI Overview\n",
       "\n",
       "Groq has emerged as a significant player in the AI inference market with its innovative Language Processing Unit (LPU) technology. Independent benchmarks from ArtificialAnalysis.ai demonstrate Groq's impressive performance, with their Llama 2 Chat (70B) API achieving 241 tokens per second - more than double the speed of competing providers [1]. Their total response time for 100 output tokens is just 0.8 seconds, positioning them as a leader in inference speed [2].\n",
       "\n",
       "The company's core technology, the Tensor Streaming Processor (TSP), delivers 500-700 tokens per second on large language models, representing a 5-10x improvement over Nvidia's latest data center GPUs [3]. This performance advantage has positioned Groq as a compelling alternative in the inference market, particularly for startups seeking cost-effective solutions [4].\n",
       "\n",
       "With approximately 300 employees, 60% of whom are software engineers, Groq has evolved beyond hardware to become a comprehensive AI solutions provider [5]. The company is strategically positioned to capture a share of the growing inference chip market, which is projected to reach $48 billion by 2027 [6].\n",
       "\n",
       "### Sources\n",
       "[1] https://groq.com/artificialanalysis-ai-llm-benchmark-doubles-axis-to-fit-new-groq-lpu-inference-engine-performance-results/\n",
       "[2] https://groq.com/inference/\n",
       "[3] https://sacra.com/c/groq/\n",
       "[4] https://venturebeat.com/ai/ai-chip-race-groq-ceo-takes-on-nvidia-claims-most-startups-will-use-speedy-lpus-by-end-of-2024/\n",
       "[5] https://www.businessinsider.com/groq-nvidia-software-advantage-cuda-moat-challenge-inference-2024-12?op=1\n",
       "[6] https://generativeaitech.substack.com/p/groq-and-its-impact-on-ai-inference\n",
       "\n",
       "## Market Comparison and Trends\n",
       "\n",
       "The AI inference market shows distinct positioning among key players Fireworks, Together.ai, and Groq. Fireworks AI differentiates itself through superior speed, leveraging its proprietary FireAttention inference engine for text, image, and audio processing, while maintaining HIPAA and SOC2 compliance for data privacy [1]. Together.ai has demonstrated remarkable growth, reaching an estimated $130M in annualized recurring revenue (ARR) in 2024, representing a 400% year-over-year increase [2].\n",
       "\n",
       "Both Fireworks and Together.ai focus on providing high-quality open-source models that can compete with proprietary alternatives [1]. Pricing structures vary across providers, with model costs ranging from $0.20 to $3.00 per million tokens for different model sizes and capabilities [3]. Together.ai's business model benefits from GPU price commoditization, allowing them to maintain competitive token pricing while focusing on developer experience and reliable inference across diverse open-source models [2].\n",
       "\n",
       "Groq distinguishes itself by offering quick access to various open-source AI models from major providers like Google, Meta, OpenAI, and Mistral through the OpenRouter API platform [4]. The market shows a trend toward comprehensive platform offerings, with providers competing on factors such as model variety, inference speed, and specialized features for different use cases.\n",
       "\n",
       "### Sources\n",
       "[1] Top 10 AI Inference Platforms in 2025: Comparing LLM API Providers: https://www.helicone.ai/blog/llm-api-providers\n",
       "[2] Together AI revenue, valuation & growth rate | Sacra: https://sacra.com/c/together-ai/\n",
       "[3] Models Leaderboard : Comparison of AI Models & API Providers - Groq AI: https://groq-ai.com/models/\n",
       "[4] Pricing: Compare Groq API Pricing With Other API Providers: https://groq-ai.com/pricing/\n",
       "\n",
       "## Conclusion\n",
       "\n",
       "The AI inference market shows clear differentiation among key players, with each provider carving out unique value propositions. Together.ai leads in revenue growth with $130M ARR, while Fireworks AI demonstrates strong potential with its recent $552M valuation. Groq distinguishes itself through superior inference speeds, achieving up to 700 tokens per second with its LPU technology.\n",
       "\n",
       "| Metric | Fireworks | Together.ai | Groq |\n",
       "|--------|-----------|-------------|------|\n",
       "| ARR | $6M | $130M | Not disclosed |\n",
       "| Speed (tokens/sec) | Up to 300 | Up to 117 | 500-700 |\n",
       "| Key Differentiator | FP8 quantization | 200+ model access | LPU technology |\n",
       "| Notable Feature | 3x faster than competitors | Token caching | Sub-second response |\n",
       "| Target Market | Enterprise security | Developer platforms | Startup solutions |\n",
       "\n",
       "The market trends indicate a shift toward comprehensive platform offerings that balance speed, cost-efficiency, and model variety. As the inference chip market approaches $48B by 2027, providers focusing on specialized hardware solutions and optimized infrastructure will likely gain competitive advantages. Success will depend on maintaining the delicate balance between performance, pricing, and platform features while addressing growing enterprise demands for security and reliability."
      ],
      "text/plain": [
       "<IPython.core.display.Markdown object>"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "final_state = graph.get_state(thread)\n",
    "report = final_state.values.get('final_report')\n",
    "Markdown(report)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "open-deep-research-env",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}



---
File: /src/open_deep_research/graph.py
---

from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command

from open_deep_research.state import (
    ReportStateInput,
    ReportStateOutput,
    Sections,
    ReportState,
    SectionState,
    SectionOutputState,
    Queries,
    Feedback
)

from open_deep_research.prompts import (
    report_planner_query_writer_instructions,
    report_planner_instructions,
    query_writer_instructions, 
    section_writer_instructions,
    final_section_writer_instructions,
    section_grader_instructions,
    section_writer_inputs
)

from open_deep_research.configuration import Configuration
from open_deep_research.utils import (
    format_sections, 
    get_config_value, 
    get_search_params, 
    select_and_execute_search
)

## Nodes -- 

async def generate_report_plan(state: ReportState, config: RunnableConfig):
    """Generate the initial report plan with sections.
    
    This node:
    1. Gets configuration for the report structure and search parameters
    2. Generates search queries to gather context for planning
    3. Performs web searches using those queries
    4. Uses an LLM to generate a structured plan with sections
    
    Args:
        state: Current graph state containing the report topic
        config: Configuration for models, search APIs, etc.
        
    Returns:
        Dict containing the generated sections
    """

    # Inputs
    topic = state["topic"]
    feedback = state.get("feedback_on_report_plan", None)

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    report_structure = configurable.report_structure
    number_of_queries = configurable.number_of_queries
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}  # Get the config dict, default to empty
    params_to_pass = get_search_params(search_api, search_api_config)  # Filter parameters

    # Convert JSON object to string if necessary
    if isinstance(report_structure, dict):
        report_structure = str(report_structure)

    # Set writer model (model used for query writing)
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider) 
    structured_llm = writer_model.with_structured_output(Queries)

    # Format system instructions
    system_instructions_query = report_planner_query_writer_instructions.format(topic=topic, report_organization=report_structure, number_of_queries=number_of_queries)

    # Generate queries  
    results = structured_llm.invoke([SystemMessage(content=system_instructions_query),
                                     HumanMessage(content="Generate search queries that will help with planning the sections of the report.")])

    # Web search
    query_list = [query.search_query for query in results.queries]

    # Search the web with parameters
    source_str = await select_and_execute_search(search_api, query_list, params_to_pass)

    # Format system instructions
    system_instructions_sections = report_planner_instructions.format(topic=topic, report_organization=report_structure, context=source_str, feedback=feedback)

    # Set the planner
    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)

    # Report planner instructions
    planner_message = """Generate the sections of the report. Your response must include a 'sections' field containing a list of sections. 
                        Each section must have: name, description, plan, research, and content fields."""

    # Run the planner
    if planner_model == "claude-3-7-sonnet-latest":
        # Allocate a thinking budget for claude-3-7-sonnet-latest as the planner model
        planner_llm = init_chat_model(model=planner_model, 
                                      model_provider=planner_provider, 
                                      max_tokens=20_000, 
                                      thinking={"type": "enabled", "budget_tokens": 16_000})

    else:
        # With other models, thinking tokens are not specifically allocated
        planner_llm = init_chat_model(model=planner_model, 
                                      model_provider=planner_provider)
    
    # Generate the report sections
    structured_llm = planner_llm.with_structured_output(Sections)
    report_sections = structured_llm.invoke([SystemMessage(content=system_instructions_sections),
                                             HumanMessage(content=planner_message)])

    # Get sections
    sections = report_sections.sections

    return {"sections": sections}

def human_feedback(state: ReportState, config: RunnableConfig) -> Command[Literal["generate_report_plan","build_section_with_web_research"]]:
    """Get human feedback on the report plan and route to next steps.
    
    This node:
    1. Formats the current report plan for human review
    2. Gets feedback via an interrupt
    3. Routes to either:
       - Section writing if plan is approved
       - Plan regeneration if feedback is provided
    
    Args:
        state: Current graph state with sections to review
        config: Configuration for the workflow
        
    Returns:
        Command to either regenerate plan or start section writing
    """

    # Get sections
    topic = state["topic"]
    sections = state['sections']
    sections_str = "\n\n".join(
        f"Section: {section.name}\n"
        f"Description: {section.description}\n"
        f"Research needed: {'Yes' if section.research else 'No'}\n"
        for section in sections
    )

    # Get feedback on the report plan from interrupt
    interrupt_message = f"""Please provide feedback on the following report plan. 
                        \n\n{sections_str}\n
                        \nDoes the report plan meet your needs?\nPass 'true' to approve the report plan.\nOr, provide feedback to regenerate the report plan:"""
    
    feedback = interrupt(interrupt_message)

    # If the user approves the report plan, kick off section writing
    if isinstance(feedback, bool) and feedback is True:
        # Treat this as approve and kick off section writing
        return Command(goto=[
            Send("build_section_with_web_research", {"topic": topic, "section": s, "search_iterations": 0}) 
            for s in sections 
            if s.research
        ])
    
    # If the user provides feedback, regenerate the report plan 
    elif isinstance(feedback, str):
        # Treat this as feedback
        return Command(goto="generate_report_plan", 
                       update={"feedback_on_report_plan": feedback})
    else:
        raise TypeError(f"Interrupt value of type {type(feedback)} is not supported.")
    
def generate_queries(state: SectionState, config: RunnableConfig):
    """Generate search queries for researching a specific section.
    
    This node uses an LLM to generate targeted search queries based on the 
    section topic and description.
    
    Args:
        state: Current state containing section details
        config: Configuration including number of queries to generate
        
    Returns:
        Dict containing the generated search queries
    """

    # Get state 
    topic = state["topic"]
    section = state["section"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    number_of_queries = configurable.number_of_queries

    # Generate queries 
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider) 
    structured_llm = writer_model.with_structured_output(Queries)

    # Format system instructions
    system_instructions = query_writer_instructions.format(topic=topic, 
                                                           section_topic=section.description, 
                                                           number_of_queries=number_of_queries)

    # Generate queries  
    queries = structured_llm.invoke([SystemMessage(content=system_instructions),
                                     HumanMessage(content="Generate search queries on the provided topic.")])

    return {"search_queries": queries.queries}

async def search_web(state: SectionState, config: RunnableConfig):
    """Execute web searches for the section queries.
    
    This node:
    1. Takes the generated queries
    2. Executes searches using configured search API
    3. Formats results into usable context
    
    Args:
        state: Current state with search queries
        config: Search API configuration
        
    Returns:
        Dict with search results and updated iteration count
    """

    # Get state
    search_queries = state["search_queries"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}  # Get the config dict, default to empty
    params_to_pass = get_search_params(search_api, search_api_config)  # Filter parameters

    # Web search
    query_list = [query.search_query for query in search_queries]

    # Search the web with parameters
    source_str = await select_and_execute_search(search_api, query_list, params_to_pass)

    return {"source_str": source_str, "search_iterations": state["search_iterations"] + 1}

def write_section(state: SectionState, config: RunnableConfig) -> Command[Literal[END, "search_web"]]:
    """Write a section of the report and evaluate if more research is needed.
    
    This node:
    1. Writes section content using search results
    2. Evaluates the quality of the section
    3. Either:
       - Completes the section if quality passes
       - Triggers more research if quality fails
    
    Args:
        state: Current state with search results and section info
        config: Configuration for writing and evaluation
        
    Returns:
        Command to either complete section or do more research
    """

    # Get state 
    topic = state["topic"]
    section = state["section"]
    source_str = state["source_str"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Format system instructions
    section_writer_inputs_formatted = section_writer_inputs.format(topic=topic, 
                                                             section_name=section.name, 
                                                             section_topic=section.description, 
                                                             context=source_str, 
                                                             section_content=section.content)

    # Generate section  
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider) 

    section_content = writer_model.invoke([SystemMessage(content=section_writer_instructions),
                                           HumanMessage(content=section_writer_inputs_formatted)])
    
    # Write content to the section object  
    section.content = section_content.content

    # Grade prompt 
    section_grader_message = ("Grade the report and consider follow-up questions for missing information. "
                              "If the grade is 'pass', return empty strings for all follow-up queries. "
                              "If the grade is 'fail', provide specific search queries to gather missing information.")
    
    section_grader_instructions_formatted = section_grader_instructions.format(topic=topic, 
                                                                               section_topic=section.description,
                                                                               section=section.content, 
                                                                               number_of_follow_up_queries=configurable.number_of_queries)

    # Use planner model for reflection
    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)

    if planner_model == "claude-3-7-sonnet-latest":
        # Allocate a thinking budget for claude-3-7-sonnet-latest as the planner model
        reflection_model = init_chat_model(model=planner_model, 
                                           model_provider=planner_provider, 
                                           max_tokens=20_000, 
                                           thinking={"type": "enabled", "budget_tokens": 16_000}).with_structured_output(Feedback)
    else:
        reflection_model = init_chat_model(model=planner_model, 
                                           model_provider=planner_provider).with_structured_output(Feedback)
    # Generate feedback
    feedback = reflection_model.invoke([SystemMessage(content=section_grader_instructions_formatted),
                                        HumanMessage(content=section_grader_message)])

    # If the section is passing or the max search depth is reached, publish the section to completed sections 
    if feedback.grade == "pass" or state["search_iterations"] >= configurable.max_search_depth:
        # Publish the section to completed sections 
        return  Command(
        update={"completed_sections": [section]},
        goto=END
    )

    # Update the existing section with new content and update search queries
    else:
        return  Command(
        update={"search_queries": feedback.follow_up_queries, "section": section},
        goto="search_web"
        )
    
def write_final_sections(state: SectionState, config: RunnableConfig):
    """Write sections that don't require research using completed sections as context.
    
    This node handles sections like conclusions or summaries that build on
    the researched sections rather than requiring direct research.
    
    Args:
        state: Current state with completed sections as context
        config: Configuration for the writing model
        
    Returns:
        Dict containing the newly written section
    """

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Get state 
    topic = state["topic"]
    section = state["section"]
    completed_report_sections = state["report_sections_from_research"]
    
    # Format system instructions
    system_instructions = final_section_writer_instructions.format(topic=topic, section_name=section.name, section_topic=section.description, context=completed_report_sections)

    # Generate section  
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider) 
    
    section_content = writer_model.invoke([SystemMessage(content=system_instructions),
                                           HumanMessage(content="Generate a report section based on the provided sources.")])
    
    # Write content to section 
    section.content = section_content.content

    # Write the updated section to completed sections
    return {"completed_sections": [section]}

def gather_completed_sections(state: ReportState):
    """Format completed sections as context for writing final sections.
    
    This node takes all completed research sections and formats them into
    a single context string for writing summary sections.
    
    Args:
        state: Current state with completed sections
        
    Returns:
        Dict with formatted sections as context
    """

    # List of completed sections
    completed_sections = state["completed_sections"]

    # Format completed section to str to use as context for final sections
    completed_report_sections = format_sections(completed_sections)

    return {"report_sections_from_research": completed_report_sections}

def compile_final_report(state: ReportState):
    """Compile all sections into the final report.
    
    This node:
    1. Gets all completed sections
    2. Orders them according to original plan
    3. Combines them into the final report
    
    Args:
        state: Current state with all completed sections
        
    Returns:
        Dict containing the complete report
    """

    # Get sections
    sections = state["sections"]
    completed_sections = {s.name: s.content for s in state["completed_sections"]}

    # Update sections with completed content while maintaining original order
    for section in sections:
        section.content = completed_sections[section.name]

    # Compile final report
    all_sections = "\n\n".join([s.content for s in sections])

    return {"final_report": all_sections}

def initiate_final_section_writing(state: ReportState):
    """Create parallel tasks for writing non-research sections.
    
    This edge function identifies sections that don't need research and
    creates parallel writing tasks for each one.
    
    Args:
        state: Current state with all sections and research context
        
    Returns:
        List of Send commands for parallel section writing
    """

    # Kick off section writing in parallel via Send() API for any sections that do not require research
    return [
        Send("write_final_sections", {"topic": state["topic"], "section": s, "report_sections_from_research": state["report_sections_from_research"]}) 
        for s in state["sections"] 
        if not s.research
    ]

# Report section sub-graph -- 

# Add nodes 
section_builder = StateGraph(SectionState, output=SectionOutputState)
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("search_web", search_web)
section_builder.add_node("write_section", write_section)

# Add edges
section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "write_section")

# Outer graph for initial report plan compiling results from each section -- 

# Add nodes
builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput, config_schema=Configuration)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("human_feedback", human_feedback)
builder.add_node("build_section_with_web_research", section_builder.compile())
builder.add_node("gather_completed_sections", gather_completed_sections)
builder.add_node("write_final_sections", write_final_sections)
builder.add_node("compile_final_report", compile_final_report)

# Add edges
builder.add_edge(START, "generate_report_plan")
builder.add_edge("generate_report_plan", "human_feedback")
builder.add_edge("build_section_with_web_research", "gather_completed_sections")
builder.add_conditional_edges("gather_completed_sections", initiate_final_section_writing, ["write_final_sections"])
builder.add_edge("write_final_sections", "compile_final_report")
builder.add_edge("compile_final_report", END)

graph = builder.compile()



---
File: /src/open_deep_research/prompts.py
---

report_planner_query_writer_instructions="""You are performing research for a report. 

<Report topic>
{topic}
</Report topic>

<Report organization>
{report_organization}
</Report organization>

<Task>
Your goal is to generate {number_of_queries} web search queries that will help gather information for planning the report sections. 

The queries should:

1. Be related to the Report topic
2. Help satisfy the requirements specified in the report organization

Make the queries specific enough to find high-quality, relevant sources while covering the breadth needed for the report structure.
</Task>

<Format>
Call the Queries tool 
</Format>
"""

report_planner_instructions="""I want a plan for a report that is concise and focused.

<Report topic>
The topic of the report is:
{topic}
</Report topic>

<Report organization>
The report should follow this organization: 
{report_organization}
</Report organization>

<Context>
Here is context to use to plan the sections of the report: 
{context}
</Context>

<Task>
Generate a list of sections for the report. Your plan should be tight and focused with NO overlapping sections or unnecessary filler. 

For example, a good report structure might look like:
1/ intro
2/ overview of topic A
3/ overview of topic B
4/ comparison between A and B
5/ conclusion

Each section should have the fields:

- Name - Name for this section of the report.
- Description - Brief overview of the main topics covered in this section.
- Research - Whether to perform web research for this section of the report.
- Content - The content of the section, which you will leave blank for now.

Integration guidelines:
- Include examples and implementation details within main topic sections, not as separate sections
- Ensure each section has a distinct purpose with no content overlap
- Combine related concepts rather than separating them

Before submitting, review your structure to ensure it has no redundant sections and follows a logical flow.
</Task>

<Feedback>
Here is feedback on the report structure from review (if any):
{feedback}
</Feedback>

<Format>
Call the Sections tool 
</Format>
"""

query_writer_instructions="""You are an expert technical writer crafting targeted web search queries that will gather comprehensive information for writing a technical report section.

<Report topic>
{topic}
</Report topic>

<Section topic>
{section_topic}
</Section topic>

<Task>
Your goal is to generate {number_of_queries} search queries that will help gather comprehensive information above the section topic. 

The queries should:

1. Be related to the topic 
2. Examine different aspects of the topic

Make the queries specific enough to find high-quality, relevant sources.
</Task>

<Format>
Call the Queries tool 
</Format>
"""

section_writer_instructions = """Write one section of a research report.

<Task>
1. Review the report topic, section name, and section topic carefully.
2. If present, review any existing section content. 
3. Then, look at the provided Source material.
4. Decide the sources that you will use it to write a report section.
5. Write the report section and list your sources. 
</Task>

<Writing Guidelines>
- If existing section content is not populated, write from scratch
- If existing section content is populated, synthesize it with the source material
- Strict 150-200 word limit
- Use simple, clear language
- Use short paragraphs (2-3 sentences max)
- Use ## for section title (Markdown format)
</Writing Guidelines>

<Citation Rules>
- Assign each unique URL a single citation number in your text
- End with ### Sources that lists each source with corresponding numbers
- IMPORTANT: Number sources sequentially without gaps (1,2,3,4...) in the final list regardless of which sources you choose
- Example format:
  [1] Source Title: URL
  [2] Source Title: URL
</Citation Rules>

<Final Check>
1. Verify that EVERY claim is grounded in the provided Source material
2. Confirm each URL appears ONLY ONCE in the Source list
3. Verify that sources are numbered sequentially (1,2,3...) without any gaps
</Final Check>
"""

section_writer_inputs=""" 
<Report topic>
{topic}
</Report topic>

<Section name>
{section_name}
</Section name>

<Section topic>
{section_topic}
</Section topic>

<Existing section content (if populated)>
{section_content}
</Existing section content>

<Source material>
{context}
</Source material>
"""

section_grader_instructions = """Review a report section relative to the specified topic:

<Report topic>
{topic}
</Report topic>

<section topic>
{section_topic}
</section topic>

<section content>
{section}
</section content>

<task>
Evaluate whether the section content adequately addresses the section topic.

If the section content does not adequately address the section topic, generate {number_of_follow_up_queries} follow-up search queries to gather missing information.
</task>

<format>
Call the Feedback tool and output with the following schema:

grade: Literal["pass","fail"] = Field(
    description="Evaluation result indicating whether the response meets requirements ('pass') or needs revision ('fail')."
)
follow_up_queries: List[SearchQuery] = Field(
    description="List of follow-up search queries.",
)
</format>
"""

final_section_writer_instructions="""You are an expert technical writer crafting a section that synthesizes information from the rest of the report.

<Report topic>
{topic}
</Report topic>

<Section name>
{section_name}
</Section name>

<Section topic> 
{section_topic}
</Section topic>

<Available report content>
{context}
</Available report content>

<Task>
1. Section-Specific Approach:

For Introduction:
- Use # for report title (Markdown format)
- 50-100 word limit
- Write in simple and clear language
- Focus on the core motivation for the report in 1-2 paragraphs
- Use a clear narrative arc to introduce the report
- Include NO structural elements (no lists or tables)
- No sources section needed

For Conclusion/Summary:
- Use ## for section title (Markdown format)
- 100-150 word limit
- For comparative reports:
    * Must include a focused comparison table using Markdown table syntax
    * Table should distill insights from the report
    * Keep table entries clear and concise
- For non-comparative reports: 
    * Only use ONE structural element IF it helps distill the points made in the report:
    * Either a focused table comparing items present in the report (using Markdown table syntax)
    * Or a short list using proper Markdown list syntax:
      - Use `*` or `-` for unordered lists
      - Use `1.` for ordered lists
      - Ensure proper indentation and spacing
- End with specific next steps or implications
- No sources section needed

3. Writing Approach:
- Use concrete details over general statements
- Make every word count
- Focus on your single most important point
</Task>

<Quality Checks>
- For introduction: 50-100 word limit, # for report title, no structural elements, no sources section
- For conclusion: 100-150 word limit, ## for section title, only ONE structural element at most, no sources section
- Markdown format
- Do not include word count or any preamble in your response
</Quality Checks>"""


---
File: /src/open_deep_research/state.py
---

from typing import Annotated, List, TypedDict, Literal
from pydantic import BaseModel, Field
import operator

class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )
    research: bool = Field(
        description="Whether to perform web research for this section of the report."
    )
    content: str = Field(
        description="The content of the section."
    )   

class Sections(BaseModel):
    sections: List[Section] = Field(
        description="Sections of the report.",
    )

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query for web search.")

class Queries(BaseModel):
    queries: List[SearchQuery] = Field(
        description="List of search queries.",
    )

class Feedback(BaseModel):
    grade: Literal["pass","fail"] = Field(
        description="Evaluation result indicating whether the response meets requirements ('pass') or needs revision ('fail')."
    )
    follow_up_queries: List[SearchQuery] = Field(
        description="List of follow-up search queries.",
    )

class ReportStateInput(TypedDict):
    topic: str # Report topic
    
class ReportStateOutput(TypedDict):
    final_report: str # Final report

class ReportState(TypedDict):
    topic: str # Report topic    
    feedback_on_report_plan: str # Feedback on the report plan
    sections: list[Section] # List of report sections 
    completed_sections: Annotated[list, operator.add] # Send() API key
    report_sections_from_research: str # String of any completed sections from research to write final sections
    final_report: str # Final report

class SectionState(TypedDict):
    topic: str # Report topic
    section: Section # Report section  
    search_iterations: int # Number of search iterations done
    search_queries: list[SearchQuery] # List of search queries
    source_str: str # String of formatted source content from web search
    report_sections_from_research: str # String of any completed sections from research to write final sections
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API

class SectionOutputState(TypedDict):
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API



---
File: /src/open_deep_research/utils.py
---

import os
import asyncio
import requests
import random 
import concurrent
import aiohttp
import time
import logging
from typing import List, Optional, Dict, Any, Union
from urllib.parse import unquote

from exa_py import Exa
from linkup import LinkupClient
from tavily import AsyncTavilyClient
from duckduckgo_search import DDGS 
from bs4 import BeautifulSoup

from langchain_community.retrievers import ArxivRetriever
from langchain_community.utilities.pubmed import PubMedAPIWrapper
from langsmith import traceable

from open_deep_research.state import Section


def get_config_value(value):
    """
    Helper function to handle both string and enum cases of configuration values
    """
    return value if isinstance(value, str) else value.value

def get_search_params(search_api: str, search_api_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Filters the search_api_config dictionary to include only parameters accepted by the specified search API.

    Args:
        search_api (str): The search API identifier (e.g., "exa", "tavily").
        search_api_config (Optional[Dict[str, Any]]): The configuration dictionary for the search API.

    Returns:
        Dict[str, Any]: A dictionary of parameters to pass to the search function.
    """
    # Define accepted parameters for each search API
    SEARCH_API_PARAMS = {
        "exa": ["max_characters", "num_results", "include_domains", "exclude_domains", "subpages"],
        "tavily": [],  # Tavily currently accepts no additional parameters
        "perplexity": [],  # Perplexity accepts no additional parameters
        "arxiv": ["load_max_docs", "get_full_documents", "load_all_available_meta"],
        "pubmed": ["top_k_results", "email", "api_key", "doc_content_chars_max"],
        "linkup": ["depth"],
    }

    # Get the list of accepted parameters for the given search API
    accepted_params = SEARCH_API_PARAMS.get(search_api, [])

    # If no config provided, return an empty dict
    if not search_api_config:
        return {}

    # Filter the config to only include accepted parameters
    return {k: v for k, v in search_api_config.items() if k in accepted_params}

def deduplicate_and_format_sources(search_response, max_tokens_per_source, include_raw_content=True):
    """
    Takes a list of search responses and formats them into a readable string.
    Limits the raw_content to approximately max_tokens_per_source tokens.
 
    Args:
        search_responses: List of search response dicts, each containing:
            - query: str
            - results: List of dicts with fields:
                - title: str
                - url: str
                - content: str
                - score: float
                - raw_content: str|None
        max_tokens_per_source: int
        include_raw_content: bool
            
    Returns:
        str: Formatted string with deduplicated sources
    """
     # Collect all results
    sources_list = []
    for response in search_response:
        sources_list.extend(response['results'])
    
    # Deduplicate by URL
    unique_sources = {source['url']: source for source in sources_list}

    # Format output
    formatted_text = "Content from sources:\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"{'='*80}\n"  # Clear section separator
        formatted_text += f"Source: {source['title']}\n"
        formatted_text += f"{'-'*80}\n"  # Subsection separator
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
        formatted_text += f"{'='*80}\n\n" # End section separator
                
    return formatted_text.strip()

def format_sections(sections: list[Section]) -> str:
    """ Format a list of sections into a string """
    formatted_str = ""
    for idx, section in enumerate(sections, 1):
        formatted_str += f"""
{'='*60}
Section {idx}: {section.name}
{'='*60}
Description:
{section.description}
Requires Research: 
{section.research}

Content:
{section.content if section.content else '[Not yet written]'}

"""
    return formatted_str

@traceable
async def tavily_search_async(search_queries):
    """
    Performs concurrent web searches using the Tavily API.

    Args:
        search_queries (List[SearchQuery]): List of search queries to process

    Returns:
            List[dict]: List of search responses from Tavily API, one per query. Each response has format:
                {
                    'query': str, # The original search query
                    'follow_up_questions': None,      
                    'answer': None,
                    'images': list,
                    'results': [                     # List of search results
                        {
                            'title': str,            # Title of the webpage
                            'url': str,              # URL of the result
                            'content': str,          # Summary/snippet of content
                            'score': float,          # Relevance score
                            'raw_content': str|None  # Full page content if available
                        },
                        ...
                    ]
                }
    """
    tavily_async_client = AsyncTavilyClient()
    search_tasks = []
    for query in search_queries:
            search_tasks.append(
                tavily_async_client.search(
                    query,
                    max_results=5,
                    include_raw_content=True,
                    topic="general"
                )
            )

    # Execute all searches concurrently
    search_docs = await asyncio.gather(*search_tasks)

    return search_docs

@traceable
def perplexity_search(search_queries):
    """Search the web using the Perplexity API.
    
    Args:
        search_queries (List[SearchQuery]): List of search queries to process
  
    Returns:
        List[dict]: List of search responses from Perplexity API, one per query. Each response has format:
            {
                'query': str,                    # The original search query
                'follow_up_questions': None,      
                'answer': None,
                'images': list,
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the search result
                        'url': str,              # URL of the result
                        'content': str,          # Summary/snippet of content
                        'score': float,          # Relevance score
                        'raw_content': str|None  # Full content or None for secondary citations
                    },
                    ...
                ]
            }
    """

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}"
    }
    
    search_docs = []
    for query in search_queries:

        payload = {
            "model": "sonar-pro",
            "messages": [
                {
                    "role": "system",
                    "content": "Search the web and provide factual information with sources."
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
        }
        
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()  # Raise exception for bad status codes
        
        # Parse the response
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        citations = data.get("citations", ["https://perplexity.ai"])
        
        # Create results list for this query
        results = []
        
        # First citation gets the full content
        results.append({
            "title": f"Perplexity Search, Source 1",
            "url": citations[0],
            "content": content,
            "raw_content": content,
            "score": 1.0  # Adding score to match Tavily format
        })
        
        # Add additional citations without duplicating content
        for i, citation in enumerate(citations[1:], start=2):
            results.append({
                "title": f"Perplexity Search, Source {i}",
                "url": citation,
                "content": "See primary source for full content",
                "raw_content": None,
                "score": 0.5  # Lower score for secondary sources
            })
        
        # Format response to match Tavily structure
        search_docs.append({
            "query": query,
            "follow_up_questions": None,
            "answer": None,
            "images": [],
            "results": results
        })
    
    return search_docs

@traceable
async def exa_search(search_queries, max_characters: Optional[int] = None, num_results=5, 
                     include_domains: Optional[List[str]] = None, 
                     exclude_domains: Optional[List[str]] = None,
                     subpages: Optional[int] = None):
    """Search the web using the Exa API.
    
    Args:
        search_queries (List[SearchQuery]): List of search queries to process
        max_characters (int, optional): Maximum number of characters to retrieve for each result's raw content.
                                       If None, the text parameter will be set to True instead of an object.
        num_results (int): Number of search results per query. Defaults to 5.
        include_domains (List[str], optional): List of domains to include in search results. 
            When specified, only results from these domains will be returned.
        exclude_domains (List[str], optional): List of domains to exclude from search results.
            Cannot be used together with include_domains.
        subpages (int, optional): Number of subpages to retrieve per result. If None, subpages are not retrieved.
        
    Returns:
        List[dict]: List of search responses from Exa API, one per query. Each response has format:
            {
                'query': str,                    # The original search query
                'follow_up_questions': None,      
                'answer': None,
                'images': list,
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the search result
                        'url': str,              # URL of the result
                        'content': str,          # Summary/snippet of content
                        'score': float,          # Relevance score
                        'raw_content': str|None  # Full content or None for secondary citations
                    },
                    ...
                ]
            }
    """
    # Check that include_domains and exclude_domains are not both specified
    if include_domains and exclude_domains:
        raise ValueError("Cannot specify both include_domains and exclude_domains")
    
    # Initialize Exa client (API key should be configured in your .env file)
    exa = Exa(api_key = f"{os.getenv('EXA_API_KEY')}")
    
    # Define the function to process a single query
    async def process_query(query):
        # Use run_in_executor to make the synchronous exa call in a non-blocking way
        loop = asyncio.get_event_loop()
        
        # Define the function for the executor with all parameters
        def exa_search_fn():
            # Build parameters dictionary
            kwargs = {
                # Set text to True if max_characters is None, otherwise use an object with max_characters
                "text": True if max_characters is None else {"max_characters": max_characters},
                "summary": True,  # This is an amazing feature by EXA. It provides an AI generated summary of the content based on the query
                "num_results": num_results
            }
            
            # Add optional parameters only if they are provided
            if subpages is not None:
                kwargs["subpages"] = subpages
                
            if include_domains:
                kwargs["include_domains"] = include_domains
            elif exclude_domains:
                kwargs["exclude_domains"] = exclude_domains
                
            return exa.search_and_contents(query, **kwargs)
        
        response = await loop.run_in_executor(None, exa_search_fn)
        
        # Format the response to match the expected output structure
        formatted_results = []
        seen_urls = set()  # Track URLs to avoid duplicates
        
        # Helper function to safely get value regardless of if item is dict or object
        def get_value(item, key, default=None):
            if isinstance(item, dict):
                return item.get(key, default)
            else:
                return getattr(item, key, default) if hasattr(item, key) else default
        
        # Access the results from the SearchResponse object
        results_list = get_value(response, 'results', [])
        
        # First process all main results
        for result in results_list:
            # Get the score with a default of 0.0 if it's None or not present
            score = get_value(result, 'score', 0.0)
            
            # Combine summary and text for content if both are available
            text_content = get_value(result, 'text', '')
            summary_content = get_value(result, 'summary', '')
            
            content = text_content
            if summary_content:
                if content:
                    content = f"{summary_content}\n\n{content}"
                else:
                    content = summary_content
            
            title = get_value(result, 'title', '')
            url = get_value(result, 'url', '')
            
            # Skip if we've seen this URL before (removes duplicate entries)
            if url in seen_urls:
                continue
                
            seen_urls.add(url)
            
            # Main result entry
            result_entry = {
                "title": title,
                "url": url,
                "content": content,
                "score": score,
                "raw_content": text_content
            }
            
            # Add the main result to the formatted results
            formatted_results.append(result_entry)
        
        # Now process subpages only if the subpages parameter was provided
        if subpages is not None:
            for result in results_list:
                subpages_list = get_value(result, 'subpages', [])
                for subpage in subpages_list:
                    # Get subpage score
                    subpage_score = get_value(subpage, 'score', 0.0)
                    
                    # Combine summary and text for subpage content
                    subpage_text = get_value(subpage, 'text', '')
                    subpage_summary = get_value(subpage, 'summary', '')
                    
                    subpage_content = subpage_text
                    if subpage_summary:
                        if subpage_content:
                            subpage_content = f"{subpage_summary}\n\n{subpage_content}"
                        else:
                            subpage_content = subpage_summary
                    
                    subpage_url = get_value(subpage, 'url', '')
                    
                    # Skip if we've seen this URL before
                    if subpage_url in seen_urls:
                        continue
                        
                    seen_urls.add(subpage_url)
                    
                    formatted_results.append({
                        "title": get_value(subpage, 'title', ''),
                        "url": subpage_url,
                        "content": subpage_content,
                        "score": subpage_score,
                        "raw_content": subpage_text
                    })
        
        # Collect images if available (only from main results to avoid duplication)
        images = []
        for result in results_list:
            image = get_value(result, 'image')
            if image and image not in images:  # Avoid duplicate images
                images.append(image)
                
        return {
            "query": query,
            "follow_up_questions": None,
            "answer": None,
            "images": images,
            "results": formatted_results
        }
    
    # Process all queries sequentially with delay to respect rate limit
    search_docs = []
    for i, query in enumerate(search_queries):
        try:
            # Add delay between requests (0.25s = 4 requests per second, well within the 5/s limit)
            if i > 0:  # Don't delay the first request
                await asyncio.sleep(0.25)
            
            result = await process_query(query)
            search_docs.append(result)
        except Exception as e:
            # Handle exceptions gracefully
            print(f"Error processing query '{query}': {str(e)}")
            # Add a placeholder result for failed queries to maintain index alignment
            search_docs.append({
                "query": query,
                "follow_up_questions": None,
                "answer": None,
                "images": [],
                "results": [],
                "error": str(e)
            })
            
            # Add additional delay if we hit a rate limit error
            if "429" in str(e):
                print("Rate limit exceeded. Adding additional delay...")
                await asyncio.sleep(1.0)  # Add a longer delay if we hit a rate limit
    
    return search_docs

@traceable
async def arxiv_search_async(search_queries, load_max_docs=5, get_full_documents=True, load_all_available_meta=True):
    """
    Performs concurrent searches on arXiv using the ArxivRetriever.

    Args:
        search_queries (List[str]): List of search queries or article IDs
        load_max_docs (int, optional): Maximum number of documents to return per query. Default is 5.
        get_full_documents (bool, optional): Whether to fetch full text of documents. Default is True.
        load_all_available_meta (bool, optional): Whether to load all available metadata. Default is True.

    Returns:
        List[dict]: List of search responses from arXiv, one per query. Each response has format:
            {
                'query': str,                    # The original search query
                'follow_up_questions': None,      
                'answer': None,
                'images': [],
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the paper
                        'url': str,              # URL (Entry ID) of the paper
                        'content': str,          # Formatted summary with metadata
                        'score': float,          # Relevance score (approximated)
                        'raw_content': str|None  # Full paper content if available
                    },
                    ...
                ]
            }
    """
    
    async def process_single_query(query):
        try:
            # Create retriever for each query
            retriever = ArxivRetriever(
                load_max_docs=load_max_docs,
                get_full_documents=get_full_documents,
                load_all_available_meta=load_all_available_meta
            )
            
            # Run the synchronous retriever in a thread pool
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(None, lambda: retriever.invoke(query))
            
            results = []
            # Assign decreasing scores based on the order
            base_score = 1.0
            score_decrement = 1.0 / (len(docs) + 1) if docs else 0
            
            for i, doc in enumerate(docs):
                # Extract metadata
                metadata = doc.metadata
                
                # Use entry_id as the URL (this is the actual arxiv link)
                url = metadata.get('entry_id', '')
                
                # Format content with all useful metadata
                content_parts = []

                # Primary information
                if 'Summary' in metadata:
                    content_parts.append(f"Summary: {metadata['Summary']}")

                if 'Authors' in metadata:
                    content_parts.append(f"Authors: {metadata['Authors']}")

                # Add publication information
                published = metadata.get('Published')
                published_str = published.isoformat() if hasattr(published, 'isoformat') else str(published) if published else ''
                if published_str:
                    content_parts.append(f"Published: {published_str}")

                # Add additional metadata if available
                if 'primary_category' in metadata:
                    content_parts.append(f"Primary Category: {metadata['primary_category']}")

                if 'categories' in metadata and metadata['categories']:
                    content_parts.append(f"Categories: {', '.join(metadata['categories'])}")

                if 'comment' in metadata and metadata['comment']:
                    content_parts.append(f"Comment: {metadata['comment']}")

                if 'journal_ref' in metadata and metadata['journal_ref']:
                    content_parts.append(f"Journal Reference: {metadata['journal_ref']}")

                if 'doi' in metadata and metadata['doi']:
                    content_parts.append(f"DOI: {metadata['doi']}")

                # Get PDF link if available in the links
                pdf_link = ""
                if 'links' in metadata and metadata['links']:
                    for link in metadata['links']:
                        if 'pdf' in link:
                            pdf_link = link
                            content_parts.append(f"PDF: {pdf_link}")
                            break

                # Join all content parts with newlines 
                content = "\n".join(content_parts)
                
                result = {
                    'title': metadata.get('Title', ''),
                    'url': url,  # Using entry_id as the URL
                    'content': content,
                    'score': base_score - (i * score_decrement),
                    'raw_content': doc.page_content if get_full_documents else None
                }
                results.append(result)
                
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': results
            }
        except Exception as e:
            # Handle exceptions gracefully
            print(f"Error processing arXiv query '{query}': {str(e)}")
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            }
    
    # Process queries sequentially with delay to respect arXiv rate limit (1 request per 3 seconds)
    search_docs = []
    for i, query in enumerate(search_queries):
        try:
            # Add delay between requests (3 seconds per ArXiv's rate limit)
            if i > 0:  # Don't delay the first request
                await asyncio.sleep(3.0)
            
            result = await process_single_query(query)
            search_docs.append(result)
        except Exception as e:
            # Handle exceptions gracefully
            print(f"Error processing arXiv query '{query}': {str(e)}")
            search_docs.append({
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            })
            
            # Add additional delay if we hit a rate limit error
            if "429" in str(e) or "Too Many Requests" in str(e):
                print("ArXiv rate limit exceeded. Adding additional delay...")
                await asyncio.sleep(5.0)  # Add a longer delay if we hit a rate limit
    
    return search_docs

@traceable
async def pubmed_search_async(search_queries, top_k_results=5, email=None, api_key=None, doc_content_chars_max=4000):
    """
    Performs concurrent searches on PubMed using the PubMedAPIWrapper.

    Args:
        search_queries (List[str]): List of search queries
        top_k_results (int, optional): Maximum number of documents to return per query. Default is 5.
        email (str, optional): Email address for PubMed API. Required by NCBI.
        api_key (str, optional): API key for PubMed API for higher rate limits.
        doc_content_chars_max (int, optional): Maximum characters for document content. Default is 4000.

    Returns:
        List[dict]: List of search responses from PubMed, one per query. Each response has format:
            {
                'query': str,                    # The original search query
                'follow_up_questions': None,      
                'answer': None,
                'images': [],
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the paper
                        'url': str,              # URL to the paper on PubMed
                        'content': str,          # Formatted summary with metadata
                        'score': float,          # Relevance score (approximated)
                        'raw_content': str       # Full abstract content
                    },
                    ...
                ]
            }
    """
    
    async def process_single_query(query):
        try:
            # print(f"Processing PubMed query: '{query}'")
            
            # Create PubMed wrapper for the query
            wrapper = PubMedAPIWrapper(
                top_k_results=top_k_results,
                doc_content_chars_max=doc_content_chars_max,
                email=email if email else "your_email@example.com",
                api_key=api_key if api_key else ""
            )
            
            # Run the synchronous wrapper in a thread pool
            loop = asyncio.get_event_loop()
            
            # Use wrapper.lazy_load instead of load to get better visibility
            docs = await loop.run_in_executor(None, lambda: list(wrapper.lazy_load(query)))
            
            print(f"Query '{query}' returned {len(docs)} results")
            
            results = []
            # Assign decreasing scores based on the order
            base_score = 1.0
            score_decrement = 1.0 / (len(docs) + 1) if docs else 0
            
            for i, doc in enumerate(docs):
                # Format content with metadata
                content_parts = []
                
                if doc.get('Published'):
                    content_parts.append(f"Published: {doc['Published']}")
                
                if doc.get('Copyright Information'):
                    content_parts.append(f"Copyright Information: {doc['Copyright Information']}")
                
                if doc.get('Summary'):
                    content_parts.append(f"Summary: {doc['Summary']}")
                
                # Generate PubMed URL from the article UID
                uid = doc.get('uid', '')
                url = f"https://pubmed.ncbi.nlm.nih.gov/{uid}/" if uid else ""
                
                # Join all content parts with newlines
                content = "\n".join(content_parts)
                
                result = {
                    'title': doc.get('Title', ''),
                    'url': url,
                    'content': content,
                    'score': base_score - (i * score_decrement),
                    'raw_content': doc.get('Summary', '')
                }
                results.append(result)
            
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': results
            }
        except Exception as e:
            # Handle exceptions with more detailed information
            error_msg = f"Error processing PubMed query '{query}': {str(e)}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())  # Print full traceback for debugging
            
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            }
    
    # Process all queries with a reasonable delay between them
    search_docs = []
    
    # Start with a small delay that increases if we encounter rate limiting
    delay = 1.0  # Start with a more conservative delay
    
    for i, query in enumerate(search_queries):
        try:
            # Add delay between requests
            if i > 0:  # Don't delay the first request
                # print(f"Waiting {delay} seconds before next query...")
                await asyncio.sleep(delay)
            
            result = await process_single_query(query)
            search_docs.append(result)
            
            # If query was successful with results, we can slightly reduce delay (but not below minimum)
            if result.get('results') and len(result['results']) > 0:
                delay = max(0.5, delay * 0.9)  # Don't go below 0.5 seconds
            
        except Exception as e:
            # Handle exceptions gracefully
            error_msg = f"Error in main loop processing PubMed query '{query}': {str(e)}"
            print(error_msg)
            
            search_docs.append({
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            })
            
            # If we hit an exception, increase delay for next query
            delay = min(5.0, delay * 1.5)  # Don't exceed 5 seconds
    
    return search_docs

@traceable
async def linkup_search(search_queries, depth: Optional[str] = "standard"):
    """
    Performs concurrent web searches using the Linkup API.

    Args:
        search_queries (List[SearchQuery]): List of search queries to process
        depth (str, optional): "standard" (default)  or "deep". More details here https://docs.linkup.so/pages/documentation/get-started/concepts

    Returns:
        List[dict]: List of search responses from Linkup API, one per query. Each response has format:
            {
                'results': [            # List of search results
                    {
                        'title': str,   # Title of the search result
                        'url': str,     # URL of the result
                        'content': str, # Summary/snippet of content
                    },
                    ...
                ]
            }
    """
    client = LinkupClient()
    search_tasks = []
    for query in search_queries:
        search_tasks.append(
                client.async_search(
                    query,
                    depth,
                    output_type="searchResults",
                )
            )

    search_results = []
    for response in await asyncio.gather(*search_tasks):
        search_results.append(
            {
                "results": [
                    {"title": result.name, "url": result.url, "content": result.content}
                    for result in response.results
                ],
            }
        )

    return search_results
@traceable
async def duckduckgo_search(search_queries):
    """Perform searches using DuckDuckGo
    
    Args:
        search_queries (List[str]): List of search queries to process
        
    Returns:
        List[dict]: List of search results
    """
    async def process_single_query(query):
        # Execute synchronous search in the event loop's thread pool
        loop = asyncio.get_event_loop()
        
        def perform_search():
            results = []
            with DDGS() as ddgs:
                ddg_results = list(ddgs.text(query, max_results=5))
                
                # Format results
                for i, result in enumerate(ddg_results):
                    results.append({
                        'title': result.get('title', ''),
                        'url': result.get('link', ''),
                        'content': result.get('body', ''),
                        'score': 1.0 - (i * 0.1),  # Simple scoring mechanism
                        'raw_content': result.get('body', '')
                    })
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': results
            }
            
        return await loop.run_in_executor(None, perform_search)

    # Execute all queries concurrently
    tasks = [process_single_query(query) for query in search_queries]
    search_docs = await asyncio.gather(*tasks)
    
    return search_docs

@traceable
async def google_search_async(search_queries: Union[str, List[str]], max_results: int = 5, include_raw_content: bool = True):
    """
    Performs concurrent web searches using Google.
    Uses Google Custom Search API if environment variables are set, otherwise falls back to web scraping.

    Args:
        search_queries (List[str]): List of search queries to process
        max_results (int): Maximum number of results to return per query
        include_raw_content (bool): Whether to fetch full page content

    Returns:
        List[dict]: List of search responses from Google, one per query
    """


    # Check for API credentials from environment variables
    api_key = os.environ.get("GOOGLE_API_KEY")
    cx = os.environ.get("GOOGLE_CX")
    use_api = bool(api_key and cx)
    
    # Handle case where search_queries is a single string
    if isinstance(search_queries, str):
        search_queries = [search_queries]
    
    # Define user agent generator
    def get_useragent():
        """Generates a random user agent string."""
        lynx_version = f"Lynx/{random.randint(2, 3)}.{random.randint(8, 9)}.{random.randint(0, 2)}"
        libwww_version = f"libwww-FM/{random.randint(2, 3)}.{random.randint(13, 15)}"
        ssl_mm_version = f"SSL-MM/{random.randint(1, 2)}.{random.randint(3, 5)}"
        openssl_version = f"OpenSSL/{random.randint(1, 3)}.{random.randint(0, 4)}.{random.randint(0, 9)}"
        return f"{lynx_version} {libwww_version} {ssl_mm_version} {openssl_version}"
    
    # Create executor for running synchronous operations
    executor = None if use_api else concurrent.futures.ThreadPoolExecutor(max_workers=5)
    
    # Use a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(5 if use_api else 2)
    
    async def search_single_query(query):
        async with semaphore:
            try:
                results = []
                
                # API-based search
                if use_api:
                    # The API returns up to 10 results per request
                    for start_index in range(1, max_results + 1, 10):
                        # Calculate how many results to request in this batch
                        num = min(10, max_results - (start_index - 1))
                        
                        # Make request to Google Custom Search API
                        params = {
                            'q': query,
                            'key': api_key,
                            'cx': cx,
                            'start': start_index,
                            'num': num
                        }
                        print(f"Requesting {num} results for '{query}' from Google API...")

                        async with aiohttp.ClientSession() as session:
                            async with session.get('https://www.googleapis.com/customsearch/v1', params=params) as response:
                                if response.status != 200:
                                    error_text = await response.text()
                                    print(f"API error: {response.status}, {error_text}")
                                    break
                                    
                                data = await response.json()
                                
                                # Process search results
                                for item in data.get('items', []):
                                    result = {
                                        "title": item.get('title', ''),
                                        "url": item.get('link', ''),
                                        "content": item.get('snippet', ''),
                                        "score": None,
                                        "raw_content": item.get('snippet', '')
                                    }
                                    results.append(result)
                        
                        # Respect API quota with a small delay
                        await asyncio.sleep(0.2)
                        
                        # If we didn't get a full page of results, no need to request more
                        if not data.get('items') or len(data.get('items', [])) < num:
                            break
                
                # Web scraping based search
                else:
                    # Add delay between requests
                    await asyncio.sleep(0.5 + random.random() * 1.5)
                    print(f"Scraping Google for '{query}'...")

                    # Define scraping function
                    def google_search(query, max_results):
                        try:
                            lang = "en"
                            safe = "active"
                            start = 0
                            fetched_results = 0
                            fetched_links = set()
                            search_results = []
                            
                            while fetched_results < max_results:
                                # Send request to Google
                                resp = requests.get(
                                    url="https://www.google.com/search",
                                    headers={
                                        "User-Agent": get_useragent(),
                                        "Accept": "*/*"
                                    },
                                    params={
                                        "q": query,
                                        "num": max_results + 2,
                                        "hl": lang,
                                        "start": start,
                                        "safe": safe,
                                    },
                                    cookies = {
                                        'CONSENT': 'PENDING+987',  # Bypasses the consent page
                                        'SOCS': 'CAESHAgBEhIaAB',
                                    }
                                )
                                resp.raise_for_status()
                                
                                # Parse results
                                soup = BeautifulSoup(resp.text, "html.parser")
                                result_block = soup.find_all("div", class_="ezO2md")
                                new_results = 0
                                
                                for result in result_block:
                                    link_tag = result.find("a", href=True)
                                    title_tag = link_tag.find("span", class_="CVA68e") if link_tag else None
                                    description_tag = result.find("span", class_="FrIlee")
                                    
                                    if link_tag and title_tag and description_tag:
                                        link = unquote(link_tag["href"].split("&")[0].replace("/url?q=", ""))
                                        
                                        if link in fetched_links:
                                            continue
                                        
                                        fetched_links.add(link)
                                        title = title_tag.text
                                        description = description_tag.text
                                        
                                        # Store result in the same format as the API results
                                        search_results.append({
                                            "title": title,
                                            "url": link,
                                            "content": description,
                                            "score": None,
                                            "raw_content": description
                                        })
                                        
                                        fetched_results += 1
                                        new_results += 1
                                        
                                        if fetched_results >= max_results:
                                            break
                                
                                if new_results == 0:
                                    break
                                    
                                start += 10
                                time.sleep(1)  # Delay between pages
                            
                            return search_results
                                
                        except Exception as e:
                            print(f"Error in Google search for '{query}': {str(e)}")
                            return []
                    
                    # Execute search in thread pool
                    loop = asyncio.get_running_loop()
                    search_results = await loop.run_in_executor(
                        executor, 
                        lambda: google_search(query, max_results)
                    )
                    
                    # Process the results
                    results = search_results
                
                # If requested, fetch full page content asynchronously (for both API and web scraping)
                if include_raw_content and results:
                    content_semaphore = asyncio.Semaphore(3)
                    
                    async with aiohttp.ClientSession() as session:
                        fetch_tasks = []
                        
                        async def fetch_full_content(result):
                            async with content_semaphore:
                                url = result['url']
                                headers = {
                                    'User-Agent': get_useragent(),
                                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
                                }
                                
                                try:
                                    await asyncio.sleep(0.2 + random.random() * 0.6)
                                    async with session.get(url, headers=headers, timeout=10) as response:
                                        if response.status == 200:
                                            # Check content type to handle binary files
                                            content_type = response.headers.get('Content-Type', '').lower()
                                            
                                            # Handle PDFs and other binary files
                                            if 'application/pdf' in content_type or 'application/octet-stream' in content_type:
                                                # For PDFs, indicate that content is binary and not parsed
                                                result['raw_content'] = f"[Binary content: {content_type}. Content extraction not supported for this file type.]"
                                            else:
                                                try:
                                                    # Try to decode as UTF-8 with replacements for non-UTF8 characters
                                                    html = await response.text(errors='replace')
                                                    soup = BeautifulSoup(html, 'html.parser')
                                                    result['raw_content'] = soup.get_text()
                                                except UnicodeDecodeError as ude:
                                                    # Fallback if we still have decoding issues
                                                    result['raw_content'] = f"[Could not decode content: {str(ude)}]"
                                except Exception as e:
                                    print(f"Warning: Failed to fetch content for {url}: {str(e)}")
                                    result['raw_content'] = f"[Error fetching content: {str(e)}]"
                                return result
                        
                        for result in results:
                            fetch_tasks.append(fetch_full_content(result))
                        
                        updated_results = await asyncio.gather(*fetch_tasks)
                        results = updated_results
                        print(f"Fetched full content for {len(results)} results")
                
                return {
                    "query": query,
                    "follow_up_questions": None,
                    "answer": None,
                    "images": [],
                    "results": results
                }
            except Exception as e:
                print(f"Error in Google search for query '{query}': {str(e)}")
                return {
                    "query": query,
                    "follow_up_questions": None,
                    "answer": None,
                    "images": [],
                    "results": []
                }
    
    try:
        # Create tasks for all search queries
        search_tasks = [search_single_query(query) for query in search_queries]
        
        # Execute all searches concurrently
        search_results = await asyncio.gather(*search_tasks)
        
        return search_results
    finally:
        # Only shut down executor if it was created
        if executor:
            executor.shutdown(wait=False)



async def select_and_execute_search(search_api: str, query_list: list[str], params_to_pass: dict) -> str:
    """Select and execute the appropriate search API.
    
    Args:
        search_api: Name of the search API to use
        query_list: List of search queries to execute
        params_to_pass: Parameters to pass to the search API
        
    Returns:
        Formatted string containing search results
        
    Raises:
        ValueError: If an unsupported search API is specified
    """
    if search_api == "tavily":
        search_results = await tavily_search_async(query_list, **params_to_pass)
        return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000, include_raw_content=False)
    elif search_api == "perplexity":
        search_results = perplexity_search(query_list, **params_to_pass)
        return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000)
    elif search_api == "exa":
        search_results = await exa_search(query_list, **params_to_pass)
        return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000)
    elif search_api == "arxiv":
        search_results = await arxiv_search_async(query_list, **params_to_pass)
        return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000)
    elif search_api == "pubmed":
        search_results = await pubmed_search_async(query_list, **params_to_pass)
        return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000)
    elif search_api == "linkup":
        search_results = await linkup_search(query_list, **params_to_pass)
        return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000)
    elif search_api == "duckduckgo":
        search_results = await duckduckgo_search(query_list)
        return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000)
    elif search_api == "googlesearch":
        search_results = await google_search_async(query_list, **params_to_pass)
        return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000)
    else:
        raise ValueError(f"Unsupported search API: {search_api}")



---
File: /.env.example
---

OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-xxx
TAVILY_API_KEY=xxx
GROQ_API_KEY=xxx
PERPLEXITY_API_KEY=xxx
LINKUP_API_KEY=xxx



---
File: /langgraph.json
---

{
    "dockerfile_lines": [],
    "graphs": {
      "open_deep_research": "./src/open_deep_research/graph.py:graph"
    },
    "python_version": "3.11",
    "env": "./.env",
    "dependencies": [
      "."
    ]
  }


---
File: /LICENSE
---

MIT License

Copyright (c) 2025 LangChain

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


---
File: /pyproject.toml
---

[project]
name = "open_deep_research"
version = "0.0.10"
description = "Planning, research, and report generation."
authors = [
    { name = "Lance Martin" }
]
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.9"
dependencies = [
    "langgraph>=0.2.55",
    "langchain-community>=0.3.9",
    "langchain-openai>=0.3.7",
    "langchain-anthropic>=0.3.9",
    "openai>=1.61.0",
    "tavily-python>=0.5.0",
    "langchain-groq>=0.2.4",
    "arxiv>=2.1.3",
    "pymupdf>=1.25.3",
    "xmltodict>=0.14.2",
    "linkup-sdk>=0.2.3",
    "duckduckgo-search>=3.0.0",
    "exa-py>=1.8.8",
    "requests>=2.32.3",
    "beautifulsoup4==4.13.3",
    "langchain-deepseek>=0.1.2",
    "python-dotenv==1.0.1",
]

[project.optional-dependencies]
dev = ["mypy>=1.11.1", "ruff>=0.6.1"]

[build-system]
requires = ["setuptools>=73.0.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["open_deep_research"]

[tool.setuptools.package-dir]
"open_deep_research" = "src/open_deep_research"

[tool.setuptools.package-data]
"*" = ["py.typed"]

[tool.ruff]
lint.select = [
    "E",    # pycodestyle
    "F",    # pyflakes
    "I",    # isort
    "D",    # pydocstyle
    "D401", # First line should be in imperative mood
    "T201",
    "UP",
]
lint.ignore = [
    "UP006",
    "UP007",
    "UP035",
    "D417",
    "E501",
]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["D", "UP"]

[tool.ruff.lint.pydocstyle]
convention = "google"



---
File: /README.md
---

# Open Deep Research
 
Open Deep Research is an open source assistant that automates research and produces customizable reports on any topic. It allows you to customize the research and writing process with specific models, prompts, report structure, and search tools. 

![report-generation](https://github.com/user-attachments/assets/6595d5cd-c981-43ec-8e8b-209e4fefc596)

## 🚀 Quickstart

Ensure you have API keys set for your desired search tools and models.

Available search tools:

* [Tavily API](https://tavily.com/) - General web search
* [Perplexity API](https://www.perplexity.ai/hub/blog/introducing-the-sonar-pro-api) - General web search
* [Exa API](https://exa.ai/) - Powerful neural search for web content
* [ArXiv](https://arxiv.org/) - Academic papers in physics, mathematics, computer science, and more
* [PubMed](https://pubmed.ncbi.nlm.nih.gov/) - Biomedical literature from MEDLINE, life science journals, and online books
* [Linkup API](https://www.linkup.so/) - General web search
* [DuckDuckGo API](https://duckduckgo.com/) - General web search
* [Google Search API/Scrapper](https://google.com/) - Create custom search engine [here](https://programmablesearchengine.google.com/controlpanel/all) and get API key [here](https://developers.google.com/custom-search/v1/introduction)

Open Deep Research uses a planner LLM for report planning and a writer LLM for report writing: 

* You can select any model that is integrated [with the `init_chat_model()` API](https://python.langchain.com/docs/how_to/chat_models_universal_init/)
* See full list of supported integrations [here](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html)

### Using the package

```bash
pip install open-deep-research
```

As mentioned above, ensure API keys for LLMs and search tools are set: 
```bash
export TAVILY_API_KEY=<your_tavily_api_key>
export ANTHROPIC_API_KEY=<your_anthropic_api_key>
```

See [src/open_deep_research/graph.ipynb](src/open_deep_research/graph.ipynb) for example usage in a Jupyter notebook:

Compile the graph:
```python
from langgraph.checkpoint.memory import MemorySaver
from open_deep_research.graph import builder
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
```

Run the graph with a desired topic and configuration:
```python
import uuid 
thread = {"configurable": {"thread_id": str(uuid.uuid4()),
                           "search_api": "tavily",
                           "planner_provider": "anthropic",
                           "planner_model": "claude-3-7-sonnet-latest",
                           "writer_provider": "anthropic",
                           "writer_model": "claude-3-5-sonnet-latest",
                           "max_search_depth": 1,
                           }}

topic = "Overview of the AI inference market with focus on Fireworks, Together.ai, Groq"
async for event in graph.astream({"topic":topic,}, thread, stream_mode="updates"):
    print(event)
```

The graph will stop when the report plan is generated, and you can pass feedback to update the report plan:
```python
from langgraph.types import Command
async for event in graph.astream(Command(resume="Include a revenue estimate (ARR) in the sections"), thread, stream_mode="updates"):
    print(event)
```

When you are satisfied with the report plan, you can pass `True` to proceed to report generation:
```python
async for event in graph.astream(Command(resume=True), thread, stream_mode="updates"):
    print(event)
```

### Running LangGraph Studio UI locally

Clone the repository:
```bash
git clone https://github.com/langchain-ai/open_deep_research.git
cd open_deep_research
```

Then edit the `.env` file to customize the environment variables according to your needs. These environment variables control the model selection, search tools, and other configuration settings. When you run the application, these values will be automatically loaded via `python-dotenv` (because `langgraph.json` point to the "env" file).
```bash
cp .env.example .env
```

Set whatever APIs needed for your model and search tools.

Here are examples for several of the model and tool integrations available:
```bash
export TAVILY_API_KEY=<your_tavily_api_key>
export ANTHROPIC_API_KEY=<your_anthropic_api_key>
export OPENAI_API_KEY=<your_openai_api_key>
export PERPLEXITY_API_KEY=<your_perplexity_api_key>
export EXA_API_KEY=<your_exa_api_key>
export PUBMED_API_KEY=<your_pubmed_api_key>
export PUBMED_EMAIL=<your_email@example.com>
export LINKUP_API_KEY=<your_linkup_api_key>
export GOOGLE_API_KEY=<your_google_api_key>
export GOOGLE_CX=<your_google_custom_search_engine_id>
```

Launch the assistant with the LangGraph server locally, which will open in your browser:

#### Mac

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and start the LangGraph server
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev
```

#### Windows / Linux

```powershell
# Install dependencies 
pip install -e .
pip install -U "langgraph-cli[inmem]" 

# Start the LangGraph server
langgraph dev
```

Use this to open the Studio UI:
```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

(1) Provide a `Topic` and hit `Submit`:

<img width="1326" alt="input" src="https://github.com/user-attachments/assets/de264b1b-8ea5-4090-8e72-e1ef1230262f" />

(2) This will generate a report plan and present it to the user for review.

(3) We can pass a string (`"..."`) with feedback to regenerate the plan based on the feedback.

<img width="1326" alt="feedback" src="https://github.com/user-attachments/assets/c308e888-4642-4c74-bc78-76576a2da919" />

(4) Or, we can just pass `true` to accept the plan.

<img width="1480" alt="accept" src="https://github.com/user-attachments/assets/ddeeb33b-fdce-494f-af8b-bd2acc1cef06" />

(5) Once accepted, the report sections will be generated.

<img width="1326" alt="report_gen" src="https://github.com/user-attachments/assets/74ff01cc-e7ed-47b8-bd0c-4ef615253c46" />

The report is produced as markdown.

<img width="1326" alt="report" src="https://github.com/user-attachments/assets/92d9f7b7-3aea-4025-be99-7fb0d4b47289" />

## 📖 Customizing the report

You can customize the research assistant's behavior through several parameters:

- `report_structure`: Define a custom structure for your report (defaults to a standard research report format)
- `number_of_queries`: Number of search queries to generate per section (default: 2)
- `max_search_depth`: Maximum number of reflection and search iterations (default: 2)
- `planner_provider`: Model provider for planning phase (default: "anthropic", but can be any provider from supported integrations with `init_chat_model` as listed [here](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html))
- `planner_model`: Specific model for planning (default: "claude-3-7-sonnet-latest")
- `writer_provider`: Model provider for writing phase (default: "anthropic", but can be any provider from supported integrations with `init_chat_model` as listed [here](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html))
- `writer_model`: Model for writing the report (default: "claude-3-5-sonnet-latest")
- `search_api`: API to use for web searches (default: "tavily", options include "perplexity", "exa", "arxiv", "pubmed", "linkup")

These configurations allow you to fine-tune the research process based on your needs, from adjusting the depth of research to selecting specific AI models for different phases of report generation.

### Search API Configuration

Not all search APIs support additional configuration parameters. Here are the ones that do:

- **Exa**: `max_characters`, `num_results`, `include_domains`, `exclude_domains`, `subpages`
  - Note: `include_domains` and `exclude_domains` cannot be used together
  - Particularly useful when you need to narrow your research to specific trusted sources, ensure information accuracy, or when your research requires using specified domains (e.g., academic journals, government sites)
  - Provides AI-generated summaries tailored to your specific query, making it easier to extract relevant information from search results
- **ArXiv**: `load_max_docs`, `get_full_documents`, `load_all_available_meta`
- **PubMed**: `top_k_results`, `email`, `api_key`, `doc_content_chars_max`
- **Linkup**: `depth`

Example with Exa configuration:
```python
thread = {"configurable": {"thread_id": str(uuid.uuid4()),
                           "search_api": "exa",
                           "search_api_config": {
                               "num_results": 5,
                               "include_domains": ["nature.com", "sciencedirect.com"]
                           },
                           # Other configuration...
                           }}
```

### Model Considerations

(1) You can pass any planner and writer models that are integrated [with the `init_chat_model()` API](https://python.langchain.com/docs/how_to/chat_models_universal_init/). See full list of supported integrations [here](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html).

(2) **The planner and writer models need to support structured outputs**: Check whether structured outputs are supported by the model you are using [here](https://python.langchain.com/docs/integrations/chat/).

(3) With Groq, there are token per minute (TPM) limits if you are on the `on_demand` service tier:
- The `on_demand` service tier has a limit of `6000 TPM`
- You will want a [paid plan](https://github.com/cline/cline/issues/47#issuecomment-2640992272) for section writing with Groq models

(4) `deepseek-R1` [is not strong at function calling](https://api-docs.deepseek.com/guides/reasoning_model), which the assistant uses to generate structured outputs for report sections and report section grading. See example traces [here](https://smith.langchain.com/public/07d53997-4a6d-4ea8-9a1f-064a85cd6072/r).  
- Consider providers that are strong at function calling such as OpenAI, Anthropic, and certain OSS models like Groq's `llama-3.3-70b-versatile`.
- If you see the following error, it is likely due to the model not being able to produce structured outputs (see [trace](https://smith.langchain.com/public/8a6da065-3b8b-4a92-8df7-5468da336cbe/r)):
```
groq.APIError: Failed to call a function. Please adjust your prompt. See 'failed_generation' for more details.
```

## How it works
   
1. `Plan and Execute` - Open Deep Research follows a [plan-and-execute workflow](https://github.com/assafelovic/gpt-researcher) that separates planning from research, allowing for human-in-the-loop approval of a report plan before the more time-consuming research phase. It uses, by default, a [reasoning model](https://www.youtube.com/watch?v=f0RbwrBcFmc) to plan the report sections. During this phase, it uses web search to gather general information about the report topic to help in planning the report sections. But, it also accepts a report structure from the user to help guide the report sections as well as human feedback on the report plan.
   
2. `Research and Write` - Each section of the report is written in parallel. The research assistant uses web search via [Tavily API](https://tavily.com/), [Perplexity](https://www.perplexity.ai/hub/blog/introducing-the-sonar-pro-api), [Exa](https://exa.ai/), [ArXiv](https://arxiv.org/), [PubMed](https://pubmed.ncbi.nlm.nih.gov/) or [Linkup](https://www.linkup.so/) to gather information about each section topic. It will reflect on each report section and suggest follow-up questions for web search. This "depth" of research will proceed for any many iterations as the user wants. Any final sections, such as introductions and conclusions, are written after the main body of the report is written, which helps ensure that the report is cohesive and coherent. The planner determines main body versus final sections during the planning phase.

3. `Managing different types` - Open Deep Research is built on LangGraph, which has native support for configuration management [using assistants](https://langchain-ai.github.io/langgraph/concepts/assistants/). The report `structure` is a field in the graph configuration, which allows users to create different assistants for different types of reports. 

## UX

### Local deployment

Follow the [quickstart](#-quickstart) to start LangGraph server locally.

### Hosted deployment
 
You can easily deploy to [LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts/#deployment-options). 

