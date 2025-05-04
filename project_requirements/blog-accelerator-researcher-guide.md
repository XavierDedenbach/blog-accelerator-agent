# Blog Accelerator Researcher Agent Guide

This comprehensive guide will help you set up and use the Blog Accelerator Researcher Agent, a powerful tool designed to enhance your blog writing process by providing in-depth research and analysis on your chosen topics.

## What's New

**OpenRouter Fallback Integration**:
- The system now automatically falls back to OpenRouter when OpenAI API rate limits are reached
- This ensures continuous processing even during heavy usage periods
- Proper rate limiting is maintained for both services (1 req/s for OpenAI, 5 req/s for OpenRouter)
- API calls are logged with a clear indicator when fallback is activated

## Overview

The Blog Accelerator Researcher Agent analyzes your blog post drafts and generates comprehensive research data including:

1. Industry challenges and their components
2. Solution arguments (both pro and counter)
3. Historical and future paradigm analysis
4. Audience segment identification with tailored strategies
5. Analogies to explain complex concepts
6. Automatic fallback to OpenRouter when OpenAI API rate limits are reached

This research data can then be used to enhance your blog posts, making them more informative, engaging, and valuable to your target audience.

## Prerequisites

Before you can run the Researcher Agent, you'll need:

1. Python 3.8 or higher installed on your system
2. MongoDB installed and running (for storing research data)
3. API keys for:
   - OpenAI API (for language model processing)
   - Brave Search API (for citation search)
   - OpenRouter API (as fallback when OpenAI quota is exceeded)
4. A blog post draft in markdown format

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/blog-accelerator-agent.git
cd blog-accelerator-agent
```

2. Set up a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file at the root of the project with your API keys:
```
MONGODB_URI=mongodb://localhost:27017
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
BRAVE_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## Environment Setup

After installing dependencies, you should run the setup script to prepare your environment for the researcher agent:

```bash
python setup_research.py
```

This script will:
1. Create necessary directories for storing research data
2. Generate default configuration files if they don't exist
3. Check if all required Python dependencies are installed
4. Verify your API keys are properly configured

If you want the script to automatically install any missing dependencies, run:

```bash
python setup_research.py --install-deps
```

The setup script should be run:
- After initial installation
- When updating to a new version of the agent
- If you encounter errors related to missing directories or configurations
- When setting up on a new machine

After running the setup script, review the output to ensure all requirements are met before proceeding to use the researcher agent.

## Starting the API Server

**CRITICAL**: Before running the researcher agent, you must start the API server in a separate terminal:

```bash
PYTHONPATH=$(pwd) MONGODB_URI="mongodb://localhost:27017" uvicorn api.main:app --host 0.0.0.0 --port 8080
```

This command:
- Sets the Python path to the current directory
- Configures the MongoDB connection URI
- Starts the FastAPI server using Uvicorn
- Makes the API available on all network interfaces (0.0.0.0) on port 8080

The API server provides essential services that the researcher agent depends on for processing blog content, making external API calls, and managing the research workflow. The researcher agent will not function properly without this server running.

Make sure MongoDB is running at the specified URI before starting the API server. The API server requires a connection to MongoDB to store and retrieve research data.

Leave this terminal window open and running while you use the researcher agent in another terminal window.

## API Rate Limiting

The researcher agent implements intelligent rate limiting to prevent API overages:

| API Service | Rate Limit | Notes |
|-------------|------------|-------|
| OpenAI API | 1 request per second | Primary service |
| Brave Search API | 20 requests per second | Used for citations |
| OpenRouter API | 5 requests per second | Fallback mechanism |

**New Feature**: The agent will automatically switch to OpenRouter when OpenAI API calls fail due to rate limiting or quota exceedance, ensuring uninterrupted processing of your blog content.

## Running the Researcher Agent

### Running with Your Own Blog Post (Requires API Keys)

To process your own blog post with live API calls:

```bash
python run_researcher_with_env.py path/to/your/blog.md
```

The researcher agent will:
1. Load environment variables from your `.env` file
2. Process your blog post
3. Generate and display comprehensive research data
4. Save the research data to MongoDB
5. Automatically switch to OpenRouter if OpenAI API rate limits are reached

#### Advanced Usage

For more control over the research process, you can pass additional arguments:

```bash
python run_researcher_with_env.py path/to/your/blog.md --verbose --max-sources 5 --audience-segments 3
```

Options:
- `--verbose`: Enable detailed logging
- `--max-sources`: Maximum number of sources to retrieve for each challenge (default: 3)
- `--audience-segments`: Number of audience segments to identify (default: 5)

## OpenRouter Fallback Feature

### How it Works

1. The agent attempts to use OpenAI API for all language model requests
2. If an OpenAI API call fails with a 429 status code (rate limit exceeded):
   - The agent logs the failure
   - Checks if an OpenRouter API key is available
   - Automatically switches to OpenRouter for future requests
   - Maintains proper rate limiting (5 requests per second)
   - Clearly logs that the fallback has been activated

### Setting Up OpenRouter

1. Sign up for an account at [OpenRouter](https://openrouter.ai/)
2. Get your API key from the dashboard
3. Add it to your `.env` file:
```
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Monitoring Fallback Events

When running with the `--verbose` flag, you'll see log messages indicating when the fallback is activated:

```
INFO - OpenAI API call failed with status 429 - Rate limit exceeded
INFO - Switching to OpenRouter fallback...
INFO - Successfully set up OpenRouter fallback for IndustryAnalysis
INFO - Continuing research with OpenRouter...
```

## Troubleshooting

### Common Issues

1. **API Keys Not Found**:
   - Make sure your `.env` file is in the project root directory
   - Check that the API key variable names match exactly as shown above

2. **MongoDB Connection Issues**:
   - Ensure MongoDB is installed and running
   - Verify the connection string in your `.env` file

3. **Rate Limit Errors**:
   - These are now handled automatically with the OpenRouter fallback
   - If you're still seeing issues, check if your OpenRouter API key is valid
   - You can reduce the number of concurrent requests by using the `--sequential` flag

4. **Timeout Errors**:
   - The research process can take time, especially for longer blog posts
   - Try running with a smaller number of sources using `--max-sources 2`

### Logging

Running with the `--verbose` flag will provide detailed logs, which can be helpful for diagnosing issues:

```bash
python run_researcher_with_env.py path/to/your/blog.md --verbose
```

## Understanding the Output

The research data is organized into several sections:

1. **Industry Analysis**:
   - Challenges facing the industry related to your blog topic
   - Components of each challenge (risk factors, slowdowns, costs, inefficiencies)
   - Relevant sources and citations

2. **Solution Analysis**:
   - Pro arguments supporting your blog's perspective
   - Counter arguments to consider and address
   - Metrics for evaluating potential solutions

3. **Paradigm Analysis**:
   - Historical context and lessons from past approaches
   - Future projections and emerging paradigms in the field

4. **Audience Analysis**:
   - Identified audience segments interested in your topic
   - Knowledge level and needs of each segment
   - Tailored communication strategies for each audience

5. **Analogy Generation**:
   - Analogies to explain complex concepts in your blog
   - Explanations of how each analogy relates to your topic
   - Visual asset suggestions for illustrating the analogies

## Example Usage

Here's a typical workflow for using the research agent to enhance your blog writing process:

1. **Research Preparation**:
   - Write a draft of your blog post in markdown format
   - Save it to a file (e.g., `my-blog-post.md`)

2. **Run the Researcher Agent**:
   ```bash
   python run_researcher_with_env.py my-blog-post.md
   ```

3. **Review the Research Data**:
   - Examine the industry challenges and their components
   - Consider the pro and counter arguments
   - Identify the most relevant audience segments

4. **Enhance Your Blog Post**:
   - Incorporate insights from the research data
   - Address counter arguments proactively
   - Use the generated analogies to explain complex concepts
   - Tailor your content to the identified audience segments

5. **New**: The system will automatically fallback to OpenRouter if you have configured an OpenRouter API key and encounter OpenAI API rate limits.

## Advanced Configuration

You can customize the behavior of the researcher agent by modifying the configuration files:

- `config/researcher_config.yaml`: General configuration parameters
- `config/llm_prompts.yaml`: Prompts used for different research components
- `config/api_config.yaml`: API-specific configuration including rate limits

## Contributing

We welcome contributions to improve the Blog Accelerator Researcher Agent! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

- This project uses various AI APIs including OpenAI and Brave Search
- Special thanks to all contributors who have helped improve this tool

## Docker Setup

In addition to the local setup instructions, you can also run the Blog Accelerator Researcher Agent using Docker, which provides a consistent environment with all dependencies pre-installed.

### Using the Docker Container

1. Make sure Docker is installed on your system:
   ```bash
   docker --version
   ```

2. Pull the Blog Accelerator Agent Docker image:
   ```bash
   docker pull yourusername/blog-accelerator-agent:latest
   ```

3. Create a `.env` file with your API keys as described in the Installation section.

4. Run the container, mounting your blog files and environment variables:
   ```bash
   docker run -it --rm \
     -v $(pwd)/.env:/app/.env \
     -v $(pwd)/blogs:/app/blogs \
     -p 8080:8080 \
     yourusername/blog-accelerator-agent:latest
   ```

5. In a separate terminal, run the researcher agent against your blog file:
   ```bash
   docker exec -it <container_id> python run_researcher_with_env.py /app/blogs/your-blog.md
   ```

### Docker Compose Setup

For a more comprehensive setup including MongoDB, you can use Docker Compose:

1. Create a `docker-compose.yml` file:
   ```yaml
   version: '3'
   services:
     app:
       image: yourusername/blog-accelerator-agent:latest
       ports:
         - "8080:8080"
       volumes:
         - ./.env:/app/.env
         - ./blogs:/app/blogs
       depends_on:
         - mongodb
       environment:
         - MONGODB_URI=mongodb://mongodb:27017
     
     mongodb:
       image: mongo:latest
       ports:
         - "27017:27017"
       volumes:
         - mongodb_data:/data/db
   
   volumes:
     mongodb_data:
   ```

2. Start the services:
   ```bash
   docker-compose up -d
   ```

3. Start the API server in the container:
   ```bash
   docker-compose exec app bash -c "PYTHONPATH=/app uvicorn api.main:app --host 0.0.0.0 --port 8080"
   ```

4. In a separate terminal, run the researcher agent:
   ```bash
   docker-compose exec app python run_researcher_with_env.py /app/blogs/your-blog.md
   ```

### Building Your Own Docker Image

If you want to build your own Docker image of the Blog Accelerator Agent:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/blog-accelerator-agent.git
   cd blog-accelerator-agent
   ```

2. Build the Docker image:
   ```bash
   docker build -t blog-accelerator-agent:latest .
   ```

3. Run the container as described above.

### Docker Environment Variables

The Docker container supports the following environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key
- `BRAVE_API_KEY`: Your Brave Search API key
- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `MONGODB_URI`: URI for connecting to MongoDB (default: mongodb://mongodb:27017)

You can set these directly in the `docker run` command or in your `docker-compose.yml` file instead of using a mounted `.env` file. 